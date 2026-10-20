"""
ONNX export for the MACVO FlowFormerCov frontend.

Two modes:
  fast     (fp16 encoder, bf16 decoder) baked into the ONNX. Pair with
           Build.py --mode fast (strongly-typed + weight streaming + opt-level 1)
           for a deployment plan that matches the gold root plan within ~1.7%
           APE on plane_nose. Default mode.
  precise  (fp32 encoder, fp32 decoder) baked. Reference-precision plan,
           closest possible match to PyTorch fp32. Pair with
           Build.py --mode precise (TF32 cleared) -- or with
           --stronglyTyped --allowWeightStreaming if you want fp32 to fit on
           an 8 GB GPU at batch=2 (otherwise OOM at runtime).

Wrapper conventions (must match the consumer pipeline):
  * Inputs are (image_1, image_2) NCHW fp32 in [0, 1]. The MACVO consumer
    (Python or the C++ MACVO_TRT runtime) divides by 255 itself before feeding
    the engine -- in-graph /255 + consumer /255 is catastrophic on real
    sequences (keypoint selector drops every match).
  * Outputs are (flow, flow_raw, cov, cov_raw). `cov` is the raw log-uncertainty;
    the consumer applies `exp(2 * cov)` itself. Same reasoning: in-graph exp
    plus consumer exp = double-exp explosion.
  * Dynamic batch only; H=W=704 fixed (matches the trained checkpoint padding).
  * Trace at batch=1 to fit an 8 GB GPU during torch.onnx.export.

Pipeline (per export run):
  1. Trace the FlowFormerCovWrapper into ONNX (opset 18, dynamic batch).
  2. Run onnx-simplifier to fold constants and prune redundant Cast / Reshape
     nodes from the per-iteration .to(dtype) shuffles inside the decoder.
  3. Save with external-data so file pair fits TRT 10's parser.

Equivalent trtexec for fast mode (matches gold root plan):
  trtexec --onnx=MACVO_FrontendCov.onnx --saveEngine=MACVO_FrontendCov.plan ^
    --stronglyTyped --allowWeightStreaming --builderOptimizationLevel=1 ^
    --minShapes=image_1:1x3x704x704,image_2:1x3x704x704 ^
    --optShapes=image_1:2x3x704x704,image_2:2x3x704x704 ^
    --maxShapes=image_1:2x3x704x704,image_2:2x3x704x704

Opset 18 is the only safe choice for TRT 10.7's parser: opset-20 graphs parse
but are silently mis-compiled (fp32 plan EPE 0.94 px -> 6.4 px) even though
ONNXRuntime parses them correctly.

Validate an export against the fp32 reference set with Scripts/Parity/compare.py.
"""
import argparse
import os
import sys

import onnx
import onnxsim
import torch

# Import the network package directly: `import Module` pulls DataLoader -> scipy, which is
# binary-incompatible with numpy>=2 in the export environment and is not needed here.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "Module"))
from Network.FlowFormer.configs.submission import get_cfg   # noqa: E402
from Network.FlowFormerCov import build_flowformer           # noqa: E402


_DTYPES = {
    "fast":    (torch.float16, torch.bfloat16),
    "precise": (torch.float32, torch.float32),
}

_DEFAULT_OUT = {
    "fast":    "MACVO_FrontendCov.onnx",
    "precise": "MACVO_FrontendCov_fp32.onnx",
}


class FlowFormerCovWrapper(torch.nn.Module):
    """
    Inputs: already-normalised images in [0, 1]. The consumer divides by 255.
    Outputs: raw log-uncertainty `cov`; the consumer applies `exp(2 * cov)`.
    Both pre/post-process intentionally live OUTSIDE the graph.
    """

    def __init__(self, mode: str = "fast", decoder_depth: int = 12):
        super().__init__()
        if mode not in _DTYPES:
            raise ValueError(f"Unknown mode {mode!r} (expected 'fast' or 'precise')")
        enc_dtype, dec_dtype = _DTYPES[mode]

        cfg = get_cfg()
        cfg.latentcostformer.decoder_depth = decoder_depth
        # Bake (encoder, decoder) build dtypes per mode.
        # fast:    fp16 / bf16 -- matches the recipe behind the gold root plan
        #          when paired with Build.py --mode fast.
        # precise: fp32 / fp32 -- TRT strongly-typed plan respects fp32 literally.
        model = build_flowformer(cfg, enc_dtype, dec_dtype)
        device = torch.device("cuda")
        ckpt = torch.load("./Model/MACVO_FrontendCov.pth",
                          map_location=device, weights_only=True)

        model.eval()
        model.to(device)
        model.load_ddp_state_dict(ckpt)
        # The outputs below and the MACVO_TRT runtime assume log sigma per axis (exp(2 * cov) downstream).
        decoder = model.memory_decoder
        assert decoder.recurrent and decoder.cov_update.cov_head.conv4.out_channels == 2, \
            "Cholesky / single-shot covariance head: emit [sigma_uu, sigma_vv, sigma_uv] in-graph first (handoff 7.4)"
        self.model = model

    def forward(self, image1, image2):
        # Inputs assumed in [0, 1] -- consumer pipeline does /255 itself.
        # Model returns ((flow_up, flow_residual), (cov_up, cov_residual)).
        # The C++ MACVO_TRT runtime binds all four output names; missing any
        # of `flow`, `flow_raw`, `cov`, `cov_raw` makes setOutputTensorAddress
        # error and the trajectory writer never runs.
        (flow_up, flow_raw), (cov_up, cov_raw) = self.model(image1, image2)
        # Return raw log-uncertainty (NOT exp'd); consumer applies exp(2*cov).
        return flow_up, flow_raw.float(), cov_up.float(), cov_raw.float()


def export(onnx_path: str | None = None,
           opset_version: int = 18,
           run_onnxsim: bool = True,
           mode: str = "fast",
           decoder_depth: int = 12) -> None:
    if onnx_path is None:
        onnx_path = _DEFAULT_OUT[mode]

    model = FlowFormerCovWrapper(mode=mode, decoder_depth=decoder_depth)

    H = W = 704
    batch_size = 1
    channels = 3
    # Trace at batch=1 to fit an 8 GB GPU during JIT trace. dynamic_axes makes
    # the ONNX accept any batch; the TRT builder's optimization profile selects
    # the runtime batch range.
    inpA = torch.randn((batch_size, channels, H, W), dtype=torch.float32).cuda()
    inpB = torch.randn((batch_size, channels, H, W), dtype=torch.float32).cuda()

    print(f"[1/3] Tracing -> ONNX (mode={mode}, opset={opset_version})...", flush=True)
    with torch.inference_mode():
        model(inpA, inpB)   # warm-up: fills size-keyed caches (coords_grid) so they trace as constants
        torch.onnx.export(
            model,
            (inpA, inpB),
            opset_version=opset_version,
            f=onnx_path,
            input_names=["image_1", "image_2"],
            output_names=["flow", "flow_raw", "cov", "cov_raw"],
            dynamic_axes={
                "image_1":  {0: "batch"},
                "image_2":  {0: "batch"},
                "flow":     {0: "batch"},
                "flow_raw": {0: "batch"},
                "cov":      {0: "batch"},
                "cov_raw":  {0: "batch"},
            },
            export_params=True,
            # The exporter's own folder re-runs weight-only subgraphs with parameters on the CPU and
            # other constants on CUDA and fails on the mixed `cat`; onnxsim / TensorRT fold instead.
            do_constant_folding=False,
        )

    # Free GPU memory before onnxsim's CUDA shape inference.
    del model
    torch.cuda.empty_cache()

    if not run_onnxsim:
        print(f"OK: {onnx_path} (no simplification)")
        return

    print(f"[2/3] Loading + simplifying with onnxsim...", flush=True)
    m = onnx.load(onnx_path, load_external_data=True)
    pre_n = len(m.graph.node)
    m_simp, ok = None, False
    for kw in ({}, {"skip_shape_inference": True}):
        try:
            m_simp, ok = onnxsim.simplify(m, **kw)
        except Exception as e:  # onnx-optimizer asserts on some of these graphs
            print(f"  onnxsim.simplify({kw}) raised: {str(e).splitlines()[-1][:120]}", flush=True)
            ok = False
        if ok:
            break
    if not ok:
        # TensorRT folds constants itself; the unsimplified graph parses and builds.
        print(f"WARN: onnxsim failed; keeping the unsimplified graph at {onnx_path}", flush=True)
        return
    post_n = len(m_simp.graph.node)
    print(f"  nodes {pre_n} -> {post_n} ({pre_n - post_n} removed)")
    # Op histogram: the TRT-unfriendly ones to watch are ScatterND, Loop/If, Pad, Tile,
    # excessive Cast/Concat/Reshape/Transpose counts.
    from collections import Counter
    hist = Counter(n.op_type for n in m_simp.graph.node)
    print("  ops: " + ", ".join(f"{k}={v}" for k, v in sorted(hist.items(), key=lambda kv: -kv[1])))

    print(f"[3/3] Saving simplified ONNX with external data...", flush=True)
    location = os.path.basename(onnx_path) + ".data"
    sidecar = os.path.join(os.path.dirname(onnx_path) or ".", location)
    if os.path.exists(sidecar):
        os.remove(sidecar)
    onnx.save(m_simp, onnx_path,
              save_as_external_data=True,
              all_tensors_to_one_file=True,
              location=location)
    print(f"OK: {onnx_path} (+ {location})")


def main() -> int:
    p = argparse.ArgumentParser(
        description="Export the MACVO FlowFormerCov frontend to ONNX")
    p.add_argument("--mode", choices=("fast", "precise"), default="fast",
                   help="fast = (fp16, bf16) baked dtypes (deployment plan); "
                        "precise = (fp32, fp32) baked dtypes (reference plan)")
    p.add_argument("--onnx", default=None,
                   help="Output ONNX path (default: derived from --mode)")
    p.add_argument("--opset", type=int, default=18,
                   help="ONNX opset version (18 is the only TRT 10.7-safe choice)")
    p.add_argument("--no-onnxsim", action="store_true",
                   help="Skip onnxsim post-pass")
    p.add_argument("--decoder-depth", type=int, default=12,
                   help="Decoder iterations (weights are shared across iterations)")
    args = p.parse_args()

    export(onnx_path=args.onnx,
           opset_version=args.opset,
           run_onnxsim=not args.no_onnxsim,
           mode=args.mode,
           decoder_depth=args.decoder_depth)
    return 0


if __name__ == "__main__":
    sys.exit(main())
