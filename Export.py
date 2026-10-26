"""
ONNX export for the MACVO FlowFormerCov frontend.

Two modes:
  fast     (fp16 encoder, bf16 decoder) baked into the ONNX. Pair with
           Build.py --mode fast (strongly-typed + weight streaming + opt-level 5)
           for the deployment plan. Default mode.
  precise  (fp32 encoder, fp32 decoder) baked. Reference-precision plan,
           closest possible match to PyTorch fp32. Pair with
           Build.py --mode precise (TF32 cleared) -- or with
           --stronglyTyped --allowWeightStreaming if you want fp32 to fit on
           an 8 GB GPU (otherwise OOM at runtime).

Wrapper conventions (must match the consumer pipeline):
  * Inputs are (image_1, image_2) NCHW fp32 in [0, 1]. The MACVO consumer
    (Python or the C++ MACVO_TRT runtime) divides by 255 itself before feeding
    the engine -- in-graph /255 + consumer /255 is catastrophic on real
    sequences (keypoint selector drops every match).
  * Outputs are (flow, flow_raw, cov, cov_raw). The C++ MACVO_TRT runtime requires `flow` and `cov`
    ([1, 2, 704, 704] fp32) and binds any other outputs (flow_raw / cov_raw, [1, 2, 88, 88]). For the shipped head (2
    channels, log sigma per axis) `cov` is the raw log-uncertainty and the
    consumer applies `exp(2 * cov)` itself. Same reasoning: in-graph exp plus
    consumer exp = double-exp explosion.
  * The Cholesky head (--ckpt, 3 channels) emits the
    covariance entries [sigma_uu, sigma_vv, sigma_uv] in fp32 in-graph
    (handoff 7.4): sigma^2 = exp(2 c) reaches ~1.6e5, beyond fp16. The
    consumer must then read 3 channels and skip its exp (handoff 7.6).
    --cov-layout log-sigma instead emits the shipped layout for any head:
    log sigma per axis of the marginal variances (sigma_uv dropped; it does
    not change MAC-VO's trajectories), so the engine drops into the shipped
    runtime unchanged.
  * Static batch 1 is the norm: inputs [1, 3, 704, 704], a plain static trace (no dynamic_axes). The pad
    zeros in Module/Network/FlowFormer/core/pad_helper.py are built as a one-element zero expanded to the pad
    shape, so the static trace exports them as runtime Expand nodes, which TensorRT keeps as in-kernel
    broadcasts. (They used to be a dense torch.zeros(...): a static trace exported literal zero Constants and
    Myelin then generated a ~3.7x slower kernel for the decoder's convex upsampler, +2.6% engine time; that was
    why the export traced with a dynamic batch axis and pinned the batch dim to 1 afterwards.) The plain static
    engine now benchmarks the same as the pinned one. --dynamic-batch keeps the dynamic batch axis (the old
    dynamic export, to reproduce older engines).
    H=W=704 fixed (matches the trained checkpoint padding).
  * Tracing at batch=1 fits an 8 GB GPU during torch.onnx.export.

Pipeline (per export run):
  1. Trace the FlowFormerCovWrapper into ONNX (opset 18, static batch 1; with --dynamic-batch a dynamic
     batch axis on all inputs / outputs, traced at batch 1).
  2. Only with --onnxsim: run onnx-simplifier's graph passes (no constant folding, see export()) and save
     with external data. Off by default: it measured +0.7% engine time (see export()).

Equivalent builder command for fast mode (static ONNX, so no min/opt/max shape profile is needed;
weight streaming on, builder optimization level 5):
  trtexec --onnx=MACVO_FrontendCov.onnx --saveEngine=MACVO_FrontendCov.plan ^
    --stronglyTyped --allowWeightStreaming --builderOptimizationLevel=5
With --dynamic-batch the shape profile is needed in addition:
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
from Network.FlowFormerCov.covhead import cov_entries, marginal_log_sigma        # noqa: E402


_DTYPES = {
    "fast":    (torch.float16, torch.bfloat16),
    "precise": (torch.float32, torch.float32),
}

_DEFAULT_OUT = {
    "fast":    "MACVO_FrontendCov.onnx",
    "precise": "MACVO_FrontendCov_fp32.onnx",
}


_SHIPPED_CKPT = "./Model/MACVO_FrontendCov.pth"


class FlowFormerCovWrapper(torch.nn.Module):
    """
    Inputs: already-normalised images in [0, 1]. The consumer divides by 255.
    Outputs: shipped head -> raw log-uncertainty `cov`, the consumer applies `exp(2 * cov)`;
    other heads -> covariance entries [sigma_uu, sigma_vv, sigma_uv] in fp32.
    Input normalisation intentionally lives OUTSIDE the graph.
    """

    def __init__(self, mode: str = "fast", decoder_depth: int = 12, ckpt: str = _SHIPPED_CKPT, cov_layout: str = "auto"):
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
        state = torch.load(ckpt, map_location=device, weights_only=True)

        model.eval()
        model.to(device)
        model.load_ddp_state_dict(state)    # adapts the covariance head to the checkpoint's layout
        # Shipped layout: log sigma per axis, the MACVO_TRT runtime applies exp(2 * cov). cov_layout "log-sigma"
        # emits it for any head (marginal_log_sigma); "auto" keeps covariance entries for heads other than the shipped one.
        self.log_sigma = model.memory_decoder.log_sigma_output
        self.marginal = cov_layout == "log-sigma" and not self.log_sigma
        self.model = model

    def forward(self, image1, image2):
        # Inputs assumed in [0, 1] -- consumer pipeline does /255 itself.
        # Model returns ((flow_up, flow_residual), (cov_up, cov_residual)).
        # The C++ MACVO_TRT runtime requires `flow` and `cov` ([1, 2, 704, 704] fp32) and binds
        # `flow_raw` / `cov_raw` too.
        (flow_up, flow_raw), (cov_up, cov_raw) = self.model(image1, image2)
        if self.log_sigma:
            # Return raw log-uncertainty (NOT exp'd); consumer applies exp(2*cov).
            return flow_up, flow_raw.float(), cov_up.float(), cov_raw.float()
        if self.marginal:
            return flow_up, flow_raw.float(), marginal_log_sigma(cov_up.float()), marginal_log_sigma(cov_raw.float())
        return flow_up, flow_raw.float(), cov_entries(cov_up.float()), cov_entries(cov_raw.float())


def export(onnx_path: str | None = None,
           opset_version: int = 18,
           run_onnxsim: bool = False,
           mode: str = "fast",
           decoder_depth: int = 12,
           ckpt: str = _SHIPPED_CKPT,
           cov_layout: str = "auto",
           dynamic_batch: bool = False) -> None:
    if onnx_path is None:
        onnx_path = _DEFAULT_OUT[mode]

    model = FlowFormerCovWrapper(mode=mode, decoder_depth=decoder_depth, ckpt=ckpt, cov_layout=cov_layout)
    print(f"Checkpoint {ckpt}: cov = " + ("log sigma per axis, the consumer applies exp(2 * cov)" if model.log_sigma
                                          else "marginal log sigma per axis (shipped layout)" if model.marginal
                                          else "[sigma_uu, sigma_vv, sigma_uv] in fp32"), flush=True)

    H = W = 704
    batch_size = 1
    channels = 3
    # Trace at batch=1 to fit an 8 GB GPU during JIT trace. Plain static trace by default; --dynamic-batch adds
    # the dynamic batch axis.
    output_names = ["flow", "flow_raw", "cov", "cov_raw"]
    dynamic_axes = ({name: {0: "batch"} for name in ["image_1", "image_2"] + output_names}
                    if dynamic_batch else None)
    inpA = torch.randn((batch_size, channels, H, W), dtype=torch.float32).cuda()
    inpB = torch.randn((batch_size, channels, H, W), dtype=torch.float32).cuda()

    print(f"[1/3] Tracing -> ONNX (mode={mode}, opset={opset_version}, "
          f"{'dynamic batch axis' if dynamic_batch else 'static'} at batch 1)...", flush=True)
    with torch.inference_mode():
        # Warm-up: fills the size-keyed caches (coords_grid) and the no-grad vproj block-diagonal weight cache
        # (encoder.py _v_block_diag) so they trace as constants.
        model(inpA, inpB)
        torch.onnx.export(
            model,
            (inpA, inpB),
            opset_version=opset_version,
            f=onnx_path,
            input_names=["image_1", "image_2"],
            output_names=output_names,
            dynamic_axes=dynamic_axes,
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
    # onnxsim is OPT-IN (--onnxsim): it measured +0.7% engine time even with graph passes only. Its
    # duplicate-node merging collapses the identical cost_perceiver global-block positional-encoding
    # subgraphs, so Myelin reads a stored (1,7744,192) PE tensor instead of computing Cos/Sin in-kernel and
    # 24-36 BF16 gemm tactics get retimed. It does cut the node count ~3x (10.6k -> 3.7k).
    # Simplification method: onnxsim with constant folding OFF and check_n=0, i.e. only its onnx-optimizer
    # graph passes (Constant -> initializer, dead-code / duplicate-node elimination, nop Cast / Pad removal).
    # Why not onnxsim's constant folding: the fast-mode graph holds bf16 tensors (bf16 decoder weights and
    # Slice/Cast chains over them). onnxruntime can run those nodes but its Python API cannot return a
    # bfloat16 tensor ("No corresponding Numpy type for Tensor Type. bfloat16"), so onnxsim's fold fails on
    # the first bf16 constant ("failed to run Cast/Slice ...") and leaves a graph that is not topologically
    # sorted. onnxruntime also cannot load the graph (Conv with bf16 inputs is outside the ONNX Conv type
    # constraint), so check_n > 0 would fail too. TensorRT folds the remaining constants itself at build time.
    # "eliminate_consecutive_idempotent_ops" is skipped: it treats Reshape as idempotent and drops the first
    # of two consecutive Reshapes, which is wrong when the second shape uses 0 / -1 (copy / infer dims). It
    # broke the dynamic-batch graph (cost_perceiver_encoder: Conv got 2 channels instead of 4); the static
    # graph happened to survive. Found by running fp32 surrogates of both graphs in onnxruntime.
    # do_constant_folding=True in torch.onnx.export is not an option either: it still fails on the mixed
    # CUDA/CPU `cat` (verified on a static-trace export).
    try:
        m_simp, ok = onnxsim.simplify(m, skip_constant_folding=True, check_n=0,
                                      skipped_optimizers=["eliminate_consecutive_idempotent_ops"])
    except Exception as e:
        print(f"  onnxsim.simplify raised: {str(e).splitlines()[-1][:120]}", flush=True)
        m_simp, ok = None, False
    if ok:
        try:
            onnx.checker.check_model(m_simp)
        except Exception as e:
            print(f"  simplified graph fails onnx.checker: {str(e)[:120]}", flush=True)
            ok = False
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
    p.add_argument("--onnxsim", action="store_true",
                   help="Opt in to the onnxsim graph-pass simplification (graph passes only, no constant "
                        "folding); off by default because it measured +0.7%% engine time")
    p.add_argument("--decoder-depth", type=int, default=12,
                   help="Decoder iterations (weights are shared across iterations)")
    p.add_argument("--ckpt", default=_SHIPPED_CKPT,
                   help="Frontend checkpoint; any head but the shipped one exports covariance entries")
    p.add_argument("--cov-layout", choices=("auto", "log-sigma"), default="auto",
                   help="log-sigma: the shipped layout (log sigma per axis) for any head, a drop-in engine")
    p.add_argument("--dynamic-batch", action="store_true",
                   help="Keep the dynamic batch axis on all inputs and outputs (the old export, to reproduce older "
                        "engines); default is a plain static batch-1 trace")
    args = p.parse_args()

    export(onnx_path=args.onnx,
           opset_version=args.opset,
           run_onnxsim=args.onnxsim,
           mode=args.mode,
           decoder_depth=args.decoder_depth,
           ckpt=args.ckpt,
           cov_layout=args.cov_layout,
           dynamic_batch=args.dynamic_batch)
    return 0


if __name__ == "__main__":
    sys.exit(main())
