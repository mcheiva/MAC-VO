"""
ONNX export for the MACVO FlowFormerCov frontend.

Pipeline (mirrors the modern PyTorch -> ONNX -> TensorRT path):
  1. Trace the FlowFormerCovWrapper into ONNX (opset 18, dynamic batch).
  2. Run onnx-simplifier to fold constants and prune redundant Cast / Reshape
     nodes left over from per-iteration .to(dtype) shuffles inside the decoder.
  3. Save with external-data so MACVO_FrontendCov.onnx + .onnx.data fit on disk
     and can be parsed by TensorRT 10's ONNX parser.

Wrapper conventions:
  * Inputs are (image_1, image_2) NCHW fp32 in [0, 255]; the wrapper does /255.
  * Outputs are (flow, cov) NCHW fp32; cov is already exp(2 * cov_raw).
  * Dynamic batch only; H=W=704 fixed (matches the trained checkpoint padding).
  * Trace at batch=1 to fit an 8 GB GPU during torch.onnx.export.

Pair this with MACVO_TRT_BuildCpp/ (or a Python tensorrt builder) to produce
a runnable engine. Opset 18 is the only safe choice for TRT 10.7's parser:
opset-20 graphs parse but are silently mis-compiled by TRT (fp32 plan EPE
jumps from 0.94 px -> 6.4 px) even though ONNXRuntime parses them correctly.
"""
import os

import onnx
import onnxsim
import torch

from Module.Network.FlowFormer.configs.submission import get_cfg
from Module.Network.FlowFormerCov import build_flowformer


class FlowFormerCovWrapper(torch.nn.Module):
    """
    Graph-side wrapper so the exported ONNX matches FlowFormerCovFrontend.inference():
    takes [0, 255] images, normalises, runs forward, returns (flow_up, exp(cov_up * 2)).
    """

    def __init__(self):
        super().__init__()

        cfg = get_cfg()
        cfg.latentcostformer.decoder_depth = 12
        # Build in fp32 end-to-end. trtexec --fp16 (or BuilderFlag.kFP16 in the
        # Python / C++ builder) lowers precision at compile time. Per MAC-VO
        # issue #18, TRT only safely compiles flow at builderOptimizationLevel=1
        # because higher levels do fusions tuned for [0,1] logits, not the flow
        # values in the hundreds. Keeping the export in fp32 maximises builder
        # freedom downstream.
        model = build_flowformer(cfg, torch.float32, torch.float32)
        device = torch.device("cuda")
        ckpt = torch.load("./Model/MACVO_FrontendCov.pth",
                          map_location=device, weights_only=True)

        model.eval()
        model.to(device)
        model.load_ddp_state_dict(ckpt)
        self.model = model

    def forward(self, image1, image2):
        image1 = image1.float() / 255.0
        image2 = image2.float() / 255.0

        # In eval mode FlowFormerCov returns ((flow_up, flow_residual), (cov_up, cov_residual)).
        # The MemoryCovDecoder skips intermediate-iteration upsamples in eval mode
        # (see covhead.py), so each list contains a single tensor.
        flow_predictions, cov_predictions = self.model(image1, image2)
        flow_up = flow_predictions[0]
        cov_up  = cov_predictions[0]
        # Force the final exp(2 * cov) into fp32. exp() in fp16 saturates around
        # 11 (max fp16 ~ 65504) so even small fp16 noise on cov_up blows up high
        # uncertainty pixels. The explicit .float() cast becomes an ONNX Cast
        # node, which TRT keeps in fp32 even when --fp16 is enabled.
        cov_fp32 = cov_up.float()
        return flow_up, torch.exp(cov_fp32 * 2.0)


def export(onnx_path: str = "MACVO_FrontendCov.onnx",
           opset_version: int = 18,
           run_onnxsim: bool = True) -> None:
    model = FlowFormerCovWrapper()

    H = W = 704
    batch_size = 1
    channels = 3
    # Trace at batch=1 to fit an 8 GB GPU. dynamic_axes makes the ONNX accept
    # any batch, and the TRT builder's optimization profile selects the
    # runtime batch range.
    inpA = torch.randn((batch_size, channels, H, W), dtype=torch.float32).cuda() * 255.0
    inpB = torch.randn((batch_size, channels, H, W), dtype=torch.float32).cuda() * 255.0

    print(f"[1/3] Tracing -> ONNX (opset={opset_version})...", flush=True)
    with torch.inference_mode():
        torch.onnx.export(
            model,
            (inpA, inpB),
            opset_version=opset_version,
            f=onnx_path,
            input_names=["image_1", "image_2"],
            output_names=["flow", "cov"],
            dynamic_axes={
                "image_1": {0: "batch"},
                "image_2": {0: "batch"},
                "flow":    {0: "batch"},
                "cov":     {0: "batch"},
            },
            export_params=True,
        )

    # Free GPU memory before loading the ONNX for simplification; onnxsim runs
    # CUDA shape inference internally and the wrapper's checkpoint is still held.
    del model
    torch.cuda.empty_cache()

    if not run_onnxsim:
        print(f"OK: {onnx_path} (no simplification)")
        return

    print(f"[2/3] Loading + simplifying with onnxsim...", flush=True)
    m = onnx.load(onnx_path, load_external_data=True)
    pre_n = len(m.graph.node)
    m_simp, ok = onnxsim.simplify(m)
    if not ok:
        raise RuntimeError("onnxsim.simplify failed validation")
    post_n = len(m_simp.graph.node)
    print(f"  nodes {pre_n} -> {post_n} ({pre_n - post_n} removed)")

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


if __name__ == "__main__":
    export()
