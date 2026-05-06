"""
ONNX -> TensorRT engine builder for the MACVO FlowFormerCov frontend.
Python mirror of MACVO_TRT_BuildCpp/. Produces a plan that loads in the same
TRT version as the Python `tensorrt` package (use the C++ tool when the
deployment runtime ships its own TRT).

Modes:
  fast     FP16 build, OBEY_PRECISION_CONSTRAINTS, cov-tail (/Mul, /Exp)
           pinned fp32, cov output fp32, profile 1x3xHxW / 2x3xHxW / 2x3xHxW.
           This is the deployment plan.
  precise  FP32 build, TF32 disabled to match PyTorch allow_tf32=False, batch=1
           only (8 GB VRAM cap). Reference plan for parity vs PyTorch.

Both modes use builderOptimizationLevel=1: per MAC-VO issue #18, higher levels
do fusions tuned for [0,1] logits and corrupt optical-flow values.
"""
import argparse
import sys

import tensorrt as trt


FP32_TAIL_NAMES = {"/Mul", "/Exp"}
INPUT_NAMES = ("image_1", "image_2")
COV_OUTPUT_NAME = "cov"


def build(onnx_path: str,
          plan_path: str,
          mode: str,
          min_batch: int,
          opt_batch: int,
          max_batch: int,
          channels: int,
          height: int,
          width: int,
          opt_level: int,
          verbose: bool) -> None:
    severity = trt.Logger.INFO if verbose else trt.Logger.WARNING
    logger = trt.Logger(severity)
    builder = trt.Builder(logger)
    network = builder.create_network(0)
    parser = trt.OnnxParser(network, logger)

    # parse_from_file (not parse(bytes)) so the parser resolves the external-
    # data sidecar (`*.onnx.data`) relative to the ONNX path, regardless of
    # the caller's CWD. The Export.py output uses external-data layout.
    if not parser.parse_from_file(onnx_path):
        for i in range(parser.num_errors):
            print(parser.get_error(i), file=sys.stderr)
        raise RuntimeError(f"ONNX parse failed: {onnx_path}")

    config = builder.create_builder_config()
    config.builder_optimization_level = opt_level

    if mode == "fast":
        config.set_flag(trt.BuilderFlag.FP16)
        # Honour the per-layer fp32 pins below; without this flag TRT may
        # silently downcast a pinned layer back to fp16 when fp16 is faster.
        config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)

        # Force the cov-branch tail (cov_up * 2 -> exp(...)) to fp32 so
        # exp() does not saturate the fp16 range (~65504). After onnxsim
        # the chain is just /Mul -> /Exp; if the export changes, run
        # MACVO_TRT_Inspect.py to find the new layer names.
        pinned = []
        for i in range(network.num_layers):
            layer = network.get_layer(i)
            if layer.name in FP32_TAIL_NAMES:
                layer.precision = trt.float32
                for j in range(layer.num_outputs):
                    layer.set_output_type(j, trt.float32)
                pinned.append(layer.name)
        print(f"Pinned {len(pinned)} tail layers to fp32: {pinned}", flush=True)

        for i in range(network.num_outputs):
            out = network.get_output(i)
            if out.name == COV_OUTPUT_NAME:
                out.dtype = trt.float32

    elif mode == "precise":
        # Disable TF32 so matmuls use full-mantissa fp32, matching PyTorch's
        # default allow_tf32=False. Without this, even the fp32 plan diverges
        # from fp32 PyTorch because TF32 truncates the mantissa to 10 bits.
        config.clear_flag(trt.BuilderFlag.TF32)
        # 8 GB cards OOM at batch>1 fp32 reference build.
        if max_batch > 1 or opt_batch > 1:
            print("[warn] precise mode: forcing batch=1 (8 GB OOM cap).",
                  file=sys.stderr, flush=True)
            min_batch = opt_batch = max_batch = 1

    else:
        raise ValueError(f"Unknown mode: {mode!r} (expected 'fast' or 'precise')")

    profile = builder.create_optimization_profile()
    for name in INPUT_NAMES:
        profile.set_shape(
            name,
            (min_batch, channels, height, width),
            (opt_batch, channels, height, width),
            (max_batch, channels, height, width),
        )
    config.add_optimization_profile(profile)

    print(f"Building TRT {trt.__version__} engine (mode={mode}). "
          f"Takes several minutes...", flush=True)
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("build_serialized_network returned None "
                           "(likely OOM or builder error)")

    plan_bytes = bytes(serialized)
    with open(plan_path, "wb") as f:
        f.write(plan_bytes)
    print(f"Wrote {plan_path}: {len(plan_bytes) / (1024 * 1024):.2f} MiB",
          flush=True)


def main() -> int:
    p = argparse.ArgumentParser(
        description="Build a TensorRT engine from the MACVO FlowFormerCov ONNX")
    p.add_argument("--onnx", default="MACVO_FrontendCov.onnx",
                   help="Input ONNX file")
    p.add_argument("--out", default=None,
                   help="Output plan path (default: derived from --mode)")
    p.add_argument("--mode", choices=("fast", "precise"), default="fast",
                   help="fast = FP16 deployment plan; precise = FP32 reference")
    p.add_argument("--min-batch", type=int, default=1)
    p.add_argument("--opt-batch", type=int, default=2)
    p.add_argument("--max-batch", type=int, default=2)
    p.add_argument("--height", type=int, default=704)
    p.add_argument("--width",  type=int, default=704)
    p.add_argument("--opt-level", type=int, default=1,
                   help="Builder optimization level (1 is the only safe value "
                        "for this network; per MAC-VO issue #18)")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    out = args.out or (
        "MACVO_FrontendCov.plan" if args.mode == "fast"
        else "MACVO_FrontendCov_fp32.plan"
    )

    build(
        onnx_path=args.onnx,
        plan_path=out,
        mode=args.mode,
        min_batch=args.min_batch,
        opt_batch=args.opt_batch,
        max_batch=args.max_batch,
        channels=3,
        height=args.height,
        width=args.width,
        opt_level=args.opt_level,
        verbose=args.verbose,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
