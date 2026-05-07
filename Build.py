"""
ONNX -> TensorRT engine builder for the MACVO FlowFormerCov frontend.
Python mirror of MACVO_TRT_BuildCpp/. Both modes build a strongly-typed
network so TRT respects the dtypes baked into the ONNX literally. Pair
with the matching --mode in Export.py.

Modes:
  fast     ONNX has (fp16 enc, bf16 dec) baked. Strongly-typed + weight
           streaming + opt-level 1 + dynamic batch profile.
           Matches the gold root plan recipe (validated on plane_nose:
           APE within 1.7% of gold).
  precise  ONNX has (fp32, fp32) baked. Strongly-typed (so TRT keeps
           fp32 throughout), TF32 cleared (matches PyTorch allow_tf32=False).
           Weight streaming optional; on by default so fp32 batch=2 fits
           on an 8 GB GPU (without streaming, fp32 batch=2 OOMs).

Both modes use builderOptimizationLevel=1 by default: per MAC-VO issue #18,
higher levels do fusions tuned for [0,1] logits and corrupt flow values.

Equivalent trtexec invocations:
  fast (deployment plan, matches gold root plan recipe):
    trtexec --onnx=MACVO_FrontendCov.onnx --saveEngine=MACVO_FrontendCov.plan ^
      --stronglyTyped --allowWeightStreaming --builderOptimizationLevel=1 ^
      --minShapes=image_1:1x3x704x704,image_2:1x3x704x704 ^
      --optShapes=image_1:2x3x704x704,image_2:2x3x704x704 ^
      --maxShapes=image_1:2x3x704x704,image_2:2x3x704x704

  precise (FP32 reference plan, batch=1/2/2, TF32 off, with streaming):
    trtexec --onnx=MACVO_FrontendCov_fp32.onnx ^
      --saveEngine=MACVO_FrontendCov_fp32.plan ^
      --stronglyTyped --allowWeightStreaming --noTF32 ^
      --builderOptimizationLevel=1 ^
      --minShapes=image_1:1x3x704x704,image_2:1x3x704x704 ^
      --optShapes=image_1:2x3x704x704,image_2:2x3x704x704 ^
      --maxShapes=image_1:2x3x704x704,image_2:2x3x704x704

Note: trtexec ships TRT 10.13 in the local CUDA bin; the Python `tensorrt`
package is 10.7. Plans built by trtexec do NOT load via Python tensorrt
("Version tag does not match. Current Version: 240, Serialized Engine
Version: 239"). Use this Python builder for plans consumed by the Python
runtime, the C++ tool in MACVO_TRT_BuildCpp/ for the C++ runtime.
"""
import argparse
import sys

import tensorrt as trt


INPUT_NAMES = ("image_1", "image_2")


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
          weight_streaming: bool,
          allow_tf32: bool,
          verbose: bool) -> None:
    severity = trt.Logger.INFO if verbose else trt.Logger.WARNING
    logger = trt.Logger(severity)
    builder = trt.Builder(logger)

    # Always strongly-typed: TRT respects the ONNX-baked dtypes literally.
    # For fast (fp16, bf16) ONNX: matches the gold root plan recipe.
    # For precise (fp32, fp32) ONNX: gives full fp32 throughout.
    # Strongly-typed is incompatible with the legacy BuilderFlag.FP16 /
    # OBEY_PRECISION_CONSTRAINTS path; do not set those flags here.
    flag = 1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
    network = builder.create_network(flag)

    parser = trt.OnnxParser(network, logger)
    # parse_from_file resolves the external-data sidecar (`*.onnx.data`)
    # relative to the ONNX path, regardless of the caller's CWD.
    if not parser.parse_from_file(onnx_path):
        for i in range(parser.num_errors):
            print(parser.get_error(i), file=sys.stderr)
        raise RuntimeError(f"ONNX parse failed: {onnx_path}")

    config = builder.create_builder_config()
    config.builder_optimization_level = opt_level

    if weight_streaming:
        # Pages weights from CPU at runtime. Reduces peak VRAM, allowing
        # fp32 batch=2 on 8 GB GPUs that would otherwise OOM. Also
        # constrains TRT's tactic search; for `fast` mode this is the
        # critical flag that reproduces the gold root plan layout.
        config.set_flag(trt.BuilderFlag.WEIGHT_STREAMING)

    if not allow_tf32:
        # Disables TF32 in fp32 matmul kernels. TF32 truncates the mantissa
        # to 10 bits, which makes a fp32-baked plan diverge from fp32
        # PyTorch (allow_tf32=False is the PyTorch default).
        config.clear_flag(trt.BuilderFlag.TF32)

    profile = builder.create_optimization_profile()
    for name in INPUT_NAMES:
        profile.set_shape(
            name,
            (min_batch, channels, height, width),
            (opt_batch, channels, height, width),
            (max_batch, channels, height, width),
        )
    config.add_optimization_profile(profile)

    print(f"Building TRT {trt.__version__} engine "
          f"(mode={mode}, weight_streaming={weight_streaming}, "
          f"allow_tf32={allow_tf32}). Takes several minutes...", flush=True)
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
    p.add_argument("--onnx", default=None,
                   help="Input ONNX file (default: derived from --mode)")
    p.add_argument("--out", default=None,
                   help="Output plan path (default: derived from --mode)")
    p.add_argument("--mode", choices=("fast", "precise"), default="fast",
                   help="fast = (fp16, bf16) baked ONNX; "
                        "precise = (fp32, fp32) baked ONNX")
    p.add_argument("--min-batch", type=int, default=1)
    p.add_argument("--opt-batch", type=int, default=2)
    p.add_argument("--max-batch", type=int, default=2)
    p.add_argument("--height", type=int, default=704)
    p.add_argument("--width",  type=int, default=704)
    p.add_argument("--opt-level", type=int, default=1,
                   help="Builder optimization level (1 is the only safe value "
                        "for this network per MAC-VO issue #18)")
    p.add_argument("--weight-streaming", dest="weight_streaming",
                   action=argparse.BooleanOptionalAction, default=None,
                   help="Enable BuilderFlag.WEIGHT_STREAMING. Default: on for "
                        "both modes (use --no-weight-streaming to disable). "
                        "Required for fast mode to match gold; needed for "
                        "precise mode to fit fp32 batch=2 on 8 GB.")
    p.add_argument("--tf32", dest="allow_tf32",
                   action=argparse.BooleanOptionalAction, default=None,
                   help="Allow TF32 in fp32 matmuls. Default: off for precise "
                        "(matches PyTorch allow_tf32=False), on for fast "
                        "(no fp32 ops to TF32-ize). Use --no-tf32 to force off.")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    # Resolve mode-derived defaults.
    if args.weight_streaming is None:
        args.weight_streaming = True  # on for both modes by default
    if args.allow_tf32 is None:
        args.allow_tf32 = (args.mode == "fast")

    onnx = args.onnx or (
        "MACVO_FrontendCov.onnx" if args.mode == "fast"
        else "MACVO_FrontendCov_fp32.onnx"
    )
    out = args.out or (
        "MACVO_FrontendCov.plan" if args.mode == "fast"
        else "MACVO_FrontendCov_fp32.plan"
    )

    build(
        onnx_path=onnx,
        plan_path=out,
        mode=args.mode,
        min_batch=args.min_batch,
        opt_batch=args.opt_batch,
        max_batch=args.max_batch,
        channels=3,
        height=args.height,
        width=args.width,
        opt_level=args.opt_level,
        weight_streaming=args.weight_streaming,
        allow_tf32=args.allow_tf32,
        verbose=args.verbose,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
