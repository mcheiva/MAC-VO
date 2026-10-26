"""
ONNX -> TensorRT engine builder for the MACVO FlowFormerCov frontend.
Python mirror of MACVO_TRT_BuildCpp/. Both modes build a strongly-typed
network so TRT respects the dtypes baked into the ONNX literally. Pair
with the matching --mode in Export.py.

Static batch 1 is the norm: export the ONNX with static shapes (batch 1) and
pass no shape flags. When every network input has fully static dims, no
optimization profile is created and --min/opt/max-batch, --height and
--width are ignored. A profile (min/opt/max, default 1/1/1) is added only if
an input is dynamic.

Modes:
  fast     ONNX has (fp16 enc, bf16 dec) baked. Strongly-typed + weight
           streaming + opt-level 5 + TF32 allowed.
  precise  ONNX has (fp32, fp32) baked. Strongly-typed (so TRT keeps
           fp32 throughout), TF32 cleared (matches PyTorch allow_tf32=False).

Weight streaming is on by default in both modes. Measured on the EIVA gaps
engine, RTX 4090, TRT 10.13.2:
  * On a dynamic-axis ONNX (the old graph) WEIGHT_STREAMING forces the
    all-Myelin graph: 942 layers vs 3222, 26.3 vs 38.4 ms per call with a
    CUDA graph, 28.4 vs 54.2 ms without.
  * On the static batch-1 ONNX that Export.py now produces the flag changes
    nothing: identical tactic lists, 964 layers, a single stream, the same
    plan size and 23.92-23.98 ms either way.
The default stays on so a dynamic-axis build also gets the fast graph. At the
default (full) budget no weights are streamed at runtime. Accuracy:
plane_tail1 was 0.036 without the flag vs 0.026 with it, outside the observed
0.023-0.028 spread but from a single dive, so weak evidence. Use
--no-weight-streaming to build without the flag.

Optimization level 5 (default): the deployed EIVA engine and the static
batch-1 engines were built at 5 and validated end to end on the VO eval;
MAC-VO issue #18 reported corrupted flow at higher levels on its build.

Equivalent trtexec invocations (static ONNX, no shape flags):
  fast:
    trtexec --onnx=MACVO_FrontendCov_b1.onnx --saveEngine=MACVO_FrontendCov.plan ^
      --stronglyTyped --allowWeightStreaming --builderOptimizationLevel=5 ^
      --avgTiming=8 --profilingVerbosity=detailed ^
      --timingCacheFile=timing.cache
  precise (FP32 reference plan, TF32 off):
    trtexec --onnx=MACVO_FrontendCov_fp32_b1.onnx ^
      --saveEngine=MACVO_FrontendCov_fp32.plan ^
      --stronglyTyped --allowWeightStreaming --noTF32 ^
      --builderOptimizationLevel=5 --avgTiming=8 ^
      --profilingVerbosity=detailed --timingCacheFile=timing.cache
  Add --errorOnTimingCacheMiss for --error-on-timing-cache-miss.
  For a dynamic ONNX add --minShapes/--optShapes/--maxShapes, e.g.
  image_1:1x3x704x704,image_2:1x3x704x704.

Writes <plan>.json next to the plan (ONNX path/size/hash, build settings,
timing-cache status, TensorRT version, GPU), same fields as the C++ builder.

Note: engines are only loadable by the exact TensorRT version that built them
("Version tag does not match. Current Version: 240, Serialized Engine
Version: 239"). Check `tensorrt.__version__` against the nvinfer_10 the C++
runtime links before shipping a plan.
"""
import argparse
import json
import os
import sys

import tensorrt as trt


INPUT_NAMES = ("image_1", "image_2")


class CountingLogger(trt.ILogger):
    """trt.Logger that counts ERROR / INTERNAL_ERROR messages and the
    timing-cache-miss warnings (one multi-line "Engine generation failed with
    backend strategy N. Error message: ... Adding new entry to timing cache is
    not allowed if `BuilderFlag::kERROR_ON_TIMING_CACHE_MISS` is set")."""

    def __init__(self, verbose: bool) -> None:
        trt.ILogger.__init__(self)
        self.verbose = verbose
        self.errors = 0
        self.strategy_failures = 0

    def log(self, severity, msg) -> None:
        if severity in (trt.ILogger.Severity.INTERNAL_ERROR,
                        trt.ILogger.Severity.ERROR):
            self.errors += 1
            print(f"[TRT ERROR] {msg}", file=sys.stderr)
        elif severity == trt.ILogger.Severity.WARNING:
            if ("kERROR_ON_TIMING_CACHE_MISS" in msg
                    or "Adding new entry to timing cache" in msg):
                self.strategy_failures += 1
            print(f"[TRT WARN ] {msg}", file=sys.stderr)
        elif self.verbose:
            print(f"[TRT INFO ] {msg}", file=sys.stderr)


def fnv1a64_hex(path: str) -> str | None:
    """64-bit FNV-1a of a file's bytes as 16 hex digits; None on read failure."""
    h = 14695981039346656037
    try:
        with open(path, "rb") as f:
            while True:
                chunk = f.read(1 << 20)
                if not chunk:
                    break
                for b in chunk:
                    h = ((h ^ b) * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    except OSError:
        return None
    return f"{h:016x}"


def all_inputs_static(network) -> bool:
    """True if every dimension of every network input is fixed (no -1)."""
    for i in range(network.num_inputs):
        if any(d < 0 for d in network.get_input(i).shape):
            return False
    return True


def gpu_info() -> tuple[str | None, str | None]:
    """(name, compute capability) via torch.cuda if importable, else (None, None)."""
    try:
        import torch
        if not torch.cuda.is_available():
            return None, None
        dev = torch.cuda.current_device()
        major, minor = torch.cuda.get_device_capability(dev)
        return torch.cuda.get_device_name(dev), f"{major}.{minor}"
    except Exception:
        return None, None


def write_sidecar(plan_path: str, onnx_path: str, mode: str, is_static: bool,
                  min_batch: int, opt_batch: int, max_batch: int,
                  height: int, width: int, opt_level: int,
                  avg_timing_iterations: int, tiling_level: int,
                  error_on_timing_cache_miss: bool, weight_streaming: bool,
                  allow_tf32: bool, timing_cache_path: str | None,
                  cache_loaded: bool, bytes_loaded: int,
                  bytes_saved: int) -> None:
    """Writes <plan_path>.json; same fields and order as the C++ builder."""
    try:
        onnx_size = os.path.getsize(onnx_path)
    except OSError:
        onnx_size = None
    # tensorrt.__version__ is "MAJOR.MINOR.PATCH.BUILD"; the C++ builder's
    # tensorrt_version is (MAJOR * 100 + MINOR) * 100 + PATCH.
    # The 4th component is the header's NV_TENSORRT_BUILD (the C++ builder writes the same); null if absent.
    parts = trt.__version__.split(".")
    major, minor, patch = (int(x) for x in parts[:3])
    build_header = int(parts[3]) if len(parts) > 3 and parts[3].isdigit() else None
    used = timing_cache_path is not None
    gpu_name, cc = gpu_info()
    side = {
        "onnx_path": os.path.abspath(onnx_path),
        "onnx_size_bytes": onnx_size,
        "onnx_fnv1a64": fnv1a64_hex(onnx_path),
        "mode": mode,
        "static": is_static,
        "min_batch": None if is_static else min_batch,
        "opt_batch": None if is_static else opt_batch,
        "max_batch": None if is_static else max_batch,
        "height": None if is_static else height,
        "width": None if is_static else width,
        "optimization_level": opt_level,
        "avg_timing_iterations": avg_timing_iterations,
        "tiling_optimization_level": tiling_level,
        "error_on_timing_cache_miss": error_on_timing_cache_miss,
        "weight_streaming": weight_streaming,
        "allow_tf32": allow_tf32,
        "timing_cache": os.path.abspath(timing_cache_path) if used else None,
        "timing_cache_status": (("loaded" if cache_loaded else "started_empty")
                                if used else None),
        "timing_cache_bytes_loaded": bytes_loaded if used else None,
        "timing_cache_bytes_saved": bytes_saved if used else None,
        "tensorrt_version": (major * 100 + minor) * 100 + patch,
        "tensorrt_version_str": f"{major}.{minor}.{patch}",
        "tensorrt_build_header": build_header,
        "gpu_name": gpu_name,
        "compute_capability": cc,
    }
    with open(plan_path + ".json", "w", encoding="utf-8") as f:
        json.dump(side, f, indent=2)
        f.write("\n")
    print(f"Wrote {plan_path}.json", file=sys.stderr, flush=True)


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
          avg_timing_iterations: int,
          tiling_level: int,
          error_on_timing_cache_miss: bool,
          weight_streaming: bool,
          allow_tf32: bool,
          timing_cache_path: str | None,
          verbose: bool,
          parse_only: bool = False) -> None:
    logger = CountingLogger(verbose)

    # A stale sidecar next to a failed build would describe the wrong plan.
    if not parse_only:
        try:
            os.remove(plan_path + ".json")
        except FileNotFoundError:
            pass
        if timing_cache_path is not None:
            parent = os.path.dirname(os.path.abspath(timing_cache_path))
            if not os.path.isdir(parent):
                raise RuntimeError(
                    f"Timing cache directory does not exist: {parent}")

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

    is_static = all_inputs_static(network)
    if parse_only:
        print(f"parsed {onnx_path}: inputs "
              f"{'all static' if is_static else 'dynamic'}, optimization "
              f"profile would be {'NOT added' if is_static else 'added'}",
              flush=True)
        return

    config = builder.create_builder_config()
    config.builder_optimization_level = opt_level
    config.avg_timing_iterations = avg_timing_iterations
    # The repo's tensorrt stubs predate TilingOptimizationLevel (TRT 10.8+).
    config.tiling_optimization_level = trt.TilingOptimizationLevel(  # pyright: ignore[reportAttributeAccessIssue]
        tiling_level)
    if error_on_timing_cache_miss:
        config.set_flag(trt.BuilderFlag.ERROR_ON_TIMING_CACHE_MISS)

    if weight_streaming:
        # Makes the builder offload every Myelin-supported op to Myelin
        # (one graph instead of many islands); see the module docstring.
        config.set_flag(trt.BuilderFlag.WEIGHT_STREAMING)

    # Makes the plan inspectable (per-layer precision/format/tactic). Also
    # raises the NVTX verbosity, which TensorRT says may add enqueueV3
    # latency.
    config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED

    if not allow_tf32:
        # Disables TF32 in fp32 matmul kernels. TF32 truncates the mantissa
        # to 10 bits, which makes a fp32-baked plan diverge from fp32
        # PyTorch (allow_tf32=False is the PyTorch default).
        config.clear_flag(trt.BuilderFlag.TF32)

    if is_static:
        print("ONNX inputs are static: no optimization profile; "
              "--min/opt/max-batch, --height and --width are ignored",
              file=sys.stderr, flush=True)
    else:
        profile = builder.create_optimization_profile()
        for name in INPUT_NAMES:
            profile.set_shape(
                name,
                (min_batch, channels, height, width),
                (opt_batch, channels, height, width),
                (max_batch, channels, height, width),
            )
        if config.add_optimization_profile(profile) < 0:
            raise RuntimeError("add_optimization_profile failed "
                               "(invalid profile)")

    # Timing cache: reusing a cache from a prior build makes tactic choice
    # repeatable in practice; TensorRT does not guarantee it. Check by
    # diffing TacticName in `trtexec --dumpLayerInfo` of two plans. The cache
    # object must outlive the build, so it is kept in a local.
    timing_cache = None
    cache_loaded = False
    bytes_loaded = 0
    bytes_saved = 0
    if timing_cache_path is not None:
        cache_bytes = b""
        if os.path.exists(timing_cache_path):
            with open(timing_cache_path, "rb") as f:
                cache_bytes = f.read()
        cache_loaded = len(cache_bytes) > 0
        bytes_loaded = len(cache_bytes)
        timing_cache = config.create_timing_cache(cache_bytes)
        if timing_cache is None:
            raise RuntimeError("create_timing_cache failed")
        if not config.set_timing_cache(timing_cache, ignore_mismatch=False):
            raise RuntimeError("set_timing_cache failed (cache was built for "
                               "different device properties?): "
                               f"{timing_cache_path}")
        if cache_loaded:
            print(f"Timing cache loaded: {timing_cache_path} "
                  f"({bytes_loaded} bytes)", file=sys.stderr, flush=True)
        else:
            print(f"Timing cache not found, starting empty: "
                  f"{timing_cache_path}", file=sys.stderr, flush=True)

    print(f"Building TRT {trt.__version__} engine "
          f"(mode={mode}, weight_streaming={weight_streaming}, "
          f"allow_tf32={allow_tf32}, opt_level={opt_level}, "
          f"avg_timing={avg_timing_iterations}, tiling={tiling_level}). "
          f"Takes several minutes...", flush=True)
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("build_serialized_network returned None "
                           "(likely OOM or builder error)")
    # In TRT 10.13.2 a miss does not fail the build: the logger gets one
    # multi-line warning ("Engine generation failed with backend strategy N.
    # Error message: ... Adding new entry to timing cache is not allowed ..."),
    # and TensorRT falls back to another strategy and writes a plan. Primary
    # signal: the cache as this build leaves it, serialized before the on-disk
    # merge, is larger than the loaded bytes, i.e. TensorRT timed new entries.
    # Secondary: the logger saw an error or the miss warning.
    if error_on_timing_cache_miss and timing_cache is not None:
        own_size = len(bytes(timing_cache.serialize()))
        if own_size > bytes_loaded:
            raise RuntimeError(
                "--error-on-timing-cache-miss: the build's timing cache grew "
                f"from {bytes_loaded} to {own_size} bytes (TensorRT timed new "
                "entries); plan not written")
    if error_on_timing_cache_miss and (logger.errors or logger.strategy_failures):
        raise RuntimeError(
            f"--error-on-timing-cache-miss: the logger saw {logger.errors} "
            f"error(s) and {logger.strategy_failures} timing-cache-miss "
            "warning(s) during the build; plan not written")

    plan_bytes = bytes(serialized)
    with open(plan_path, "wb") as f:
        f.write(plan_bytes)
    print(f"Wrote {plan_path}: {len(plan_bytes) / (1024 * 1024):.2f} MiB",
          flush=True)

    cache_blob = None
    if timing_cache is not None and timing_cache_path is not None:
        # Another build (e.g. on the other GPU) may have saved since we
        # loaded: merge its entries so neither run's work is lost.
        if os.path.exists(timing_cache_path):
            with open(timing_cache_path, "rb") as f:
                disk = f.read()
            if disk:
                on_disk = config.create_timing_cache(disk)
                if on_disk is None or not timing_cache.combine(
                        on_disk, ignore_mismatch=False):
                    print("Timing cache: could not merge the on-disk copy, "
                          "overwriting it with this build's cache",
                          file=sys.stderr, flush=True)
        cache_blob = bytes(timing_cache.serialize())
        bytes_saved = len(cache_blob)

    # Sidecar goes right after the plan, before the cache is touched.
    write_sidecar(plan_path, onnx_path, mode, is_static, min_batch, opt_batch,
                  max_batch, height, width, opt_level, avg_timing_iterations,
                  tiling_level, error_on_timing_cache_miss, weight_streaming,
                  allow_tf32, timing_cache_path, cache_loaded, bytes_loaded,
                  bytes_saved)

    if cache_blob is not None and timing_cache_path is not None:
        tmp = timing_cache_path + ".tmp"
        with open(tmp, "wb") as f:
            f.write(cache_blob)
        # os.replace() overwrites an existing file on Windows too.
        os.replace(tmp, timing_cache_path)
        print(f"Timing cache saved: {timing_cache_path} "
              f"({len(cache_blob)} bytes)", file=sys.stderr, flush=True)


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
    prof_note = (" Optimization profile only; ignored (no profile is created) "
                 "when the ONNX inputs are fully static.")
    p.add_argument("--min-batch", type=int, default=1,
                   help="Min batch." + prof_note)
    p.add_argument("--opt-batch", type=int, default=1,
                   help="Opt batch." + prof_note)
    p.add_argument("--max-batch", type=int, default=1,
                   help="Max batch." + prof_note)
    p.add_argument("--height", type=int, default=704,
                   help="Input height (px)." + prof_note)
    p.add_argument("--width",  type=int, default=704,
                   help="Input width (px)." + prof_note)
    p.add_argument("--opt-level", type=int, default=5, choices=range(6),
                   metavar="{0..5}",
                   help="Builder optimization level (0-5, default 5). The "
                        "deployed EIVA engine and the static batch-1 engines "
                        "were built at 5 and validated on the VO eval; MAC-VO "
                        "#18 reported corrupted flow at higher levels on its "
                        "build.")
    p.add_argument("--avg-timing", type=int, default=8,
                   help="Timing runs averaged per tactic (trtexec default 8, "
                        "API default 1). Only affects timing-cache misses; the "
                        "existing production tactics were timed at 8.")
    p.add_argument("--tiling-level", type=int, default=0, choices=range(4),
                   metavar="{0..3}",
                   help="Tiling optimization level: 0 none, 1 fast, "
                        "2 moderate, 3 full (default 0)")
    p.add_argument("--error-on-timing-cache-miss", action="store_true",
                   help="Set BuilderFlag.ERROR_ON_TIMING_CACHE_MISS and fail "
                        "the build (no plan written) if the build's timing "
                        "cache grew beyond the loaded bytes (TensorRT timed "
                        "new entries) or TensorRT logged an error or the "
                        "cache-miss warning. Reproduces a plan exactly from a "
                        "populated --timing-cache.")
    p.add_argument("--weight-streaming", dest="weight_streaming",
                   action=argparse.BooleanOptionalAction, default=None,
                   help="Enable BuilderFlag.WEIGHT_STREAMING (default: on for "
                        "both modes). On a dynamic-axis ONNX (the old graph) "
                        "TRT 10.13 then builds the all-Myelin graph: 942 vs "
                        "3222 layers, 26.3 vs 38.4 ms per call with a CUDA "
                        "graph (28.4 vs 54.2 without). On the static batch-1 "
                        "ONNX Export.py produces the flag changes nothing "
                        "(same tactics, 964 layers, 23.92-23.98 ms). At the "
                        "default budget nothing is streamed at runtime.")
    p.add_argument("--tf32", dest="allow_tf32",
                   action=argparse.BooleanOptionalAction, default=None,
                   help="Allow TF32 in fp32 matmuls. Default: off for precise "
                        "(matches PyTorch allow_tf32=False), on for fast "
                        "(no fp32 ops to TF32-ize). Use --no-tf32 to force off.")
    p.add_argument("--timing-cache", default=None,
                   help="Path to a timing-cache file, directory must exist. "
                        "Loaded if present, merged with any newer on-disk copy "
                        "and saved atomically after the build. Reuse makes "
                        "tactic choice repeatable in practice (TensorRT does "
                        "not guarantee it); verify by diffing TacticName in "
                        "trtexec --dumpLayerInfo.")
    p.add_argument("--parse-only", action="store_true",
                   help=argparse.SUPPRESS)
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    # Resolve mode-derived defaults.
    if args.weight_streaming is None:
        args.weight_streaming = True  # on for both modes by default
    if args.allow_tf32 is None:
        args.allow_tf32 = (args.mode == "fast")
    if args.avg_timing < 1:
        p.error("--avg-timing must be >= 1")

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
        avg_timing_iterations=args.avg_timing,
        tiling_level=args.tiling_level,
        error_on_timing_cache_miss=args.error_on_timing_cache_miss,
        weight_streaming=args.weight_streaming,
        allow_tf32=args.allow_tf32,
        timing_cache_path=args.timing_cache,
        verbose=args.verbose,
        parse_only=args.parse_only,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
