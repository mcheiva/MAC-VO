// trtexec-style ONNX -> TRT engine builder for MACVO FlowFormerCov.
// All build logic lives in macvo_trt_builder.hpp; main() is just CLI glue.

#include <CLI/CLI.hpp>
#include <exception>
#include <iostream>
#include <string>

#include "macvo_trt_builder.hpp"

int main(int argc, char** argv) {
    CLI::App app{ "MACVO TensorRT engine builder" };
    app.set_version_flag("--version", "macvo_trt_build 0.2.0");

    std::string onnx;
    std::string engine;
    std::string mode_str = "fast";
    std::string timing_cache;
    bool no_weight_streaming = false;
    bool no_tf32 = false;
    bool tf32_explicit = false;
    macvo_trt::BuildOptions opts;

    app.add_option("--onnx", onnx, "Input ONNX model")
        ->required()->check(CLI::ExistingFile);
    app.add_option("--out", engine, "Output TRT plan file")
        ->required();
    app.add_option("--mode", mode_str,
                   "Build mode: fast = (fp16, bf16) baked ONNX (deployment); "
                   "precise = (fp32, fp32) baked ONNX (reference)")
        ->check(CLI::IsMember({ "fast", "precise" }))
        ->capture_default_str();
    const std::string prof_note = " Optimization profile only; ignored (no profile "
                                  "is created) when the ONNX inputs are fully static.";
    app.add_option("--min-batch", opts.min_batch, "Min batch." + prof_note)->capture_default_str();
    app.add_option("--opt-batch", opts.opt_batch, "Opt batch." + prof_note)->capture_default_str();
    app.add_option("--max-batch", opts.max_batch, "Max batch." + prof_note)->capture_default_str();
    app.add_option("--height", opts.height, "Input height (px)." + prof_note)->capture_default_str();
    app.add_option("--width",  opts.width,  "Input width (px)." + prof_note)->capture_default_str();
    app.add_option("--opt-level", opts.optimization_level,
                   "Builder optimization level (0-5). The deployed EIVA engine "
                   "and the static batch-1 engines were built at 5 and "
                   "validated on the VO eval; MAC-VO #18 reported corrupted "
                   "flow at higher levels on its build.")
        ->check(CLI::Range(0, 5))->capture_default_str();
    app.add_option("--avg-timing", opts.avg_timing_iterations,
                   "Timing runs averaged per tactic (trtexec default 8, API "
                   "default 1). Only affects timing-cache misses; the existing "
                   "production tactics were timed at 8.")
        ->check(CLI::PositiveNumber)->capture_default_str();
    app.add_option("--tiling-level", opts.tiling_level,
                   "Tiling optimization level: 0 none, 1 fast, 2 moderate, 3 full")
        ->check(CLI::Range(0, 3))->capture_default_str();
    app.add_flag("--error-on-timing-cache-miss", opts.error_on_timing_cache_miss,
                 "Set kERROR_ON_TIMING_CACHE_MISS and fail the build (no plan "
                 "written) if the build's timing cache grew beyond the loaded "
                 "bytes (TensorRT timed new entries) or TensorRT logged an "
                 "error or the cache-miss warning. Reproduces a plan exactly "
                 "from a populated --timing-cache.");
    app.add_flag("--parse-only", opts.parse_only,
                 "Parse the ONNX, print whether a profile would be added, "
                 "and exit without building")->group("");
    app.add_flag("--no-weight-streaming", no_weight_streaming,
                 "Disable BuilderFlag::kWEIGHT_STREAMING (default: enabled). "
                 "On a dynamic-axis ONNX (the old graph) TRT 10.13 then builds "
                 "3222 layers instead of the all-Myelin 942: 38.4 vs 26.3 ms "
                 "per call with a CUDA graph (54.2 vs 28.4 without). On the "
                 "static batch-1 ONNX Export.py produces the flag changes "
                 "nothing (same tactics, 964 layers, 23.92-23.98 ms). At the "
                 "default budget nothing is streamed at runtime.");
    auto* tf32_flag = app.add_flag("--no-tf32", no_tf32,
                                   "Force TF32 off in fp32 matmul kernels "
                                   "(default: off for precise, on for fast).");
    app.add_option("--timing-cache", timing_cache,
                   "TensorRT timing cache file, directory must exist. Loaded "
                   "if present, merged with any newer on-disk copy and saved "
                   "atomically after the build. Reuse makes tactic choice "
                   "repeatable in practice (TensorRT does not guarantee it); "
                   "verify by diffing TacticName in trtexec --dumpLayerInfo.");
    app.add_flag("--verbose", opts.verbose, "Verbose TRT logs");

    CLI11_PARSE(app, argc, argv);

    tf32_explicit = (tf32_flag->count() > 0);

    opts.onnx_path = onnx;
    opts.engine_path = engine;
    opts.timing_cache_path = timing_cache;
    opts.mode = (mode_str == "precise") ? macvo_trt::Mode::Precise
                                        : macvo_trt::Mode::Fast;
    opts.weight_streaming = !no_weight_streaming;

    // TF32 default depends on mode unless user explicitly passed --no-tf32.
    if (tf32_explicit) {
        opts.allow_tf32 = !no_tf32;
    } else {
        opts.allow_tf32 = (opts.mode == macvo_trt::Mode::Fast);
    }

    try {
        macvo_trt::Build(opts);
        if (!opts.parse_only) {
            std::cout << "OK: " << opts.engine_path.string() << '\n';
        }
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Build failed: " << e.what() << '\n';
        return 1;
    }
}
