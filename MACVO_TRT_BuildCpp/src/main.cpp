// trtexec-style ONNX -> TRT engine builder for MACVO FlowFormerCov.
// All build logic lives in macvo_trt_builder.hpp; main() is just CLI glue.

#include <CLI/CLI.hpp>
#include <exception>
#include <iostream>
#include <string>

#include "macvo_trt_builder.hpp"

int main(int argc, char** argv) {
    CLI::App app{ "MACVO TensorRT engine builder" };
    app.set_version_flag("--version", "macvo_trt_build 0.1.0");

    std::string onnx;
    std::string engine;
    std::string mode_str = "fast";
    macvo_trt::BuildOptions opts;

    app.add_option("--onnx", onnx, "Input ONNX model")
        ->required()->check(CLI::ExistingFile);
    app.add_option("--out", engine, "Output TRT plan file")
        ->required();
    app.add_option("--mode", mode_str, "Build mode: fast (FP16) | precise (FP32, TF32 off)")
        ->check(CLI::IsMember({ "fast", "precise" }))
        ->capture_default_str();
    app.add_option("--min-batch", opts.min_batch, "Min batch")->capture_default_str();
    app.add_option("--opt-batch", opts.opt_batch, "Opt batch")->capture_default_str();
    app.add_option("--max-batch", opts.max_batch, "Max batch")->capture_default_str();
    app.add_option("--height", opts.height, "Input height (px)")->capture_default_str();
    app.add_option("--width",  opts.width,  "Input width (px)")->capture_default_str();
    app.add_option("--opt-level", opts.optimization_level,
                   "Builder optimization level (1 is the only safe value for this net)")
        ->capture_default_str();
    app.add_flag("--verbose", opts.verbose, "Verbose TRT logs");

    CLI11_PARSE(app, argc, argv);

    opts.onnx_path = onnx;
    opts.engine_path = engine;
    opts.mode = (mode_str == "precise") ? macvo_trt::Mode::Precise
                                        : macvo_trt::Mode::Fast;

    if (opts.mode == macvo_trt::Mode::Precise && opts.max_batch > 1) {
        std::cerr << "[warn] precise mode: forcing batch=1 (8 GB OOM cap).\n";
        opts.min_batch = opts.opt_batch = opts.max_batch = 1;
    }

    try {
        macvo_trt::Build(opts);
        std::cout << "OK: " << opts.engine_path.string() << '\n';
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Build failed: " << e.what() << '\n';
        return 1;
    }
}
