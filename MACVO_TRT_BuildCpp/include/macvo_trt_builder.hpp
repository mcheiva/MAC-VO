#pragma once
//
// Header-only TensorRT engine builder for the MACVO FlowFormerCov frontend.
//
// Mirrors MACVO_TRT_Build.py / MACVO_TRT_Build_FP32.py:
//   * Mode::Fast    -> FP16 plan, builderOptimizationLevel=1, cov-tail
//                      (Cast -> Mul*2 -> Exp) pinned to FP32 so exp() does
//                      not saturate the fp16 range (~65504).
//   * Mode::Precise -> FP32 plan, TF32 disabled to match PyTorch's default
//                      allow_tf32=False matmul behaviour. Batch=1 only.
//
#include <NvInfer.h>
#include <NvOnnxParser.h>

#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace macvo_trt {

enum class Mode { Fast, Precise };

struct BuildOptions {
    std::filesystem::path onnx_path;
    std::filesystem::path engine_path;
    Mode mode = Mode::Fast;

    // Optimization profile (NCHW). Defaults match the Python builders.
    std::int32_t min_batch = 1;
    std::int32_t opt_batch = 2;
    std::int32_t max_batch = 2;
    std::int32_t channels  = 3;
    std::int32_t height    = 704;
    std::int32_t width     = 704;

    // Per MAC-VO issue #18: keep this at 1. Higher levels enable fusions
    // tuned for [0,1] logits and corrupt optical-flow values.
    int optimization_level = 1;

    // Page weights from CPU at runtime. Reduces peak VRAM, allowing fp32
    // batch=2 to fit on 8 GB GPUs. Also the critical flag for `fast` mode
    // -- without it the trajectory regresses ~14% APE on plane_nose.
    bool weight_streaming = true;

    // Allow TF32 in fp32 matmul kernels. Default true for fast (no fp32
    // ops to TF32-ize), default false for precise (TF32 truncates the
    // mantissa to 10 bits, breaking parity with PyTorch allow_tf32=False).
    // The constructor of BuildOptions does not know the mode, so the
    // caller / CLI applies the mode-specific default before Build().
    bool allow_tf32 = true;

    // Cov-tail fp32 pin set. Empty by default: the MACVO consumer applies
    // /255 + exp(2 * cov_raw) itself, so neither lives in the ONNX graph and
    // there is no fp16 exp() saturation risk to guard. Re-populate (e.g.
    // {"/Mul", "/Exp"}) only if a future Export.py change moves pre/post
    // back into the graph; in that case run MACVO_TRT_Inspect.py against the
    // re-exported ONNX to recover the actual post-onnxsim layer names.
    std::vector<std::string> fp32_layer_names = {};

    std::array<std::string, 2> input_names{ "image_1", "image_2" };
    std::string cov_output_name = "cov";

    bool verbose = false;
};

class Logger : public nvinfer1::ILogger {
public:
    explicit Logger(bool verbose) noexcept : verbose_(verbose) {}
    void log(Severity s, char const* msg) noexcept override {
        using S = Severity;
        if (s == S::kINTERNAL_ERROR || s == S::kERROR) {
            std::cerr << "[TRT ERROR] " << msg << '\n';
        } else if (s == S::kWARNING) {
            std::cerr << "[TRT WARN ] " << msg << '\n';
        } else if (verbose_) {
            std::cerr << "[TRT INFO ] " << msg << '\n';
        }
    }
private:
    bool verbose_;
};

template <typename T>
using TrtPtr = std::unique_ptr<T>;

// The legacy FP16 / OBEY_PRECISION_CONSTRAINTS path plus per-layer
// setPrecision / setOutputType / ITensor::setType are flagged deprecated in
// TRT 10's strongly-typed flow, but still functional and required to mirror
// MACVO_TRT_Build.py exactly. Scope the suppression to this function only.
#if defined(_MSC_VER)
#  pragma warning(push)
#  pragma warning(disable: 4996)
#elif defined(__GNUC__) || defined(__clang__)
#  pragma GCC diagnostic push
#  pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif

inline void Build(const BuildOptions& opts) {
    Logger logger(opts.verbose);

    TrtPtr<nvinfer1::IBuilder> builder{ nvinfer1::createInferBuilder(logger) };
    if (!builder) throw std::runtime_error("createInferBuilder failed");

    // Always strongly typed: TRT respects the ONNX-baked dtypes literally.
    // For Mode::Fast (fp16, bf16) ONNX: matches the gold root plan recipe.
    // For Mode::Precise (fp32, fp32) ONNX: gives full fp32 throughout.
    // Strongly-typed is incompatible with BuilderFlag::kFP16 /
    // kOBEY_PRECISION_CONSTRAINTS; do not set those here.
    const std::uint32_t net_flag = 1U << static_cast<std::uint32_t>(
        nvinfer1::NetworkDefinitionCreationFlag::kSTRONGLY_TYPED);
    TrtPtr<nvinfer1::INetworkDefinition> network{ builder->createNetworkV2(net_flag) };
    if (!network) throw std::runtime_error("createNetworkV2 failed");

    TrtPtr<nvonnxparser::IParser> parser{ nvonnxparser::createParser(*network, logger) };
    if (!parser) throw std::runtime_error("createParser failed");

    const int parser_severity = static_cast<int>(nvinfer1::ILogger::Severity::kWARNING);
    if (!parser->parseFromFile(opts.onnx_path.string().c_str(), parser_severity)) {
        for (int i = 0; i < parser->getNbErrors(); ++i) {
            std::cerr << "ONNX parser: " << parser->getError(i)->desc() << '\n';
        }
        throw std::runtime_error("ONNX parse failed: " + opts.onnx_path.string());
    }

    TrtPtr<nvinfer1::IBuilderConfig> config{ builder->createBuilderConfig() };
    if (!config) throw std::runtime_error("createBuilderConfig failed");
    config->setBuilderOptimizationLevel(opts.optimization_level);

    if (opts.weight_streaming) {
        // Pages weights from CPU at runtime. Reduces peak VRAM (allowing
        // fp32 batch=2 on 8 GB GPUs that would otherwise OOM) and is the
        // critical flag for Mode::Fast to reproduce the gold root plan
        // layout -- without it the trajectory regresses ~14% APE.
        config->setFlag(nvinfer1::BuilderFlag::kWEIGHT_STREAMING);
    }

    if (!opts.allow_tf32) {
        // TF32 truncates fp32 matmul mantissa to 10 bits. Disabling it
        // matches PyTorch's allow_tf32=False default. Default for
        // Mode::Precise; harmless for Mode::Fast (no fp32 ops to TF32-ize).
        config->clearFlag(nvinfer1::BuilderFlag::kTF32);
    }

    auto* profile = builder->createOptimizationProfile();
    if (!profile) throw std::runtime_error("createOptimizationProfile failed");
    const nvinfer1::Dims4 dmin{ opts.min_batch, opts.channels, opts.height, opts.width };
    const nvinfer1::Dims4 dopt{ opts.opt_batch, opts.channels, opts.height, opts.width };
    const nvinfer1::Dims4 dmax{ opts.max_batch, opts.channels, opts.height, opts.width };
    for (const auto& iname : opts.input_names) {
        profile->setDimensions(iname.c_str(), nvinfer1::OptProfileSelector::kMIN, dmin);
        profile->setDimensions(iname.c_str(), nvinfer1::OptProfileSelector::kOPT, dopt);
        profile->setDimensions(iname.c_str(), nvinfer1::OptProfileSelector::kMAX, dmax);
    }
    config->addOptimizationProfile(profile);

    std::cerr << "Building TRT engine (mode="
              << (opts.mode == Mode::Fast ? "fast" : "precise")
              << "). This takes several minutes...\n";

    TrtPtr<nvinfer1::IHostMemory> plan{
        builder->buildSerializedNetwork(*network, *config) };
    if (!plan) throw std::runtime_error("buildSerializedNetwork returned null");

    std::ofstream f(opts.engine_path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open output: " + opts.engine_path.string());
    f.write(static_cast<const char*>(plan->data()),
            static_cast<std::streamsize>(plan->size()));
    if (!f) throw std::runtime_error("Failed to write: " + opts.engine_path.string());

    std::cerr << "Wrote " << opts.engine_path << " ("
              << (plan->size() / (1024.0 * 1024.0)) << " MiB)\n";
}

#if defined(_MSC_VER)
#  pragma warning(pop)
#elif defined(__GNUC__) || defined(__clang__)
#  pragma GCC diagnostic pop
#endif

// Convenience for downstream cache-keying: TRT runtime version baked into
// libnvinfer. Combine with cudaDeviceProp::name + sm_<major><minor> on the
// caller side to invalidate cached plans across GPU/driver/TRT changes.
inline int InferLibVersion() noexcept {
    return ::getInferLibVersion();
}

} // namespace macvo_trt
