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

    // ONNX layer names of the cov-branch tail. exp() over these must run in
    // fp32 to avoid fp16 saturation; the Python builder pins the same set.
    // Post onnx-simplifier the chain is just (cov_up * 2.0) -> exp(...), so
    // pinning /Mul and /Exp covers the whole fp32 island. If the wrapper or
    // simplifier is changed, run MACVO_TRT_Inspect.py to refresh these.
    std::vector<std::string> fp32_layer_names = {
        "/Mul",
        "/Exp",
    };

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

    // flag=0: legacy network (allows mixed precision via FP16 flag).
    // Matches Python `builder.create_network(0)`.
    TrtPtr<nvinfer1::INetworkDefinition> network{ builder->createNetworkV2(0U) };
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

    if (opts.mode == Mode::Fast) {
        config->setFlag(nvinfer1::BuilderFlag::kFP16);
        config->setFlag(nvinfer1::BuilderFlag::kOBEY_PRECISION_CONSTRAINTS);

        const std::set<std::string> fp32_set(
            opts.fp32_layer_names.begin(), opts.fp32_layer_names.end());

        std::vector<std::string> pinned;
        const int n_layers = network->getNbLayers();
        for (int i = 0; i < n_layers; ++i) {
            auto* layer = network->getLayer(i);
            const std::string name = layer->getName();
            if (fp32_set.contains(name)) {
                layer->setPrecision(nvinfer1::DataType::kFLOAT);
                const int n_outs = layer->getNbOutputs();
                for (int o = 0; o < n_outs; ++o) {
                    layer->setOutputType(o, nvinfer1::DataType::kFLOAT);
                }
                pinned.push_back(name);
            }
        }
        std::cerr << "Pinned " << pinned.size() << " tail layers to fp32:";
        for (const auto& n : pinned) std::cerr << ' ' << n;
        std::cerr << '\n';

        const int n_outs = network->getNbOutputs();
        for (int i = 0; i < n_outs; ++i) {
            auto* out = network->getOutput(i);
            if (opts.cov_output_name == out->getName()) {
                out->setType(nvinfer1::DataType::kFLOAT);
            }
        }
    } else {
        // PyTorch default is allow_tf32=False; match it to upper-bound parity.
        config->clearFlag(nvinfer1::BuilderFlag::kTF32);
        // Precise reference plan is batch=1 only on 8 GB cards.
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
