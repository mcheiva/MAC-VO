#pragma once
//
// Header-only TensorRT engine builder for the MACVO FlowFormerCov frontend.
//
// Mirrors MACVO_TRT_Build.py / MACVO_TRT_Build_FP32.py:
//   * Mode::Fast    -> strongly typed (fp16, bf16) plan, weight streaming,
//                      TF32 allowed. The fp32 layer pin set is empty by
//                      default (see fp32_layer_names): the cov tail is
//                      applied by the MACVO consumer, not in the ONNX.
//   * Mode::Precise -> FP32 plan, TF32 disabled to match PyTorch's default
//                      allow_tf32=False matmul behaviour. Batch=1 only.
//
// Static batch 1 is the norm: export the ONNX with static shapes (batch 1)
// and pass no shape flags. When every network input has fully static dims,
// no optimization profile is created and --min/opt/max-batch, --height and
// --width are ignored. A profile (min/opt/max, default 1/1/1) is added only
// if an input is dynamic.
//
// Equivalent trtexec command for a static ONNX (no shape flags):
//   trtexec --onnx=model_b1.onnx --saveEngine=model_b1.plan --stronglyTyped
//     --allowWeightStreaming --builderOptimizationLevel=5 --avgTiming=8
//     --profilingVerbosity=detailed --timingCacheFile=timing.cache
//   (add --errorOnTimingCacheMiss for --error-on-timing-cache-miss)
//
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime_api.h>

#include <array>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <set>
#include <sstream>
#include <system_error>
#include <stdexcept>
#include <string>
#include <vector>

namespace macvo_trt {

enum class Mode { Fast, Precise };

struct BuildOptions {
    std::filesystem::path onnx_path;
    std::filesystem::path engine_path;
    Mode mode = Mode::Fast;

    // Optimization profile (NCHW), used only if an ONNX input is dynamic.
    // Static batch 1 by default; ignored for a fully static ONNX.
    std::int32_t min_batch = 1;
    std::int32_t opt_batch = 1;
    std::int32_t max_batch = 1;
    std::int32_t channels  = 3;
    std::int32_t height    = 704;
    std::int32_t width     = 704;

    // The deployed EIVA engine (MACVO_TRT commit 9842f44) and the static
    // batch-1 engines were built at level 5 and validated end to end on the
    // VO eval. MAC-VO issue #18 reported corrupted optical-flow values at
    // higher levels on its build.
    int optimization_level = 5;

    // Number of timing runs averaged per tactic. trtexec's default is 8 and
    // the API's default is 1. Only affects timing-cache misses; the existing
    // production tactics were timed at 8.
    int avg_timing_iterations = 8;

    // Tiling optimization level, 0-3 (kNONE, kFAST, kMODERATE, kFULL).
    // 0 is TensorRT's default.
    int tiling_level = 0;

    // Sets BuilderFlag::kERROR_ON_TIMING_CACHE_MISS. In TRT 10.13.2 a miss
    // does not fail the build: the logger gets one multi-line warning,
    // "Engine generation failed with backend strategy N. Error message:
    // ... Adding new entry to timing cache is not allowed if
    // `BuilderFlag::kERROR_ON_TIMING_CACHE_MISS` is set", and TensorRT falls
    // back to another strategy and writes a plan. So Build() fails if (a) the
    // timing cache, serialized right after the build and before the on-disk
    // merge, is larger than the bytes loaded (TensorRT timed new entries; the
    // primary signal, needs --timing-cache), or (b) the logger saw a
    // kERROR / kINTERNAL_ERROR or that cache-miss warning. Use it to
    // reproduce a plan exactly from a populated timing cache.
    bool error_on_timing_cache_miss = false;

    // Parse the ONNX, report whether a profile would be added and return
    // without building.
    bool parse_only = false;

    // kWEIGHT_STREAMING, measured on the EIVA gaps engine, RTX 4090, TRT
    // 10.13.2. On the dynamic-axis ONNX (the old graph) it forces the
    // all-Myelin graph: 942 layers vs 3222, 26.3 vs 38.4 ms per call with a
    // CUDA graph, 28.4 vs 54.2 ms without. On the static batch-1 ONNX that
    // Export.py now produces the flag changes nothing: identical tactic
    // lists, 964 layers, a single stream, the same plan size and
    // 23.92-23.98 ms either way. Default on, so a dynamic-axis build also
    // gets the fast graph. At the default (full) budget nothing is streamed
    // at runtime; the cost is a host copy of the weights at deserialization
    // (a TensorRT warning). Accuracy: plane_tail1 was 0.036 without the flag
    // vs 0.026 with it, outside the observed 0.023-0.028 spread but from a
    // single dive, so weak evidence; the reason for the default is speed on
    // the dynamic graph.
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

    // Timing cache file. Loaded if it exists, merged with any newer on-disk
    // copy and written back after a successful build. In practice, reusing
    // it makes tactic choice repeatable across builds with the same flags
    // and GPU; TensorRT does not guarantee it (NvInfer.h notes the same
    // cache may not give stable builds in all cases). Check by diffing
    // TacticName in `trtexec --dumpLayerInfo` of two plans.
    // Empty = no cache file.
    std::filesystem::path timing_cache_path;

    bool verbose = false;
};

class Logger : public nvinfer1::ILogger {
public:
    explicit Logger(bool verbose) noexcept : verbose_(verbose) {}
    void log(Severity s, char const* msg) noexcept override {
        using S = Severity;
        if (s == S::kINTERNAL_ERROR || s == S::kERROR) {
            errors_.fetch_add(1);
            std::cerr << "[TRT ERROR] " << msg << '\n';
        } else if (s == S::kWARNING) {
            // One multi-line message: "Engine generation failed with backend
            // strategy N. Error message: ... Adding new entry to timing
            // cache is not allowed if `BuilderFlag::kERROR_ON_TIMING_CACHE_MISS`
            // is set ...".
            if (std::strstr(msg, "kERROR_ON_TIMING_CACHE_MISS") != nullptr ||
                std::strstr(msg, "Adding new entry to timing cache") != nullptr) {
                strategy_failures_.fetch_add(1);
            }
            std::cerr << "[TRT WARN ] " << msg << '\n';
        } else if (verbose_) {
            std::cerr << "[TRT INFO ] " << msg << '\n';
        }
    }
    // kERROR + kINTERNAL_ERROR messages seen so far.
    int error_count() const noexcept { return errors_.load(); }
    // Warnings carrying the timing-cache-miss text: what a miss produces
    // under kERROR_ON_TIMING_CACHE_MISS (the size check in Build() is the
    // primary signal; this is the secondary one).
    int strategy_failure_count() const noexcept { return strategy_failures_.load(); }
private:
    bool verbose_;
    std::atomic<int> errors_{0};
    std::atomic<int> strategy_failures_{0};
};

template <typename T>
using TrtPtr = std::unique_ptr<T>;

// JSON string literal with backslashes, quotes and control chars escaped.
inline std::string JsonString(const std::string& in) {
    std::ostringstream o;
    o << '"';
    for (const unsigned char c : in) {
        switch (c) {
            case '\\': o << "\\\\"; break;
            case '"':  o << "\\\""; break;
            case '\n': o << "\\n";  break;
            case '\r': o << "\\r";  break;
            case '\t': o << "\\t";  break;
            default:
                if (c < 0x20) {
                    char buf[8];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", c);
                    o << buf;
                } else {
                    o << static_cast<char>(c);
                }
        }
    }
    o << '"';
    return o.str();
}

// UTF-8 JSON string for a path (u8string() so the JSON is valid UTF-8 on MSVC).
inline std::string JsonPath(const std::filesystem::path& p) {
    const auto u8 = p.u8string();
    return JsonString(std::string(u8.begin(), u8.end()));
}

// 64-bit FNV-1a of a file's bytes, as 16 hex digits. Empty on read failure.
// Identifies the model: different exports can share the same byte size.
inline std::string Fnv1aHex(const std::filesystem::path& p) {
    std::ifstream f(p, std::ios::binary);
    if (!f) return {};
    std::uint64_t h = 14695981039346656037ULL;
    std::vector<char> buf(1 << 20);
    while (f) {
        f.read(buf.data(), static_cast<std::streamsize>(buf.size()));
        const std::streamsize n = f.gcount();
        for (std::streamsize i = 0; i < n; ++i) {
            h ^= static_cast<unsigned char>(buf[static_cast<std::size_t>(i)]);
            h *= 1099511628211ULL;
        }
    }
    if (f.bad()) return {};
    char out[17];
    std::snprintf(out, sizeof(out), "%016llx", static_cast<unsigned long long>(h));
    return out;
}

// What happened to the timing cache during one Build(), for the sidecar.
struct CacheReport {
    bool used = false;            // a timing cache path was given
    bool loaded = false;          // file existed and was non-empty
    std::size_t bytes_loaded = 0;
    std::size_t bytes_saved = 0;
};

// True if every dimension of every network input is fixed (no -1).
inline bool AllInputsStatic(const nvinfer1::INetworkDefinition& network) {
    for (int i = 0; i < network.getNbInputs(); ++i) {
        const nvinfer1::Dims d = network.getInput(i)->getDimensions();
        for (int k = 0; k < d.nbDims; ++k) {
            if (d.d[k] < 0) return false;
        }
    }
    return true;
}

// Writes <engine_path>.json next to the plan: what the plan was built from
// and with, so a plan can be traced back without guessing.
inline void WriteSidecar(const BuildOptions& opts, const CacheReport& cache,
                         bool is_static) {
    cudaDeviceProp prop{};
    int dev = 0;
    // JSON null when no GPU is visible (the Python builder writes null too).
    std::string gpu_name = "null";
    std::string cc = "null";
    if (cudaGetDevice(&dev) == cudaSuccess &&
        cudaGetDeviceProperties(&prop, dev) == cudaSuccess) {
        gpu_name = JsonString(prop.name);
        cc = JsonString(std::to_string(prop.major) + "." + std::to_string(prop.minor));
    }

    std::error_code ec;
    const auto onnx_size = std::filesystem::file_size(opts.onnx_path, ec);
    const std::string onnx_hash = Fnv1aHex(opts.onnx_path);

    // getInferLibVersion() = (MAJOR * 100 + MINOR) * 100 + PATCH.
    const int v = ::getInferLibVersion();
    const std::string trt_str = std::to_string(v / 10000) + "."
        + std::to_string((v / 100) % 100) + "." + std::to_string(v % 100);

    std::filesystem::path sidecar = opts.engine_path;
    sidecar += ".json";
    std::ofstream j(sidecar);
    if (!j) throw std::runtime_error("Cannot open sidecar: " + sidecar.string());
    j << "{\n"
      << "  \"onnx_path\": " << JsonPath(std::filesystem::absolute(opts.onnx_path)) << ",\n"
      << "  \"onnx_size_bytes\": "
      << (ec ? std::string("null") : std::to_string(static_cast<std::uintmax_t>(onnx_size))) << ",\n"
      << "  \"onnx_fnv1a64\": "
      << (onnx_hash.empty() ? std::string("null") : "\"" + onnx_hash + "\"") << ",\n"
      << "  \"mode\": \"" << (opts.mode == Mode::Fast ? "fast" : "precise") << "\",\n"
      << "  \"static\": " << (is_static ? "true" : "false") << ",\n"
      << "  \"min_batch\": " << (is_static ? std::string("null") : std::to_string(opts.min_batch)) << ",\n"
      << "  \"opt_batch\": " << (is_static ? std::string("null") : std::to_string(opts.opt_batch)) << ",\n"
      << "  \"max_batch\": " << (is_static ? std::string("null") : std::to_string(opts.max_batch)) << ",\n"
      << "  \"height\": " << (is_static ? std::string("null") : std::to_string(opts.height)) << ",\n"
      << "  \"width\": " << (is_static ? std::string("null") : std::to_string(opts.width)) << ",\n"
      << "  \"optimization_level\": " << opts.optimization_level << ",\n"
      << "  \"avg_timing_iterations\": " << opts.avg_timing_iterations << ",\n"
      << "  \"tiling_optimization_level\": " << opts.tiling_level << ",\n"
      << "  \"error_on_timing_cache_miss\": " << (opts.error_on_timing_cache_miss ? "true" : "false") << ",\n"
      << "  \"weight_streaming\": " << (opts.weight_streaming ? "true" : "false") << ",\n"
      << "  \"allow_tf32\": " << (opts.allow_tf32 ? "true" : "false") << ",\n"
      << "  \"timing_cache\": "
      << (cache.used ? JsonPath(std::filesystem::absolute(opts.timing_cache_path))
                     : std::string("null")) << ",\n"
      << "  \"timing_cache_status\": "
      << (!cache.used ? std::string("null")
                      : std::string(cache.loaded ? "\"loaded\"" : "\"started_empty\"")) << ",\n"
      << "  \"timing_cache_bytes_loaded\": "
      << (cache.used ? std::to_string(cache.bytes_loaded) : std::string("null")) << ",\n"
      << "  \"timing_cache_bytes_saved\": "
      << (cache.used ? std::to_string(cache.bytes_saved) : std::string("null")) << ",\n"
      << "  \"tensorrt_version\": " << v << ",\n"
      << "  \"tensorrt_version_str\": \"" << trt_str << "\",\n"
      << "  \"tensorrt_build_header\": " << NV_TENSORRT_BUILD << ",\n"
      << "  \"gpu_name\": " << gpu_name << ",\n"
      << "  \"compute_capability\": " << cc << "\n"
      << "}\n";
    if (!j) throw std::runtime_error("Failed to write: " + sidecar.string());
    std::cerr << "Wrote " << sidecar << '\n';
}

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

    // The timing cache must outlive every builder that uses it (NvInfer.h),
    // so it is declared before the builder and destroyed after it.
    TrtPtr<nvinfer1::ITimingCache> timing_cache;
    CacheReport cache_report;

    // A stale sidecar next to a failed build would describe the wrong plan.
    if (!opts.parse_only) {
        std::filesystem::path stale = opts.engine_path;
        stale += ".json";
        std::error_code ec;
        std::filesystem::remove(stale, ec);
    }
    if (!opts.parse_only && !opts.timing_cache_path.empty()) {
        const auto parent = opts.timing_cache_path.parent_path();
        if (!parent.empty() && !std::filesystem::is_directory(parent)) {
            throw std::runtime_error("Timing cache directory does not exist: "
                                     + parent.string());
        }
    }

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

    const bool is_static = AllInputsStatic(*network);
    if (opts.parse_only) {
        std::cout << "parsed " << opts.onnx_path.string() << ": inputs "
                  << (is_static ? "all static" : "dynamic")
                  << ", optimization profile would be "
                  << (is_static ? "NOT added" : "added") << '\n';
        return;
    }

    TrtPtr<nvinfer1::IBuilderConfig> config{ builder->createBuilderConfig() };
    if (!config) throw std::runtime_error("createBuilderConfig failed");
    config->setBuilderOptimizationLevel(opts.optimization_level);
    config->setAvgTimingIterations(opts.avg_timing_iterations);
    config->setTilingOptimizationLevel(
        static_cast<nvinfer1::TilingOptimizationLevel>(opts.tiling_level));
    if (opts.error_on_timing_cache_miss) {
        config->setFlag(nvinfer1::BuilderFlag::kERROR_ON_TIMING_CACHE_MISS);
    }

    if (opts.weight_streaming) {
        // Makes the builder offload every Myelin-supported op to Myelin
        // (one graph instead of many islands); see BuildOptions.
        config->setFlag(nvinfer1::BuilderFlag::kWEIGHT_STREAMING);
    }

    // Makes the plan inspectable (per-layer precision/format/tactic). Also
    // raises the NVTX verbosity, which TensorRT says may add enqueueV3
    // latency; consumers not using CUDA graphs can call
    // context->setNvtxVerbosity(kNONE) to remove it. No difference was
    // measurable on this model.
    config->setProfilingVerbosity(nvinfer1::ProfilingVerbosity::kDETAILED);

    // Timing cache: reuse tactic choices from earlier builds.
    if (!opts.timing_cache_path.empty()) {
        std::vector<char> cache_data;
        std::ifstream cf(opts.timing_cache_path, std::ios::binary | std::ios::ate);
        if (cf) {
            const auto n = static_cast<std::streamsize>(cf.tellg());
            cache_data.resize(static_cast<std::size_t>(n));
            cf.seekg(0);
            cf.read(cache_data.data(), n);
            if (!cf) throw std::runtime_error(
                "Failed to read timing cache: " + opts.timing_cache_path.string());
        }
        const bool have_cache = !cache_data.empty();
        cache_report.used = true;
        cache_report.loaded = have_cache;
        cache_report.bytes_loaded = cache_data.size();
        timing_cache.reset(config->createTimingCache(
            have_cache ? cache_data.data() : nullptr, cache_data.size()));
        if (!timing_cache) throw std::runtime_error("createTimingCache failed");
        if (!config->setTimingCache(*timing_cache, /*ignoreMismatch=*/false)) {
            throw std::runtime_error("setTimingCache failed (cache was built for different "
                                     "device properties?): " + opts.timing_cache_path.string());
        }
        if (have_cache) {
            std::cerr << "Timing cache loaded: " << opts.timing_cache_path
                      << " (" << cache_data.size() << " bytes)\n";
        } else {
            std::cerr << "Timing cache not found, starting empty: "
                      << opts.timing_cache_path << '\n';
        }
    }

    if (!opts.allow_tf32) {
        // TF32 truncates fp32 matmul mantissa to 10 bits. Disabling it
        // matches PyTorch's allow_tf32=False default. Default for
        // Mode::Precise; harmless for Mode::Fast (no fp32 ops to TF32-ize).
        config->clearFlag(nvinfer1::BuilderFlag::kTF32);
    }

    if (is_static) {
        std::cerr << "ONNX inputs are static: no optimization profile; "
                     "--min/opt/max-batch, --height and --width are ignored\n";
    } else {
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
        if (config->addOptimizationProfile(profile) < 0) {
            throw std::runtime_error("addOptimizationProfile failed (invalid profile)");
        }
    }

    std::cerr << "Building TRT engine (mode="
              << (opts.mode == Mode::Fast ? "fast" : "precise")
              << "). This takes several minutes...\n";

    TrtPtr<nvinfer1::IHostMemory> plan{
        builder->buildSerializedNetwork(*network, *config) };
    if (!plan) throw std::runtime_error("buildSerializedNetwork returned null");
    if (opts.error_on_timing_cache_miss && timing_cache) {
        // Primary signal: the cache as this build leaves it, before the
        // on-disk merge, is larger than what was loaded, so TensorRT timed
        // tactics that were not in the cache.
        TrtPtr<nvinfer1::IHostMemory> own{ timing_cache->serialize() };
        if (!own) throw std::runtime_error("timing cache serialize failed");
        if (own->size() > cache_report.bytes_loaded) {
            throw std::runtime_error(
                "--error-on-timing-cache-miss: the build's timing cache grew "
                "from " + std::to_string(cache_report.bytes_loaded) + " to "
                + std::to_string(own->size()) + " bytes (TensorRT timed new "
                  "entries); plan not written");
        }
    }
    if (opts.error_on_timing_cache_miss
        && (logger.error_count() > 0 || logger.strategy_failure_count() > 0)) {
        throw std::runtime_error(
            "--error-on-timing-cache-miss: the logger saw "
            + std::to_string(logger.error_count()) + " error(s) and "
            + std::to_string(logger.strategy_failure_count())
            + " timing-cache-miss warning(s) during the build; plan not "
              "written");
    }

    std::ofstream f(opts.engine_path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open output: " + opts.engine_path.string());
    f.write(static_cast<const char*>(plan->data()),
            static_cast<std::streamsize>(plan->size()));
    if (!f) throw std::runtime_error("Failed to write: " + opts.engine_path.string());

    std::cerr << "Wrote " << opts.engine_path << " ("
              << (plan->size() / (1024.0 * 1024.0)) << " MiB)\n";

    if (timing_cache) {
        // Another build (e.g. on the other GPU) may have saved since we
        // loaded: merge its entries so neither run's work is lost.
        std::ifstream cf(opts.timing_cache_path, std::ios::binary | std::ios::ate);
        if (cf) {
            const auto n = static_cast<std::streamsize>(cf.tellg());
            std::vector<char> disk(static_cast<std::size_t>(n));
            cf.seekg(0);
            cf.read(disk.data(), n);
            if (cf && n > 0) {
                TrtPtr<nvinfer1::ITimingCache> on_disk{
                    config->createTimingCache(disk.data(), disk.size()) };
                if (!on_disk || !timing_cache->combine(*on_disk, /*ignoreMismatch=*/false)) {
                    std::cerr << "Timing cache: could not merge the on-disk copy, "
                                 "overwriting it with this build's cache\n";
                }
            }
        }
    }
    TrtPtr<nvinfer1::IHostMemory> cache_blob;
    if (timing_cache) {
        cache_blob.reset(timing_cache->serialize());
        if (!cache_blob) throw std::runtime_error("timing cache serialize failed");
        cache_report.bytes_saved = cache_blob->size();
    }

    // Sidecar goes right after the plan, before the cache is touched.
    WriteSidecar(opts, cache_report, is_static);

    if (cache_blob) {
        std::filesystem::path tmp = opts.timing_cache_path;
        tmp += ".tmp";
        {
            std::ofstream cf(tmp, std::ios::binary);
            if (!cf) throw std::runtime_error(
                "Cannot open timing cache for write: " + tmp.string());
            cf.write(static_cast<const char*>(cache_blob->data()),
                     static_cast<std::streamsize>(cache_blob->size()));
            if (!cf) throw std::runtime_error(
                "Failed to write timing cache: " + tmp.string());
        }
        // rename() replaces an existing file on Windows too.
        std::filesystem::rename(tmp, opts.timing_cache_path);
        std::cerr << "Timing cache saved: " << opts.timing_cache_path
                  << " (" << cache_blob->size() << " bytes)\n";
    }
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
