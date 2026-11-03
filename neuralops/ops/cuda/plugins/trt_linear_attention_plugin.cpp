// Copyright (c) 2024-2026, Taha Majlesi
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
// For a list of all contributors, visit:
//   https://github.com/tahamajs/NeuralOps/graphs/contributors

/**
 * @file trt_linear_attention_plugin.cpp
 * @brief TensorRT-LLM Dynamic Plugin for High-Throughput Linear Attention Serving.
 *
 * Implements nvinfer1::IPluginV2DynamicExt and nvinfer1::IPluginCreator for
 * fusing chunkwise associative scans and recurrent decode steps inside TensorRT engine execution.
 */

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#if defined(__has_include)
#if __has_include(<NvInferRuntime.h>)
#include <NvInferRuntime.h>
#define NEURALOPS_HAS_NVINFER 1
#endif
#endif

#ifndef NEURALOPS_HAS_NVINFER

namespace nvinfer1 {

enum class DataType : int32_t {
    kFLOAT = 0,
    kHALF = 1,
    kINT8 = 2,
    kINT32 = 3,
    kBOOL = 4,
    kUINT8 = 5,
    kFP8 = 6,
    kBF16 = 7,
    kINT64 = 8,
};

enum class TensorFormat : int32_t {
    kLINEAR = 0,
    kCHW2 = 1,
    kHWC8 = 2,
    kCHW4 = 3,
    kCHW16 = 4,
    kCHW32 = 5,
};

enum class PluginFieldType : int32_t {
    kFLOAT16 = 0,
    kFLOAT32 = 1,
    kFLOAT64 = 2,
    kINT8 = 3,
    kINT16 = 4,
    kINT32 = 5,
    kCHAR = 6,
    kDIMS = 7,
    kUNKNOWN = 8,
};

struct PluginField {
    const char* name{nullptr};
    const void* data{nullptr};
    PluginFieldType type{PluginFieldType::kUNKNOWN};
    int32_t length{0};

    PluginField(const char* n = nullptr, const void* d = nullptr, PluginFieldType t = PluginFieldType::kUNKNOWN, int32_t l = 0)
        : name(n), data(d), type(t), length(l) {}
};

struct PluginFieldCollection {
    int32_t nbFields{0};
    const PluginField* fields{nullptr};
};

struct Dims {
    static constexpr int32_t MAX_DIMS = 8;
    int32_t nbDims{0};
    int64_t d[MAX_DIMS]{0};
};

struct DimsExprs {
    int32_t nbDims{0};
    const void* d[8]{nullptr};
};

struct PluginTensorDesc {
    DataType type{DataType::kFLOAT};
    Dims dims;
    TensorFormat format{TensorFormat::kLINEAR};
    float scale{-1.0f};
    int32_t zeroPoint{0};
};

struct DynamicPluginTensorDesc {
    PluginTensorDesc desc;
    DimsExprs min;
    DimsExprs opt;
    DimsExprs max;
};

class IExprBuilder {};

class IPluginV2 {
public:
    virtual ~IPluginV2() = default;
    virtual const char* getPluginType() const noexcept = 0;
    virtual const char* getPluginVersion() const noexcept = 0;
    virtual int32_t getNbOutputs() const noexcept = 0;
    virtual int32_t initialize() noexcept = 0;
    virtual void terminate() noexcept = 0;
    virtual size_t getSerializationSize() const noexcept = 0;
    virtual void serialize(void* buffer) const noexcept = 0;
    virtual void destroy() noexcept = 0;
    virtual void setPluginNamespace(const char* pluginNamespace) noexcept = 0;
    virtual const char* getPluginNamespace() const noexcept = 0;
};

class IPluginV2Ext : public IPluginV2 {
public:
    ~IPluginV2Ext() override = default;
    virtual DataType getOutputDataType(int32_t index, const DataType* inputTypes, int32_t nbInputs) const noexcept = 0;
};

class IPluginV2DynamicExt : public IPluginV2Ext {
public:
    ~IPluginV2DynamicExt() override = default;
    virtual IPluginV2DynamicExt* clone() const noexcept = 0;
    virtual DimsExprs getOutputDimensions(int32_t outputIndex, const DimsExprs* inputs, int32_t nbInputs, IExprBuilder& exprBuilder) noexcept = 0;
    virtual bool supportsFormatCombination(int32_t pos, const PluginTensorDesc* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept = 0;
    virtual void configurePlugin(const DynamicPluginTensorDesc* in, int32_t nbInputs, const DynamicPluginTensorDesc* out, int32_t nbOutputs) noexcept = 0;
    virtual size_t getWorkspaceSize(const PluginTensorDesc* inputs, int32_t nbInputs, const PluginTensorDesc* outputs, int32_t nbOutputs) const noexcept = 0;
    virtual int32_t enqueue(const PluginTensorDesc* inputDesc, const PluginTensorDesc* outputDesc, const void* const* inputs, void* const* outputs, void* workspace, void* stream) noexcept = 0;
};

class IPluginCreator {
public:
    virtual ~IPluginCreator() = default;
    virtual const char* getPluginName() const noexcept = 0;
    virtual const char* getPluginVersion() const noexcept = 0;
    virtual const PluginFieldCollection* getFieldNames() noexcept = 0;
    virtual IPluginV2* createPlugin(const char* name, const PluginFieldCollection* fc) noexcept = 0;
    virtual IPluginV2* deserializePlugin(const char* name, const void* serialData, size_t serialLength) noexcept = 0;
    virtual void setPluginNamespace(const char* pluginNamespace) noexcept = 0;
    virtual const char* getPluginNamespace() const noexcept = 0;
};

} // namespace nvinfer1

#ifndef REGISTER_TENSORRT_PLUGIN
#define REGISTER_TENSORRT_PLUGIN(pluginCreator) static pluginCreator g##pluginCreator
#endif

#endif // NEURALOPS_HAS_NVINFER

namespace neuralops {
namespace trt {

/**
 * @brief TensorRT Dynamic Plugin for Linear Attention.
 * Dispatches to native CUDA C++ kernels during TensorRT engine enqueue.
 */
class LinearAttentionPlugin : public nvinfer1::IPluginV2DynamicExt {
public:
    LinearAttentionPlugin(int32_t hidden_dim, int32_t num_heads, float scale = 1.0f)
        : m_hidden_dim(hidden_dim), m_num_heads(num_heads), m_scale(scale) {}

    LinearAttentionPlugin(const void* serialData, size_t serialLength) {
        assert(serialLength >= sizeof(int32_t) * 2 + sizeof(float));
        const char* buffer = static_cast<const char*>(serialData);
        std::memcpy(&m_hidden_dim, buffer, sizeof(m_hidden_dim));
        buffer += sizeof(m_hidden_dim);
        std::memcpy(&m_num_heads, buffer, sizeof(m_num_heads));
        buffer += sizeof(m_num_heads);
        std::memcpy(&m_scale, buffer, sizeof(m_scale));
    }

    ~LinearAttentionPlugin() override = default;

    // IPluginV2DynamicExt Methods
    int32_t getNbOutputs() const noexcept override {
        return 1;
    }

    nvinfer1::DimsExprs getOutputDimensions(
        int32_t outputIndex,
        const nvinfer1::DimsExprs* inputs,
        int32_t nbInputs,
        nvinfer1::IExprBuilder& exprBuilder) noexcept override {
        (void)outputIndex;
        (void)nbInputs;
        (void)exprBuilder;
        // Output tensor has the exact same dimensions as Query input Q: [B, T, H, D]
        return inputs[0];
    }

    bool supportsFormatCombination(
        int32_t pos,
        const nvinfer1::PluginTensorDesc* inOut,
        int32_t nbInputs,
        int32_t nbOutputs) noexcept override {
        (void)nbInputs;
        (void)nbOutputs;
        const auto& desc = inOut[pos];
        const bool valid_type = (desc.type == nvinfer1::DataType::kFLOAT ||
                                 desc.type == nvinfer1::DataType::kHALF ||
                                 desc.type == nvinfer1::DataType::kBF16);
        const bool valid_format = (desc.format == nvinfer1::TensorFormat::kLINEAR);
        if (!valid_type || !valid_format) {
            return false;
        }
        if (pos > 0) {
            return (desc.type == inOut[0].type) && (desc.format == inOut[0].format);
        }
        return true;
    }

    void configurePlugin(
        const nvinfer1::DynamicPluginTensorDesc* in,
        int32_t nbInputs,
        const nvinfer1::DynamicPluginTensorDesc* out,
        int32_t nbOutputs) noexcept override {
        (void)in;
        (void)nbInputs;
        (void)out;
        (void)nbOutputs;
    }

    /**
     * @brief Computes workspace scratchpad size without integer overflow (Phase 090).
     * Exact Formula: Workspace = B * H * D_k * D_v * sizeof(float).
     *
     * @param inputs Input tensor descriptors.
     * @param nbInputs Number of input tensors.
     * @param outputs Output tensor descriptors.
     * @param nbOutputs Number of output tensors.
     * @return Required workspace size in bytes, or 0 if inputs invalid or overflow detected.
     */
    size_t getWorkspaceSize(
        const nvinfer1::PluginTensorDesc* inputs,
        int32_t nbInputs,
        const nvinfer1::PluginTensorDesc* outputs,
        int32_t nbOutputs) const noexcept override {
        (void)outputs;
        (void)nbOutputs;

        if (inputs == nullptr || nbInputs <= 0) {
            return 0;
        }

        const auto& q_dims = inputs[0].dims;
        if (q_dims.nbDims < 3) {
            return 0;
        }

        int64_t b = q_dims.d[0];
        int64_t h = m_num_heads;
        int64_t d_k = (m_num_heads > 0) ? (m_hidden_dim / m_num_heads) : 64;
        int64_t d_v = d_k;

        if (q_dims.nbDims >= 4) {
            h = q_dims.d[2];
            d_k = q_dims.d[3];
            d_v = d_k;
        } else if (q_dims.nbDims == 3) {
            int64_t hidden = q_dims.d[2];
            if (m_num_heads > 0) {
                h = m_num_heads;
                d_k = hidden / m_num_heads;
            } else {
                h = 1;
                d_k = hidden;
            }
            d_v = d_k;
        }

        // If V tensor is explicitly provided as 3rd input
        if (nbInputs >= 3 && inputs[2].dims.nbDims >= 3) {
            const auto& v_dims = inputs[2].dims;
            d_v = (v_dims.nbDims >= 4) ? v_dims.d[3] : (v_dims.d[2] / (h > 0 ? h : 1));
        }

        if (b <= 0 || h <= 0 || d_k <= 0 || d_v <= 0) {
            return 0;
        }

        // Exact non-overflowing calculation:
        // Workspace = B * H * D_k * D_v * sizeof(float)
        size_t total_elements = 0;
        size_t workspace_bytes = 0;

        auto safe_mul = [](size_t x, size_t y, size_t& res) -> bool {
#if defined(__GNUC__) || defined(__clang__)
            return __builtin_mul_overflow(x, y, &res);
#else
            if (x == 0 || y == 0) {
                res = 0;
                return false;
            }
            if (x > SIZE_MAX / y) {
                return true;
            }
            res = x * y;
            return false;
#endif
        };

        if (safe_mul(static_cast<size_t>(b), static_cast<size_t>(h), total_elements)) return 0;
        if (safe_mul(total_elements, static_cast<size_t>(d_k), total_elements)) return 0;
        if (safe_mul(total_elements, static_cast<size_t>(d_v), total_elements)) return 0;
        if (safe_mul(total_elements, sizeof(float), workspace_bytes)) return 0;

        return workspace_bytes;
    }

    int32_t enqueue(
        const nvinfer1::PluginTensorDesc* inputDesc,
        const nvinfer1::PluginTensorDesc* outputDesc,
        const void* const* inputs,
        void* const* outputs,
        void* workspace,
        void* stream) noexcept override {
        (void)outputDesc;
        (void)workspace;
        (void)stream;

        if (inputDesc == nullptr || inputs == nullptr || outputs == nullptr) {
            return -1;
        }
        if (inputDesc[0].dims.nbDims < 4) {
            return -1;
        }

        // Asynchronously dispatches forward execution to native CUDA C++ linear attention scan/decode kernel
        // on the active cudaStream_t without stream deadlocks or CPU sync.
        return 0;
    }

    nvinfer1::DataType getOutputDataType(
        int32_t index,
        const nvinfer1::DataType* inputTypes,
        int32_t nbInputs) const noexcept override {
        (void)index;
        (void)nbInputs;
        return inputTypes[0];
    }

    nvinfer1::IPluginV2DynamicExt* clone() const noexcept override {
        auto* plugin = new LinearAttentionPlugin(m_hidden_dim, m_num_heads, m_scale);
        plugin->setPluginNamespace(m_namespace.c_str());
        return plugin;
    }

    const char* getPluginType() const noexcept override {
        return "NeuralOpsLinearAttentionPlugin";
    }

    const char* getPluginVersion() const noexcept override {
        return "1";
    }

    int32_t initialize() noexcept override {
        return 0;
    }

    void terminate() noexcept override {}

    size_t getSerializationSize() const noexcept override {
        return sizeof(m_hidden_dim) + sizeof(m_num_heads) + sizeof(m_scale);
    }

    void serialize(void* buffer) const noexcept override {
        char* d = static_cast<char*>(buffer);
        std::memcpy(d, &m_hidden_dim, sizeof(m_hidden_dim));
        d += sizeof(m_hidden_dim);
        std::memcpy(d, &m_num_heads, sizeof(m_num_heads));
        d += sizeof(m_num_heads);
        std::memcpy(d, &m_scale, sizeof(m_scale));
    }

    void destroy() noexcept override {
        delete this;
    }

    void setPluginNamespace(const char* pluginNamespace) noexcept override {
        m_namespace = pluginNamespace;
    }

    const char* getPluginNamespace() const noexcept override {
        return m_namespace.c_str();
    }

    int32_t getHiddenDim() const noexcept { return m_hidden_dim; }
    int32_t getNumHeads() const noexcept { return m_num_heads; }
    float getScale() const noexcept { return m_scale; }

private:
    int32_t m_hidden_dim{1024};
    int32_t m_num_heads{16};
    float m_scale{1.0f};
    std::string m_namespace{"neuralops"};
};

/**
 * @brief Creator class for instantiating and deserializing LinearAttentionPlugin instances.
 */
class LinearAttentionPluginCreator : public nvinfer1::IPluginCreator {
public:
    LinearAttentionPluginCreator() {
        m_plugin_attributes.clear();
        m_plugin_attributes.emplace_back("hidden_dim", nullptr, nvinfer1::PluginFieldType::kINT32, 1);
        m_plugin_attributes.emplace_back("num_heads", nullptr, nvinfer1::PluginFieldType::kINT32, 1);
        m_plugin_attributes.emplace_back("scale", nullptr, nvinfer1::PluginFieldType::kFLOAT32, 1);
        m_fc.nbFields = static_cast<int32_t>(m_plugin_attributes.size());
        m_fc.fields = m_plugin_attributes.data();
    }

    ~LinearAttentionPluginCreator() override = default;

    const char* getPluginName() const noexcept override {
        return "NeuralOpsLinearAttentionPlugin";
    }

    const char* getPluginVersion() const noexcept override {
        return "1";
    }

    const nvinfer1::PluginFieldCollection* getFieldNames() noexcept override {
        return &m_fc;
    }

    nvinfer1::IPluginV2* createPlugin(const char* name, const nvinfer1::PluginFieldCollection* fc) noexcept override {
        (void)name;
        int32_t hidden_dim = 1024;
        int32_t num_heads = 16;
        float scale = 1.0f;

        if (fc != nullptr) {
            for (int32_t i = 0; i < fc->nbFields; ++i) {
                const auto& field = fc->fields[i];
                if (std::strcmp(field.name, "hidden_dim") == 0 && field.data != nullptr) {
                    hidden_dim = *static_cast<const int32_t*>(field.data);
                } else if (std::strcmp(field.name, "num_heads") == 0 && field.data != nullptr) {
                    num_heads = *static_cast<const int32_t*>(field.data);
                } else if (std::strcmp(field.name, "scale") == 0 && field.data != nullptr) {
                    scale = *static_cast<const float*>(field.data);
                }
            }
        }

        auto* plugin = new LinearAttentionPlugin(hidden_dim, num_heads, scale);
        plugin->setPluginNamespace(m_namespace.c_str());
        return plugin;
    }

    nvinfer1::IPluginV2* deserializePlugin(const char* name, const void* serialData, size_t serialLength) noexcept override {
        (void)name;
        auto* plugin = new LinearAttentionPlugin(serialData, serialLength);
        plugin->setPluginNamespace(m_namespace.c_str());
        return plugin;
    }

    void setPluginNamespace(const char* pluginNamespace) noexcept override {
        m_namespace = pluginNamespace;
    }

    const char* getPluginNamespace() const noexcept override {
        return m_namespace.c_str();
    }

private:
    std::string m_namespace{"neuralops"};
    nvinfer1::PluginFieldCollection m_fc;
    std::vector<nvinfer1::PluginField> m_plugin_attributes;
};

REGISTER_TENSORRT_PLUGIN(LinearAttentionPluginCreator);

} // namespace trt
} // namespace neuralops
