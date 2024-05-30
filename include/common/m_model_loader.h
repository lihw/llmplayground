/** 
* The abstract model loader class
*
* Author: lihw81@gmail.com
*/

#ifndef M_MODEL_LOADER_H
#define M_MODEL_LOADER_H

#include <common/m_defs.h>
#include <common/m_gguf.h>

#include <ggml/ggml.h>

#include <fmt/core.h>

#include <string>
#include <vector>
#include <stdexcept>


M_BEGIN_NAMESPACE


class ModelLoader {

    M_NO_COPY_CONSTRUCTOR(ModelLoader)
    M_NO_MOVE_CONSTRUCTOR(ModelLoader)

public:
    /**
     * The weight information of a model
    */
    struct Weight {
        uint16_t  idx; // source file index
        size_t   offs; // tensor data offset in the original file

        ggml_tensor * tensor;

        Weight(uint16_t idx, const char * name, const struct gguf_context * gguf_ctx, ggml_tensor * tensor) 
            : idx(idx)
            , tensor(tensor) {

            const int tensor_idx = gguf_find_tensor(gguf_ctx, name);
            offs = gguf_get_data_offset(gguf_ctx) + gguf_get_tensor_offset(gguf_ctx, tensor_idx);
        }
    };

public:
    explicit ModelLoader();

    virtual ~ModelLoader();

    virtual bool load(const std::string& file, bool useMmap) noexcept = 0;

    const char* getTensorName(int i) const noexcept;
    Weight* getWeight(const char * name) noexcept;
    ggml_tensor* getTensorMeta(const char* name) noexcept; 
    ggml_tensor* getTensorMeta(int i) noexcept;
    gguf_context* getContext() const noexcept 
    {
        return mMeta;
    }
    const std::string& getArchName() const noexcept
    {
        return mArchName;
    }

    template<typename T>
    bool getKey(const std::string& key, T& result, const bool required = true) 
    {
        const bool found = GGUFMeta::GKV<T>::set(mMeta, key, result, nullptr);

        if (required && !found) {
            spdlog::error("key not found in model: %s", key.c_str());
            return false;
        }

        return found;
    }

    template<typename T>
    bool getKey(const Kv kid, T& result, const bool required = true) {
        return getKey(Kv(kid), result, required);
    }

protected:
    size_t mNumKeyValues;
    size_t mNumTensors;
    size_t mNumElements;
    size_t mNumBytes;

    //bool mUseMmap = false;
    //llama_files files;
    //llama_mmaps mappings;
    //std::unordered_map<std::string, struct llama_model_kv_override> kv_overrides;
    
    std::vector<Weight>           mWeights;
    gguf_context*                 mMeta;
    std::vector<ggml_context*>    mContexts;
    
    std::string mArchName; //! The model arch
    //LLM_KV      llm_kv    = LLM_KV(LLM_ARCH_UNKNOWN);
};

M_END_NAMESPACE

#endif // !M_MODEL_LOADER_H

