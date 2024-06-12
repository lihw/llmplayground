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

class Model;

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


    struct Parameters {
        int32_t numGpuLayers = 0; //! number of layers to store in VRAM
        //enum llama_split_mode split_mode; // how to split the model across multiple GPUs

        // main_gpu interpretation depends on split_mode:
        // LLAMA_SPLIT_NONE: the GPU that is used for the entire model
        // LLAMA_SPLIT_ROW: the GPU that is used for small tensors and intermediate results
        // LLAMA_SPLIT_LAYER: ignored
        int32_t mainGpu = -1;

        // proportion of the model (layers or rows) to offload to each GPU, size: llama_max_devices()
        //const float* tensor_split;

        // Called with a progress value between 0.0 and 1.0. Pass NULL to disable.
        // If the provided progress_callback returns true, model loading continues.
        // If it returns false, model loading is immediately aborted.
        //llama_progress_callback progress_callback;

        // context pointer passed to the progress callback
        //void* progress_callback_user_data;

        // override key-value pairs of the model meta data
        //const struct llama_model_kv_override* kv_overrides;

        // Keep the booleans together to avoid misalignment during copy-by-value.
        bool vocabOnly = false; // only load the vocabulary, no weights
        //bool use_mmap;   // use mmap if possible
        bool useMemoryLock = true;  // force system to keep model in RAM
    } params;

public:
    explicit ModelLoader();

    virtual ~ModelLoader();

    virtual bool load(const std::string& file) noexcept = 0;

    size_t getTensorCount() const noexcept
    {
        return mNumTensors;
    }
    const char* getTensorName(int i) const noexcept;
    const Weight* getWeight(const char * name) const noexcept;
    ggml_tensor* getTensorMeta(const char* name) const noexcept; 
    ggml_tensor* getTensorMeta(int i) const noexcept;
    gguf_context* getContext() const noexcept 
    {
        return mMeta;
    }
    const std::string& getArchName() const noexcept
    {
        return mArchName;
    }

    const std::vector<std::string>& getFiles() const noexcept
    {
        return mFiles;
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
        return getKey(getKvString(kid, mArchName), result, required);
    }


    template<typename T>
    typename std::enable_if<std::is_integral<T>::value, bool>::type
    getArrayLength(const std::string & key, T & result, const bool required = true) {
        const int kid = gguf_find_key(mMeta, key.c_str());

        if (kid < 0) {
            if (required) {
                throw std::runtime_error(fmt::format("key not found in model: {}", key.c_str()));
            }
            return false;
        }

        struct GGUFMeta::ArrayInfo arr_info =
            GGUFMeta::GKV<GGUFMeta::ArrayInfo>::get_kv(mMeta, kid);


        result = arr_info.length;
        return true;
    }

    template<typename T>
    typename std::enable_if<std::is_integral<T>::value, bool>::type
    getArrayLength(const Kv kid, T & result, const bool required = true) {
        return getArrayLength(getKvString(kid, mArchName), result, required);
    }

    /**
     * Create a model object after a successful loading 
    */
    virtual Model* build() noexcept = 0;

    /**
     * 
    */
    ggml_tensor* createTensorFor(ggml_context *ctx, const ggml_tensor *cur);
    /**
     * 
    */
    const ggml_tensor* checkTensorDims(const std::string &name, const std::vector<int64_t> &ne, bool required) const;
    /**
     * 
    */
    ggml_tensor *createTensor(ggml_context *ctx,
        const std::string &name,
        const std::vector<int64_t> &ne,
        bool required = true);
    /**
     * 
    */
    ggml_tensor *createTensorAsView(struct ggml_context *ctx,
        ggml_tensor *base,
        const std::string &name,
        const std::vector<int64_t> &ne,
        size_t offset,
        bool required = true);

    /**
     * If all tensors in this model have been created
    */
   bool areAllTensorsCreated() const;

  protected:
    size_t mNumKeyValues;
    size_t mNumTensors;
    size_t mNumElements;
    size_t mNumBytes;

    //bool mUseMmap = false;
    std::vector<std::string> mFiles;
    //llama_mmaps mappings;
    //std::unordered_map<std::string, struct llama_model_kv_override> kv_overrides;
    
    std::vector<Weight>           mWeights;
    gguf_context*                 mMeta;
    std::vector<ggml_context*>    mContexts;
    
    std::string mArchName; //! The model arch

    uint32_t mNumCreated = 0; //! The number of created tensors
};


extern Model loadModel(const char* file) noexcept;

M_END_NAMESPACE

#endif // !M_MODEL_LOADER_H

