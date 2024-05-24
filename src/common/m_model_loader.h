/** 
* Load the GGUF models from files
*
* Author: lihw81@gmail.com
*/

#ifndef M_MODEL_LOADER
#define M_MODEL_LOADER

#include <common/m_defs.h>

#include <string>
#include <vector>

M_BEGIN_NAMESPACE

class ModelLoader final {
    M_NO_COPY_CONSTRUCTOR(ModelLoader)
    M_NO_MOVE_CONSTRUCTOR(ModelLoader)
public:
    explicit ModelLoader();

    ~ModelLoader();

    bool load(const std::string& file, bool useMmap) noexcept;

private:
    uint32_t mNumKeyValues;
    uint32_t mTensors;

    bool mUseMmap = false;
    
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

    std::vector<Weight> mWeights;
    
    struct gguf_context * meta = NULL;
    std::vector<ggml_context *> contexts;
    
    std::string mArchName; //! The model arch
    //LLM_KV      llm_kv    = LLM_KV(LLM_ARCH_UNKNOWN);
};

M_END_NAMESPACE

#endif // !M_MODEL_LOADER

