/**
 * The model
 * Author: lihw81@gmail.com
*/ 

#include <common/m_vocab.h>

#include <common/m_misc.h>

#include <ggml/ggml-alloc.h>

#ifndef M_MODEL_H
#define M_MODEL_H

M_BEGIN_NAMESPACE

class Model
{
    M_NO_COPY_CONSTRUCTOR(Model)
    M_NO_MOVE_CONSTRUCTOR(Model)

public:
    enum class Type {
        MODEL_UNKNOWN,
        MODEL_17M,
        MODEL_22M,
        MODEL_33M,
        MODEL_109M,
        MODEL_137M,
        MODEL_335M,
        MODEL_0_5B,
        MODEL_1B,
        MODEL_2B,
        MODEL_3B,
        MODEL_4B,
        MODEL_7B,
        MODEL_8B,
        MODEL_12B,
        MODEL_13B,
        MODEL_14B,
        MODEL_15B,
        MODEL_20B,
        MODEL_30B,
        MODEL_34B,
        MODEL_35B,
        MODEL_40B,
        MODEL_65B,
        MODEL_70B,
        MODEL_314B,
        MODEL_SMALL,
        MODEL_MEDIUM,
        MODEL_LARGE,
        MODEL_XL,
        MODEL_A2_7B,
        MODEL_8x7B,
        MODEL_8x22B,
        MODEL_16x12B,
    };

    struct Parameters
    {
        bool vocabOnly;
        uint32_t vocabSize;//! The size of vocabulary
        // bool rope_finetuned;

        uint32_t contextLength;// context length the model was trained on
        uint32_t embedingLength;
        uint32_t attentionHeadCount;
        uint32_t attentionHeadCountKv;
        uint32_t layerCount;//! The layer count
        // uint32_t n_rot;
        uint32_t
            attentionKeyLength;// dimension of keys (d_k). d_q is assumed to be the same, but there are n_head q heads,
        //                        // and only n_head_kv k-v heads
        uint32_t attentionValueLength;// dimension of values (d_v) aka n_embd_head
        uint32_t feedForwardLength;
        uint32_t expertCount = 0;
        uint32_t expertUsedCount = 0;
        uint32_t vocabTypeCount = 0;// for BERT-style token types

        // float f_norm_eps;
        // float f_norm_rms_eps;

        // float rope_freq_base_train;
        // float rope_freq_scale_train;
        // uint32_t n_yarn_orig_ctx;

        //// for State Space Models
        // uint32_t ssm_d_conv = 0;
        // uint32_t ssm_d_inner = 0;
        // uint32_t ssm_d_state = 0;
        // uint32_t ssm_dt_rank = 0;

        // float f_clamp_kqv = 0.0f;
        // float f_max_alibi_bias = 0.0f;
        // float f_logit_scale = 0.0f;

        // bool causal_attn = true;
        // bool need_kq_pos = false;

        // enum llama_pooling_type pooling_type = LLAMA_POOLING_TYPE_NONE;
        // enum llama_rope_type rope_type = LLAMA_ROPE_TYPE_NONE;
        // enum llama_rope_scaling_type rope_scaling_type_train = LLAMA_ROPE_SCALING_TYPE_NONE;
    } params = {};

public:
    explicit Model();

private:
    Type type = Type::MODEL_UNKNOWN;
    Arch arch = Arch::UNKNOWN;
    GgufType ftype = GgufType::ALL_F32;

    std::string name = "n/a";

    Vocab vocab;


    // llama_split_mode split_mode;
    // int main_gpu;
    // int n_gpu_layers;

    // gguf metadata
    std::unordered_map<std::string, std::string> ggufKv;

    // layer -> buffer type mapping
    struct LayerBufferType
    {
        LayerBufferType() : bufferTypeMatrix(nullptr), bufferType(nullptr) {}
        LayerBufferType(ggml_backend_buffer_type_t matrix) : bufferTypeMatrix(matrix), bufferType(matrix) {}
        LayerBufferType(ggml_backend_buffer_type_t matrix, ggml_backend_buffer_type_t other)
            : bufferTypeMatrix(matrix), bufferType(other)
        {}

        ggml_backend_buffer_type_t bufferTypeMatrix;// matrices only - used by split buffers and backends that support
                                                    // only matrix multiplication
        ggml_backend_buffer_type_t bufferType;// everything else
    };

    LayerBufferType layerBufferTypeInput;
    LayerBufferType layerBufferTypeOutput;
    std::vector<LayerBufferType> layerBufferTypes;


    //  // model memory mapped files
    //  llama_mmaps mappings;

    //  ~llama_model() {
    //      for (struct ggml_context * ctx : ctxs) {
    //          ggml_free(ctx);
    //      }
    //      for (ggml_backend_buffer_t buf : bufs) {
    // #ifdef GGML_USE_CUDA
    //             if (ggml_backend_buffer_get_type(buf) == ggml_backend_cpu_buffer_type()) {
    //                 ggml_backend_cuda_unregister_host_buffer(ggml_backend_buffer_get_base(buf));
    //             }
    // #endif
    //             ggml_backend_buffer_free(buf);
    //         }
    //     }
public:
    bool loadParameters(ModelLoader &ml);

    bool loadTensors(ModelLoader &ml, int mainGpu, int32_t numGpuLayers, bool useMemoryLock);

    bool loadVocab(ModelLoader& ml);


private:
    // contexts where the model tensors metadata is stored
    std::vector<ggml_context *> mContexts;

    //! the model memory buffers for the tensor data
    std::vector<ggml_backend_buffer_t> mBuffers;

    // objects representing data potentially being locked in memory
    MemoryLocks mMemoryLocks;
    //  llama_mlocks mlock_mmaps;

    // for quantize-stats only
    std::vector<std::pair<std::string, struct ggml_tensor *>> mTensorsByName;
    
    struct {
        ggml_tensor *token_embed;//! ???
        //ggml_tensor *typeEmbd;//! ???
        //ggml_tensor *posEmbd;//! ???
        //ggml_tensor *tokNorm;//! ???
        //ggml_tensor *tokNormB;//! ???

        ggml_tensor *output_norm;
        //ggml_tensor *output_norm_b;
        //ggml_tensor *output;
        //ggml_tensor *output_b;

    } mTensors;
    
    struct Layer {
        // normalization
        struct ggml_tensor *attn_norm;
        struct ggml_tensor *attn_norm_b;
        struct ggml_tensor *attn_norm_2;
        struct ggml_tensor *attn_norm_2_b;
        struct ggml_tensor *attn_q_norm;
        struct ggml_tensor *attn_q_norm_b;
        struct ggml_tensor *attn_k_norm;
        struct ggml_tensor *attn_k_norm_b;
        struct ggml_tensor *attn_out_norm;
        struct ggml_tensor *attn_out_norm_b;

        // attention
        struct ggml_tensor *wq;
        struct ggml_tensor *wk;
        struct ggml_tensor *wv;
        struct ggml_tensor *wo;
        struct ggml_tensor *wqkv;

        // attention bias
        struct ggml_tensor *bq;
        struct ggml_tensor *bk;
        struct ggml_tensor *bv;
        struct ggml_tensor *bo;
        struct ggml_tensor *bqkv;

        // normalization
        struct ggml_tensor *ffn_norm;
        struct ggml_tensor *ffn_norm_b;
        struct ggml_tensor *layer_out_norm;
        struct ggml_tensor *layer_out_norm_b;

        // ff
        struct ggml_tensor *ffn_gate;// w1
        struct ggml_tensor *ffn_down;// w2
        struct ggml_tensor *ffn_up;// w3

        // ff MoE
        struct ggml_tensor *ffn_gate_inp;
        struct ggml_tensor *ffn_gate_exps;
        struct ggml_tensor *ffn_down_exps;
        struct ggml_tensor *ffn_up_exps;

        // ff shared expert (shexp)
        struct ggml_tensor *ffn_gate_inp_shexp;
        struct ggml_tensor *ffn_gate_shexp;
        struct ggml_tensor *ffn_down_shexp;
        struct ggml_tensor *ffn_up_shexp;

        // ff bias
        struct ggml_tensor *ffn_down_b;// b2
        struct ggml_tensor *ffn_up_b;// b3
        struct ggml_tensor *ffn_act;

        // mamba proj
        struct ggml_tensor *ssm_in;
        struct ggml_tensor *ssm_x;
        struct ggml_tensor *ssm_dt;
        struct ggml_tensor *ssm_out;

        // mamba
        struct ggml_tensor *ssm_conv1d;
        struct ggml_tensor *ssm_a;
        struct ggml_tensor *ssm_d;

        // mamba bias
        struct ggml_tensor *ssm_conv1d_b;
        struct ggml_tensor *ssm_dt_b;
    };
    std::vector<Layer> mLayers;

    int64_t mLoadUs = 0;//! when to load the model
    int64_t mStartUs = 0;//! when to load the tensors

};

M_END_NAMESPACE

#endif // !M_MODEL_H