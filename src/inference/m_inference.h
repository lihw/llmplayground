/** 
* The LLM inference process
*
* Author: lihw81@gmail.com
*/

#ifndef M_INFERENCE_H
#define M_INFERENCE_H

#include <common/m_model.h>

M_BEGIN_NAMESPACE

class Batch;

namespace infer {

class Context {
    M_NO_COPY_CONSTRUCTOR(Context);
    M_NO_MOVE_CONSTRUCTOR(Context);

public:
    struct Parameters {
        size_t seed = 0xdeadbeef;
        size_t contextSize = 512;
        size_t batchSize = 2048;
        size_t unitBatchSize = 512; // FIXME: unit batch? original n_ubatch

        size_t numThreads = 4;
        size_t numThreadsBatch = 4;

        bool embeddings;
        //bool causal_attn;    // FIXME: what does it mean?
        //bool offload_kqv;

        size_t maxNumSeqs = 1;

        PoolingType poolingType = PoolingType::UNSPECIFIED;

        ggml_type typeK = GGML_TYPE_F16;
        ggml_type typeV = GGML_TYPE_F16;

#if 0
        
        /*.rope_scaling_type           =*/ LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED,
        
        /*.rope_freq_base              =*/ 0.0f,
        /*.rope_freq_scale             =*/ 0.0f,
        /*.yarn_ext_factor             =*/ -1.0f,
        /*.yarn_attn_factor            =*/ 1.0f,
        /*.yarn_beta_fast              =*/ 32.0f,
        /*.yarn_beta_slow              =*/ 1.0f,
        /*.yarn_orig_ctx               =*/ 0,
        /*.defrag_thold                =*/ -1.0f,
        /*.cb_eval                     =*/ nullptr,
        /*.cb_eval_user_data           =*/ nullptr,
        /*.logits_all                  =*/ false,
        /*.embeddings                  =*/ false,
        /*.offload_kqv                 =*/ true,
        /*.abort_callback              =*/ nullptr,
        /*.abort_callback_data         =*/ nullptr,
#endif
    } params;

    friend Context* createContext(Model* model, const infer::Context::Parameters& parameters);

    /**
     * Autogressively infer the next token given a batch of tokens.
     */
    int32_t decode(Batch* batch);

private:
    /**
     */
    explicit Context(Model* model);

    // Make sure enough space is available for outputs.
    // Returns max number of outputs for which space was reserved.
    size_t _reserveOutputs(uint32_t numOutputs);

    ggml_cgraph* _buildGraph(const Batch& batch, bool worstCase);


    using BuildCallback = std::function<void(struct ggml_tensor * cur, const char * name, int nl)>;

    struct KvCache;

    struct Builder {
        explicit Builder();

        ~Builder();

        ggml_cgraph* buildLlama(Context* context, const Batch& batch, const KvCache& kvCache,
                const Model::Parameters& hparams, bool worseCase, const BuildCallback& cb);

        ggml_tensor* buildInputEmbed(Context* context, const Model::Parameters& hparams, const Batch& batch,
                ggml_tensor* tokenEmbed, const BuildCallback& cb);

        ggml_tensor* buildNorm(ggml_tensor * cur, const Model::Parameters& hparams, 
                ggml_tensor* mw, ggml_tensor* mb, NormType type, const BuildCallback& cb, int il);

        ggml_tensor* buildInputPosition(Context* context, const Batch& batch, const BuildCallback& cb);

        ggml_tensor* buildInputKqMask(Context* context, const Batch& batch, const BuildCallback& cb, bool causal = true);

        ggml_tensor* buildKv(const Model* model, const KvCache& kv, ggml_cgraph* graph, ggml_tensor* wo, ggml_tensor* woB,
            ggml_tensor* k, ggml_tensor* v, ggml_tensor* q, ggml_tensor* kqMask, ggml_tensor* kqPosition, int64_t contextLength,
            int32_t numTokens, int32_t kv_head, int32_t n_kv, float kqScale, const BuildCallback& cb, int il);

        ggml_tensor* buildFFN(ggml_context* ctx, ggml_tensor* cur, ggml_tensor* upW, ggml_tensor* upB, ggml_tensor* gate, ggml_tensor* gate_b,
            ggml_tensor* downW, ggml_tensor* downB, ggml_tensor* act_scales, FfnOpType op, FfnGateType gateType, const BuildCallback& cb, int il);

        ggml_tensor* buildInputOutIds(Context* context, const BuildCallback& cb, int32_t numOutputs);

        //
        //
        //

        ggml_context* ctx;

        uint32_t numKv;
        uint32_t numOutputs;

        float extFactor;
        float attnFactor;
        float betaFast;
        float betaSlow;
    };

    struct KvCache {
        explicit KvCache(ggml_type kType, ggml_type vType, uint32_t kvSize);

        ~KvCache();

        std::vector<ggml_context*> mContexts;

        size_t head;
        size_t size;
        size_t count;
        bool   hasShift;
        bool   recurrent;
        bool   used;

        ggml_type kType;
        ggml_type vType;

        std::vector<ggml_tensor*> kLayers;
        std::vector<ggml_tensor*> vLayers;
    } mKvCache;

    struct Control {
        std::vector<ggml_tensor *> tensors; // per layer
        std::vector<ggml_context *> contexts;
        std::vector<ggml_backend_buffer_t> buffers;

        int32_t layerStart = -1;
        int32_t layerEnd   = -1;

        ggml_tensor * tensorFor(int il) const {
            if (il < 0 || il < layerStart || il > layerEnd || (size_t) il >= tensors.size()) {
                return nullptr;
            }
            return tensors[il];
        }

        ~Control() {
            for (struct ggml_context * ctx : contexts) {
                ggml_free(ctx);
            }
            for (ggml_backend_buffer_t buffer : buffers) {
                ggml_backend_buffer_free(buffer);
            }
        }
    };       
    
private:
    Model* mModel;

    std::vector<ggml_backend_t> mBackends; //! The backend
    ggml_backend_t mBackendCpu = nullptr; // The backend using CPU

    size_t mComputeStartUs; //! When the decoding computation starts

    size_t mNumQueuedTokens; //! The queued unprocessed tokens

    size_t mOutputSize = 0; //! capacity (of tokens positions) for the output buffers

    size_t mNumOutputs = 0; //! The current output tokens/logits

    ggml_backend_buffer_t mOutputBuffer; //! host buffer for the model output (logits and embeddings)

    bool mOutputAllLogits; //! Output logits of all tokens
    float* mLogits; //! decode output (2-dimensional array: [n_outputs][n_vocab])
    size_t mLogitSize; //! The capacity of logits in bytes

    std::vector<int32_t> mOutputIds; //! map batch token positions to ids of the logits and embd buffers

    /**
     * embeddings output (2-dimensional array: [n_outputs][n_embd])
     * populated only when pooling_type == LLAMA_POOLING_TYPE_NONE
     */
    float* mEmbeds;
    size_t mEmbedSize; //! The capacity of embeds in bytes

    std::vector<uint8_t> mBufferComputeMeta;
    ggml_backend_sched_t mSched = nullptr;

    struct {
        ggml_tensor* tokens;    // I32 [n_batch]
        ggml_tensor* embeds;      // F32 [n_embd, n_batch]
        ggml_tensor* positions; // I32 [n_batch]
        ggml_tensor* kqMask;   // F32 [kv_size, n_batch]
        ggml_tensor* inputOutIds; // I32 [n_outputs]
    } mInputs;
};

extern Context* createContext(Model* model, const infer::Context::Parameters& parameters);


}; // namespace infer

#if 0
public:
    struct CParams {
        uint32_t n_ctx;           // context size used during inference
        uint32_t n_batch;
        uint32_t n_ubatch; //???
        uint32_t n_seq_max;
        uint32_t n_threads;       // number of threads to use for generation
        uint32_t n_threads_batch; // number of threads to use for batch processing

        float rope_freq_base;
        float rope_freq_scale;

        uint32_t n_yarn_orig_ctx;
        // These hyperparameters are not exposed in GGUF, because all
        // existing YaRN models use the same values for them.
        float yarn_ext_factor;
        float yarn_attn_factor;
        float yarn_beta_fast;
        float yarn_beta_slow;
        float defrag_thold;

        bool embeddings;
        bool causal_attn; // ???
        bool offload_kqv;

        enum llama_pooling_type pooling_type;

        ggml_backend_sched_eval_callback cb_eval;
        void * cb_eval_user_data;
    };

public:
    int32_t decode(Session& session, Batch& batch);

private:
    int32_t mNumQueuedTokens = 0;
};

#endif

M_END_NAMESPACE

#endif // !M_INFERENCE_H
