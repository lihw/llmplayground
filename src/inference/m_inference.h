/** 
* The LLM inference process
*
* Author: lihw81@gmail.com
*/

#ifndef M_INFERENCE_H
#define M_INFERENCE_H

#include <common/m_defs.h>
#include "m_batch.h"

M_BEGIN_NAMESPACE

class Inference {
    M_REMOVE_COPY_CONSTRUCTOR(Inference);
    M_REMOVE_MOVE_CONSTRUCTOR(Inference);

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
    void decode(Batch& batch);

private:
    int32_t mNumQueuedTokens = 0;
};


M_END_NAMESPACE

#endif // !M_INFERENCE_H
