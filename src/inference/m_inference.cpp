/** 
* The LLM inference process
*
* Author: lihw81@gmail.com
*/

#include "m_inference.h"

#include "m_batch.h"

#include <common/m_model.h>

#include <ggml/ggml.h>
#include <ggml/ggml-backend.h>
#include <ggml/ggml-alloc.h>
#include <spdlog/spdlog.h>

#include <algorithm>

#include <cassert>

M_BEGIN_NAMESPACE

#undef min
#undef max

namespace infer {

Context::Context(Model* model)
    : mModel(model)
{
    assert(mModel != nullptr);
}

Context* createContext(Model* model, const infer::Context::Parameters& parameters)
{
    const auto LOG_HEAD = "createContext()";

    if (!model) {
        spdlog::error("{}: model cannot be NULL", LOG_HEAD);
        return nullptr;
    }
    
    if (parameters.batchSize == 0 && parameters.unitBatchSize == 0) {
        spdlog::error("{}: batchSize and unitBatchSize cannot both be zero", LOG_HEAD);
        return nullptr;
    }

    if (parameters.contextSize == 0 && model->hparams.contextSize == 0) {
        spdlog::error("{}: contextSize and model->hparams.contextSize cannot both be zero", LOG_HEAD);
        return nullptr;
    }

    auto* ctx = new Context(model);
    
    //ret->params = parameters;
    // TODO: validate the parameters.


    const auto & hparams = model->params;
    auto       & cparams = ctx->params;

    cparams.maxNumSeqs         = std::max(1u, params.maxNumSeqs);
    cparams.numThreads         = parameters.numThreads;
    cparams.numThreadsBatch    = parameters.numThreadsBatch;
    //cparams.yarn_ext_factor  = parameters.yarn_ext_factor;
    //cparams.yarn_attn_factor = parameters.yarn_attn_factor;
    //cparams.yarn_beta_fast   = parameters.yarn_beta_fast;
    //cparams.yarn_beta_slow   = parameters.yarn_beta_slow;
    //cparams.defrag_thold     = parameters.defrag_thold;
    cparams.embeddings         = parameters.embeddings;
    //cparams.offload_kqv      = parameters.offload_kqv;
    cparams.poolingType        = parameters.poolingType;

    cparams.contextSize        = parameters.contextSize == 0    ? hparams.n_ctx_train           : params.contextSize;
    //cparams.rope_freq_base     = params.rope_freq_base  == 0.0f ? hparams.rope_freq_base_train  : params.rope_freq_base;
    //cparams.rope_freq_scale    = params.rope_freq_scale == 0.0f ? hparams.rope_freq_scale_train : params.rope_freq_scale;

    // this is necessary due to kv_self.n being padded later during inference
    cparams.contextSize = GGML_PAD(cparams.contextSize, 32);

    // with causal attention, the batch size is limited by the context size
    cparams.batchSize     = hparams.causal_attn ? std::min(cparams.n_ctx, params.n_batch) : params.n_batch;
    cparams.unitBatchSize = std::min(cparams.n_batch, params.n_ubatch == 0 ? params.n_batch : params.n_ubatch);


    //cparams.n_yarn_orig_ctx  = params.yarn_orig_ctx    != 0 ? params.yarn_orig_ctx    :
    //                           hparams.n_yarn_orig_ctx != 0 ? hparams.n_yarn_orig_ctx :
    //                                                          hparams.n_ctx_train;

    //cparams.cb_eval           = params.cb_eval;
    //cparams.cb_eval_user_data = params.cb_eval_user_data;

    //auto rope_scaling_type = params.rope_scaling_type;
    //if (rope_scaling_type == LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED) {
    //    rope_scaling_type = hparams.rope_scaling_type_train;
    //}

    //if (rope_scaling_type == LLAMA_ROPE_SCALING_TYPE_NONE) {
    //    cparams.rope_freq_scale = 1.0f; // never scale if scaling type is none
    //}

    //if (cparams.yarn_ext_factor < 0.0f) { // negative indicates 'not set'
    //    cparams.yarn_ext_factor = rope_scaling_type == LLAMA_ROPE_SCALING_TYPE_YARN ? 1.0f : 0.0f;
    //}

    cparams.causal_attn = hparams.causal_attn;

    if (cparams.poolingType == PoolingType::UNSPECIFIED) {
        if (hparams.poolingType == PoolingType::UNSPECIFIED) {
            cparams.poolingType = PoolingType::NONE;
        } else {
            cparams.poolingType = hparams.poolingType;
        }
    }

    if (params.seed == LLAMA_DEFAULT_SEED) {
        params.seed = time(NULL);
    }

    LLAMA_LOG_INFO("%s: n_ctx      = %u\n",     __func__, cparams.n_ctx);
    LLAMA_LOG_INFO("%s: n_batch    = %u\n",     __func__, cparams.n_batch);
    LLAMA_LOG_INFO("%s: n_ubatch   = %u\n",     __func__, cparams.n_ubatch);
    LLAMA_LOG_INFO("%s: freq_base  = %.1f\n",   __func__, cparams.rope_freq_base);
    LLAMA_LOG_INFO("%s: freq_scale = %g\n",     __func__, cparams.rope_freq_scale);

    //ctx->abort_callback      = params.abort_callback;
    //ctx->abort_callback_data = params.abort_callback_data;

    ctx->rng                 = std::mt19937(params.seed);
    ctx->logits_all          = params.logits_all;

    uint32_t kv_size = cparams.n_ctx;
    ggml_type type_k = params.type_k;
    ggml_type type_v = params.type_v;

    // Mamba only needs a constant number of KV cache cells per sequence
    if (model->arch == LLM_ARCH_MAMBA) {
        // Mamba needs at least as many KV cells as there are sequences kept at any time
        kv_size = std::max((uint32_t) 1, params.n_seq_max);
        // it's probably best to keep as much precision as possible for the states
        type_k = GGML_TYPE_F32; // required by ggml_ssm_conv for Mamba's conv_states
        type_v = GGML_TYPE_F32; // required by ggml_ssm_scan for Mamba's ssm_states
    }

    assert(hparams.n_embd_head_k % ggml_blck_size(type_k) == 0);
    assert(hparams.n_embd_head_v % ggml_blck_size(type_v) == 0);

    if (!hparams.vocabOnly) {
        // initialize backends
        ctx->backend_cpu = ggml_backend_cpu_init();
        if (ctx->backend_cpu == nullptr) {
            LLAMA_LOG_ERROR("%s: failed to initialize CPU backend\n", __func__);
            llama_free(ctx);
            return nullptr;
        }
        ctx->backends.push_back(ctx->backend_cpu);

        if (!llama_kv_cache_init(ctx->kv_self, ctx->model, type_k, type_v, kv_size, cparams.offload_kqv)) {
            LLAMA_LOG_ERROR("%s: llama_kv_cache_init() failed for self-attention cache\n", __func__);
            llama_free(ctx);
            return nullptr;
        }

        {
            size_t memory_size_k = 0;
            size_t memory_size_v = 0;

            for (auto & k : ctx->kv_self.k_l) {
                memory_size_k += ggml_nbytes(k);
            }

            for (auto & v : ctx->kv_self.v_l) {
                memory_size_v += ggml_nbytes(v);
            }

            LLAMA_LOG_INFO("%s: KV self size  = %7.2f MiB, K (%s): %7.2f MiB, V (%s): %7.2f MiB\n", __func__,
                    (float)(memory_size_k + memory_size_v) / (1024.0f * 1024.0f),
                    ggml_type_name(type_k), (float)memory_size_k / (1024.0f * 1024.0f),
                    ggml_type_name(type_v), (float)memory_size_v / (1024.0f * 1024.0f));
        }

        // graph outputs buffer
        {
            // resized during inference when a batch uses more outputs
            if (llama_output_reserve(*ctx, params.n_seq_max) < params.n_seq_max) {
                LLAMA_LOG_ERROR("%s: failed to reserve initial output buffer\n", __func__);
                llama_free(ctx);
                return nullptr;
            }

            LLAMA_LOG_INFO("%s: %10s  output buffer size = %8.2f MiB\n", __func__,
                    ggml_backend_buffer_name(ctx->buf_output),
                    ggml_backend_buffer_get_size(ctx->buf_output) / 1024.0 / 1024.0);
        }

        // scheduler and compute buffers
        {
            // buffer types used for the compute buffer of each backend
            std::vector<ggml_backend_buffer_type_t> backend_buft;
            for (auto * backend : ctx->backends) {
                if (ggml_backend_is_cpu(backend)) {
                    // use host buffers for the CPU backend compute buffer
                    backend_buft.push_back(llama_default_buffer_type_cpu(true));
                } else {
                    backend_buft.push_back(ggml_backend_get_default_buffer_type(backend));
                }
            }

            // buffer used to store the computation graph and the tensor meta data
            ctx->buf_compute_meta.resize(ggml_tensor_overhead()*LLAMA_MAX_NODES + ggml_graph_overhead_custom(LLAMA_MAX_NODES, false));

            // enabling pipeline parallelism in the scheduler increases memory usage, so it is only done when necessary
            bool pipeline_parallel = llama_get_device_count() > 1 && model->n_gpu_layers > (int)model->hparams.n_layer && model->split_mode == LLAMA_SPLIT_MODE_LAYER;
        
            ctx->sched = ggml_backend_sched_new(ctx->backends.data(), backend_buft.data(), ctx->backends.size(), LLAMA_MAX_NODES, pipeline_parallel);

            if (pipeline_parallel) {
                LLAMA_LOG_INFO("%s: pipeline parallelism enabled (n_copies=%d)\n", __func__, ggml_backend_sched_get_n_copies(ctx->sched));
            }

            // build worst-case graph
            int n_tokens = (int)std::min(cparams.n_ctx, cparams.n_ubatch);
            int n_past = cparams.n_ctx - n_tokens;
            llama_token token = llama_token_bos(&ctx->model); // not actually used by llama_build_graph, but required to choose between token and embedding inputs graph
            ggml_cgraph * gf = llama_build_graph(*ctx, llama_batch_get_one(&token, n_tokens, n_past, 0), true);

            // initialize scheduler with the worst-case graph
            if (!ggml_backend_sched_reserve(ctx->sched, gf)) {
                LLAMA_LOG_ERROR("%s: failed to allocate compute buffers\n", __func__);
                llama_free(ctx);
                return nullptr;
            }

            for (size_t i = 0; i < ctx->backends.size(); i++) {
                ggml_backend_t backend = ctx->backends[i];
                ggml_backend_buffer_type_t buft = backend_buft[i];
                size_t size = ggml_backend_sched_get_buffer_size(ctx->sched, backend);
                if (size > 1) {
                    LLAMA_LOG_INFO("%s: %10s compute buffer size = %8.2f MiB\n", __func__,
                            ggml_backend_buft_name(buft),
                            size / 1024.0 / 1024.0);
                }
            }

            // note: the number of splits during measure is higher than during inference due to the kv shift
            int n_splits = ggml_backend_sched_get_n_splits(ctx->sched);
            LLAMA_LOG_INFO("%s: graph nodes  = %d\n", __func__, gf->n_nodes);
            LLAMA_LOG_INFO("%s: graph splits = %d\n", __func__, n_splits);
        }
    }


    return ret;
}

int32_t Context::decode(Batch* batch)
{
    const auto LOG_HEAD = "Inference::decode()";

    const uint32_t numTokens = (uint32_t)(batch->tokens.size());
    
    if (batch->tokens.empty() && batch->embeds.empty()) {
        spdlog::error("{}: tokens == 0", LOG_HEAD);
        return -1;
    }
    
    const auto& hparams = mModel->params;
    const auto& cparams = params;

    assert((!batch->tokens.empty() && batch->embeds.empty()) || 
            (batch->tokens.empty() && !batch->embeds.empty())); // NOLINT

    // TODO: is it because there is only one token per batch?
    assert(numTokens <= cparams.batchSize);

    //assert((cparams.causal_attn || cparams.batchSize >= numTokens) && "non-causal attention requires batch size >= token count");
    //
    if (mComputeStartUs == 0) {
        mComputeStartUs = ggml_time_us();
    }
    mNumQueuedTokens += numTokens;
    
    // count outputs
    uint32_t numOutputs = 0;
    if (!batch->logits.empty()) {
        // The outputs are marked by the batch's logit array.
        for (uint32_t i = 0; i < numTokens; ++i) {
            numOutputs += batch->logits[i] != 0;
        }
    } else if (mOutputAllLogits || (cparams.embeddings && cparams.poolingType != PoolingType::NONE)) {
        // One output per each token.
        numOutputs = numTokens;
    } else {
        // keep last output only
        numOutputs = 1;
    }
    
    const int64_t embedLength = hparams.embedingLength;
    const int64_t vocabSize   = hparams.vocabSize;
    
    // reserve output buffer
    if (reserveOutputs(numOutputs) < numOutputs) {
        spdlog::error("{}: could not reserve space for batch with {} outputs", LOG_HEAD, numOutputs);
        return -2;
    }

    // set output mappings
    if (!batch->logits.empty()) {
        int32_t i_logits = 0;
        for (uint32_t i = 0; i < numTokens; ++i) {
            if (batch->logits[i]) {
                mOutputIds[i] = i_logits++;
            }
        }
    }
    else {
        for (uint32_t i = 0; i < numTokens; ++i) {
            mOutputIds[i] = i;
        }
    }

    // autogressive inference.
    for (uint32_t cur_token = 0; cur_token < numTokens; cur_token += n_ubatch) {
        const uint32_t n_tokens = std::min(n_ubatch, n_tokens_all - cur_token);
        Batch u_batch = {
            /* .n_tokens   = */ (int32_t)n_tokens,
            /* .token      = */ batch_all.token ? batch_all.token + cur_token : nullptr,
            /* .embd       = */ batch_all.embd ? batch_all.embd + cur_token * n_embd : nullptr,
            /* .pos        = */ batch_all.pos ? batch_all.pos + cur_token : nullptr,
            /* .n_seq_id   = */ batch_all.n_seq_id ? batch_all.n_seq_id + cur_token : nullptr,
            /* .seq_id     = */ batch_all.seq_id ? batch_all.seq_id + cur_token : nullptr,
            /* .logits     = */ batch_all.logits ? batch_all.logits + cur_token : nullptr,
            /* .all_pos_0  = */ batch_all.all_pos_0 + (llama_pos)cur_token * batch_all.all_pos_1,
            /* .all_pos_1  = */ batch_all.all_pos_1,
            /* .all_seq_id = */ batch_all.all_seq_id,
        };
    }


    return 0;
}

size_t Context::reserveOutputs(uint32_t numOutputs) 
{
    const auto LOG_HEAD = "Context::reserveOutputs()";

    const auto & cparams = params;
    const auto & hparams = mModel->params;

    const size_t maxNumOutputs = std::max((size_t)numOutputs, cparams.maxNumSeqs);

    const auto batchSize      = cparams.batchSize;
    const auto vocabSize      = hparams.vocabSize;
    const auto embedingLength = hparams.embedingLength;

    // TODO: use a per-batch flag for logits presence instead
    const bool hasLogits = hparams.causal_attn;
    const bool hasEmbeds = cparams.embeddings && (hparams.causal_attn || cparams.poolingType == PoolingType::NONE);

    const size_t logitSize = hasLogits ? vocabSize * maxNumOutputs : 0;
    const size_t embedSize = hasEmbeds ?  embedingLength * maxNumOutputs : 0;

    if (mOutputIds.empty()) {
        // init, never resized afterwards
        mOutputIds.resize(batchSize);
    }

    const size_t prevSize = mOutputBuffer ? ggml_backend_buffer_get_size(mOutputBuffer) : 0;
    const size_t newSize  = (logitSize + embedSize) * sizeof(float);

    // alloc only when more than the current capacity is required
    // TODO: also consider shrinking the buffer
    if (mOutputBuffer || prevSize < newSize) {
        if (mOutputBuffer) {
#ifndef NDEBUG
            // This doesn't happen often, but may be annoying in some cases (like the HellaSwag benchmark)
            spdlog::info("{}: reallocating output buffer from size {:.02f} MiB to {:.02f} MiB", LOG_HEAD, 
                    prevSize / 1024.0 / 1024.0, newSize / 1024.0 / 1024.0);
#endif
            ggml_backend_buffer_free(mOutputBuffer);
            mOutputBuffer = nullptr;
            mLogits = nullptr;
            mEmbeds = nullptr;
        }

        mOutputBuffer = ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(), newSize);
        if (mOutputBuffer == nullptr) {
            spdlog::error("{}: failed to allocate output buffer of size {:.2f} MiB\n", LOG_HEAD, newSize / (1024.0 * 1024.0));
            return 0;
        }
    }

    float* outputBase = (float *) ggml_backend_buffer_get_base(mOutputBuffer);

    mLogits = hasLogits? outputBase             : nullptr;
    mEmbeds = hasEmbeds? outputBase + logitSize : nullptr;

    mOutputSize = newSize; // FIXME: it is not the same as llama.cpp
    mLogitSize  = logitSize;
    mEmbedSize  = embedSize;

    // set all ids as invalid (negative)
    std::fill(mOutputIds.begin(), mOutputIds.end(), -1);

    ggml_backend_buffer_clear(mOutputBuffer, 0);

    mNumOutputs = 0;

    return maxNumOutputs;
}

}; // namespace infer

#if 0
int32_t Inference::decode(Session& session, Batch& batch)
{

    if (session.computeStartUs == 0) {
        session.computeStartUs = ggml_time_us();
    }
    session.numQueuedTokens += numTokens;

    auto& kv_self = session.kv_self;

    // count outputs
    uint32_t numOutputs = 0;
    if (!batch.logits.empty()) {
        // The outputs are marked by the batch's logit array.
        for (uint32_t i = 0; i < numTokens; ++i) {
            numOutputs += batch.logits[i] != 0;
        }
    } else if (session.allLogits || (cparams.embeddings && cparams.pooling_type != LLAMA_POOLING_TYPE_NONE)) {
        // One output per each token.
        numOutputs = numTokens;
    } else {
        // keep last output only
        numOutputs = 1;
    }
    
    const int64_t n_embd  = hparams.n_embd;
    const int64_t n_vocab = hparams.n_vocab;

    // reserve output buffer
    if (llama_output_reserve(lctx, numOutputs) < numOutputs) {
        spdlog::error("{}: could not reserve space for batch with {} outputs\n", LOG_HEAD, numOutputs);
        return -2;
    }

    // set output mappings
    if (!batch.logits.empty()) {
        int32_t logitsIndex = 0;
        for (uint32_t i = 0; i < numTokens; ++i) {
            if (batch.logits[i]) {
                session.outputIds[i] = logitsIndex;
            }
        }
    } else {
        for (uint32_t i = 0; i < numOutputs; ++i) {
            session.outputIds[i] = i;
        }
    }

    uint32_t numPrevOutputs = 0;
    const auto n_ubatch = cparams.n_ubatch;
    for (uint32_t curToken = 0; curToken < numTokens; curToken += n_ubatch) {
        const uint32_t curNumTokens = std::min(n_ubatch, numTokens - curToken);
        Batch curBatch;
        curBatch.tokens = std::vector<Token>(batch.tokens.begin() + curToken, batch.tokens.end());
        curBatch.embeds = std::vector<float>(batch.embeds.begin() + curToken, batch.embeds.end());
        curBatch.pos    = std::vector<Pos>(batch.pos.begin() + curToken, batch.pos.end());
        curBatch.seqIds = std::vector<std::vector<SeqId>>(batch.seqIds.begin() + curToken, batch.seqIds.end());
        curBatch.logits = std::vector<int8_t>(batch.logits.begin() + curToken, batch.logits.end());
            ///* .all_pos_0  = */ batch_all.all_pos_0 + (llama_pos) cur_token*batch_all.all_pos_1,
            ///* .all_pos_1  = */ batch_all.all_pos_1,
            ///* .all_seq_id = */ batch_all.all_seq_id,

        // count the outputs in this curBatch
        {
            int32_t numNewOutputs = 0;

            if (!curBatch.logits.empty()) {
                for (uint32_t i = 0; i < curNumTokens ; i++) {
                    numNewOutputs += curBatch.logits[i] != 0;
                }
            } else if (numOutputs == numTokens) {
                numNewOutputs = numTokens;
            } else {
                // keep last output only
                if (curToken + curNumTokens >= numTokens) {
                    numNewOutputs = 1;
                }
            }

            // needs to happen before the graph is built
            numOutputs = numNewOutputs;
        }

        int numThreads = curNumTokens == 1 ? cparams.n_threads : cparams.n_threads_batch;
        assert(numThreads > 0);

        // helpers for smoother batch API transition
        // after deprecating the llama_eval calls, these will be removed
        if (curBatch.pos.empty()) {
            curBatch.pos.resize(curNumTokens);
            for (uint32_t i = 0; auto &p : curBatch.pos) {
                curBatch.pos[i] = curBatch.all_pos_0 + i * curBatch.all_pos_1;
            }
        }

        if (curBatch.seqIds.empty()) {
            curBatch.seqIds.resize(curNumTokens);
            for (auto &s : curBatch.seqIds) {
                s.resize(1);
                s[0] = curBatch.all_seq_id;
            }
        }

        // non-causal masks do not use the KV cache
        if (hparams.causal_attn) {
            _updateKvCache();

            // if we have enough unused cells before the current head ->
            //   better to start searching from the beginning of the cache, hoping to fill it
            if (kv_self.head > kv_self.used + 2 * curNumTokens) {
                kv_self.head = 0;
            }

            if (!_findKvCacheSlot(kv_self, curBatch)) {
                return 1;
            }

            if (!kv_self.recurrent) {
                // a heuristic, to avoid attending the full cache if it is not yet utilized
                // after enough generations, the benefit from this heuristic disappears
                // if we start defragmenting the cache, the benefit from this will be more important
                kv_self.n = std::min(kv_self.size, std::max(32u, GGML_PAD(llama_kv_cache_cell_max(kv_self), 32)));
                //kv_self.n = llama_kv_cache_cell_max(kv_self);
            }
        }

        //printf("kv_self.n = %5d, kv_self.used = %5d, kv_self.head = %5d\n", kv_self.n, kv_self.used, kv_self.head);

        ggml_backend_sched_reset(session.sched);
        ggml_backend_sched_set_eval_callback(session.sched, lctx.cparams.cb_eval, lctx.cparams.cb_eval_user_data);

        ggml_cgraph * gf = llama_build_graph(session, curBatch, false);

        // the output is always the last tensor in the graph
        struct ggml_tensor * res  = gf->nodes[gf->n_nodes - 1];
        struct ggml_tensor * embd = gf->nodes[gf->n_nodes - 2];

        if (lctx.n_outputs == 0) {
            // no output
            res  = nullptr;
            embd = nullptr;
        } else if (!hparams.causal_attn) {
            res = nullptr; // do not extract logits for embedding models such as BERT

            // token or sequence embeddings
            embd = gf->nodes[gf->n_nodes - 1];

            assert(strcmp(embd->name, "result_embd") == 0 || strcmp(embd->name, "result_embd_pooled") == 0);
        } else if (cparams.embeddings) {
            // the embeddings could be in the second to last tensor, or any of the previous tensors
            int i_embd = gf->n_nodes - 2;
            for (int i = 3; strcmp(embd->name, "result_norm") != 0; ++i) {
                i_embd = gf->n_nodes - i;
                if (i_embd < 0) { break; }
                embd = gf->nodes[i_embd];
            }
            GGML_ASSERT(i_embd >= 0 && "missing result_norm tensor");

            // TODO: use a per-batch flag to know when to skip logits while keeping embeddings
            if (!cparams.causal_attn) {
                res = nullptr; // do not extract logits when not needed
                // skip computing logits
                // TODO: is this safe?
                gf->n_nodes = i_embd + 1;
            }
        } else {
            embd = nullptr; // do not extract embeddings when not needed
            GGML_ASSERT(strcmp(res->name, "result_output") == 0 && "missing result_output tensor");
        }
        // LLAMA_LOG_INFO("graph build time: %.3f ms (%d nodes, %d leafs)\n", (ggml_time_us() - t_start_us)/1000.0, gf->n_nodes, gf->n_leafs);

        // for big prompts, if BLAS is enabled, it is better to use only one thread
        // otherwise, the threads are spin-lock waiting for the BLAS calls and are degrading the performance
        // TODO: this is mostly important for Apple Silicon where CBLAS is still performing very well
        //       we still need some threads to process all non-mul_mat ops, but not too much to avoid interfering
        //       with the BLAS calls. need a better solution
        // MoE Special Case: This logic applies when hparams.n_expert == 0, i.e. the model is NOT an MoE model. When an MoE is
        //                   being processed then Accelerate/BLAS will not be involved, so capping would limit performance.
        if (curNumTokens >= 32 && hparams.n_expert == 0 && ggml_cpu_has_blas() && !ggml_cpu_has_gpublas()) {
            numThreads = std::min(4, numThreads);
        }

        ggml_backend_sched_alloc_graph(session.sched, gf);

        _setInputs(session, curBatch);

        _computeGraph(gf, numThreads);

        // update the kv ring buffer
        {
            kv_self.head += curNumTokens;

            // Ensure kv cache head points to a valid index.
            if (kv_self.head >= kv_self.size) {
                kv_self.head = 0;
            }
        }

#ifdef GGML_PERF
        // print timing information per ggml operation (for debugging purposes)
        // requires GGML_PERF to be defined
        ggml_graph_print(gf);
#endif

        // plot the computation graph in dot format (for debugging purposes)
        //if (n_past%100 == 0) {
        //    ggml_graph_dump_dot(gf, NULL, "llama.dot");
        //}

        // extract logits
        if (res) {
            ggml_backend_t backendRes = ggml_backend_sched_get_tensor_backend(session.sched, res);
            assert(backendRes != nullptr);
            assert(!session.logits.empty());

            float* outLogits = session.logits.data() + numPrevOutputs * hparams.n_vocab;
            const int32_t numOutputsNew = session.numOutputs;

            if (numOutputsNew) {
                assert(session.numPrevOutputs + numNewOutputs <= n_outputs);
                assert((session.numPrevOutputs + numNewOutputs) * hparams.n_vocab <= (int64_t)session.logitsSize);
                ggml_backend_tensor_get_async(backendRes, res, outLogits, 
                        0, numNewOutputs * hparams.n_vocab * sizeof(float));
            }
        }

        // extract embeddings
        if (embd) {
            ggml_backend_t backendEmbd = ggml_backend_sched_get_tensor_backend(session.sched, embd);
            assert(backendEmbd != nullptr);

            switch (cparams.pooling_type) {
                case LLAMA_POOLING_TYPE_NONE:
                    {
                        // extract token embeddings
                        assert(session.embd != nullptr);
                        float * embdOut = session.embd.data() + numPrevOutputs * hparams.n_embd;
                        const int32_t numNewOutputs = session.numOutputs;

                        if (numNewOutputs) {
                            assert( session.numPrevOutputs + numNewOutputs <= n_outputs);
                            assert((session.numPrevOutputs + numNewOutputs) * hparams.n_embd <= (int64_t) lctx.embd_size);
                            ggml_backend_tensor_get_async(backendEmbd, embd, embdOut, 0, numNewOutputs * hparams.n_embd * sizeof(float));
                        }
                    } break;
                case LLAMA_POOLING_TYPE_CLS:
                case LLAMA_POOLING_TYPE_MEAN:
                    {
                        GGML_ASSERT(strcmp(embd->name, "result_embd_pooled") == 0);

                        // extract sequence embeddings
                        auto & embdSeqOut = session.embdSeq;
                        embdSeqOut.clear();

                        for (uint32_t i = 0; i < curNumTokens; i++) {
                            const SeqId seqId = curBatch.seqIds[i][0];
                            if (embdSeqOut.find(seqId) != embdSeqOut.end()) {
                                continue;
                            }
                            embdSeqOut[seqId].resize(hparams.n_embd);
                            ggml_backend_tensor_get_async(backendEmbd, embd, embdSeqOut[seqId].data(), (hparams.n_embd * seqId) * sizeof(float), 
                                    hparams.n_embd * sizeof(float));
                        }
                    } break;
                case LLAMA_POOLING_TYPE_UNSPECIFIED:
                    {
                        assert(!"unknown pooling type");
                    } break;
            }
        }
        numPrevOutputs += session.numOutputs;
    }
    
    // set to total number of outputs in the batch, for use in llama_get_logits_ith
    session.numOutputs = numOutputs;

    // wait for the computation to finish (automatically done when obtaining the model output)
    //llama_synchronize(&lctx);

    // decide if we need to defrag the kv cache
    if (cparams.causal_attn && cparams.defrag_thold >= 0.0f) {
        const float fragmentation = kv_self.n >= 128 ? 1.0f - float(kv_self.used)/float(kv_self.n) : 0.0f;

        // queue defragmentation for next llama_kv_cache_update
        if (fragmentation > cparams.defrag_thold) {
            //LLAMA_LOG_INFO("fragmentation: %.2f\n", fragmentation);

            llama_kv_cache_defrag(kv_self);
        }
    }

    return 0;
}

#endif

M_END_NAMESPACE
