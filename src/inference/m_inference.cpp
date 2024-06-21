/** 
* The LLM inference process
*
* Author: lihw81@gmail.com
*/

#include "m_inference.h"

#include <ggml/ggml.h>
#include <spdlog/spdlog.h>

#include <algorithm>

#include <cassert>

M_BEGIN_NAMESPACE

#undef min
#undef max

int32_t Inference::decode(Session& session, Batch& batch)
{
    const auto LOG_HEAD = "Inference::decode()";

    const uint32_t numTokens = (uint32_t)(batch.tokens.size());
    
    if (batch.tokens.empty() && batch.embeds.empty()) {
        spdlog::error("{}: tokens == 0", LOG_HEAD);
        return -1;
    }

    const auto& hparams = mModel.hparams;
    const auto& cparams = lctx.cparams;

    assert((!batch.tokens.empty() && batch.embeds.empty()) || (batch.tokens.empty() && !batch.embeds.empty())); // NOLINT

    // TODO: is it because there is only one token per batch?
    assert(numTokens <= cparams.n_batch);

    assert((cparams.causal_attn || cparams.n_ubatch >= numTokens) && "non-causal attention requires n_ubatch >= n_tokens");

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

M_END_NAMESPACE
