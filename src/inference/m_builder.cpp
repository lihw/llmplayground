
#include "m_inference.h"

#include "m_batch.h"

#include <ggml/ggml-backend.h>

#include <spdlog/spdlog.h>

#include <cassert>

M_BEGIN_NAMESPACE

namespace infer {

Context::Builder::Builder() {
    //n_kv = (worst_case ? kv_self.size : kv_self.n);

    n_kv(worst_case ? kv_self.size : kv_self.n),
        n_outputs(worst_case ? n_tokens : lctx.n_outputs),
}

ggml_tensor* Context::Builder::buildInputEmbed(
        Context* context,
        const Model::Parameters& hparams,
        const Batch& batch,
        ggml_tensor * tokenEmbed,
        const BuildCallback& cb) {

    struct ggml_tensor * inputLayer;

    if (!batch.tokens.empty()) {
        context->mInputs.tokens = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, batch.numTokens);
        cb(context->mInputs.tokens, "inp_tokens", -1);
        ggml_set_input(context->mInputs.tokens);

        inputLayer = ggml_get_rows(ctx, tokenEmbed, context->mInputs.tokens);
    } else {
        context->mInputs.embeds = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hparams.embedingLength, batch.numTokens);
        inputLayer = context->mInputs.embeds;
        ggml_set_input(inputLayer);
    }

    cb(inputLayer, "inp_embd", -1);

    return inputLayer;
}

ggml_tensor* Context::Builder::buildNorm(
        ggml_tensor* cur,
        const Model::Parameters& hparams,
        struct ggml_tensor * mw,
        struct ggml_tensor * mb,
        NormType type,
        const BuildCallback& cb,
        int il) {
    switch (type) {
        case NormType::NORM:     
            assert(!"LLama uses RMS normalization");
            cur = ggml_norm(ctx, cur, hparams.normEps);
            break;
        case NormType::NORM_RMS: 
            cur = ggml_rms_norm(ctx, cur, hparams.normRmsEps); 
            break;
    }

    if (mw != nullptr || mb != nullptr) {
        cb(cur, "norm", il);
    }

    if (mw) {
        cur = ggml_mul(ctx, cur, mw);
        if (mb) {
            cb(cur, "norm_w", il);
        }
    }

    if (mb) {
        cur = ggml_add(ctx, cur, mb);
    }

    return cur;
}

ggml_tensor * Context::Builder::buildInputPosition(Context* context, const Batch& batch, const BuildCallback& cb) {
    context->mInputs.positions = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, batch.numTokens);
    cb(context->mInputs.positions, "inp_pos", -1);
    ggml_set_input(context->mInputs.positions);
    return context->mInputs.positions;
}


ggml_tensor* Context::Builder::buildInputKqMask(Context* context, const Batch& batch, const BuildCallback& cb, bool causal) {

    if (causal) {
        context->mInputs.kqMask = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, numKvEntries, batch.numTokens);
    } else {
        context->mInputs.kqMask = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, batch.numTokens, batch.numTokens);
    }
    cb(context->mInputs.kqMask, "KQ_mask", -1);
    ggml_set_input(context->mInputs.kqMask);
    return context->mInputs.kqMask;
}

ggml_tensor* Context::Builder::buildKv(const Model* model,
        const Context::KvCache& kv, ggml_cgraph* graph, ggml_tensor* wo,
        ggml_tensor* woB, ggml_tensor* k, ggml_tensor* v, ggml_tensor* q,
        ggml_tensor* kqMask, ggml_tensor* kqPos, int64_t contextLength, 
        int32_t numTokens, int32_t kv_head, int32_t n_kv, float kqScale, const BuildCallback& cb,
        int il) {

    // these nodes are added to the graph together so that they are not reordered
    // by doing so, the number of splits in the graph is reduced
    ggml_build_forward_expand(graph, q);
    ggml_build_forward_expand(graph, k);
    ggml_build_forward_expand(graph, v);

    const auto& hparams = model->params;

    {
        const int64_t embedingKGqa   = hparams.attentionKeyLength * hparams.attentionHeadCountKv;
        const int64_t embedingVGqa   = hparams.attentionValueLength * hparams.attentionHeadCountKv;

        assert(kv.size == hparams.contextLength);

        // compute the transposed [n_tokens, n_embd] V matrix
        assert(v->ne[0] == embedingVGqa && v->ne[1] == numTokens);

        ggml_tensor* v_t = ggml_transpose(ctx, v);
        cb(v_t, "v_cur_t", il);

        //??
        auto kl = kv.kLayers[il];
        ggml_tensor* kCacheView = ggml_view_1d(ctx, kl, numTokens * embedingKGqa,
                ggml_row_size(kl->type, embedingKGqa) * kv_head);
        cb(kCacheView, "k_cache_view", il);

        //??
        auto vl = kv.vLayers[il];
        ggml_tensor* vCacheView = ggml_view_2d(ctx, vl, numTokens, embedingVGqa,
                ggml_element_size(vl) * hparams.contextLength,
                ggml_element_size(vl) * kv_head);
        cb(vCacheView, "v_cache_view", il);

        // important: storing RoPE-ed version of K in the KV cache!
        ggml_build_forward_expand(graph, ggml_cpy(ctx, k, kCacheView));
        ggml_build_forward_expand(graph, ggml_cpy(ctx, v_t, vCacheView));
    }

    ggml_tensor * cur;

    {
        const int64_t n_head        = hparams.attentionHeadCount;
        const int64_t n_head_kv     = hparams.n_head_kv;
        const int64_t n_embd_head_k = hparams.n_embd_head_k;
        const int64_t embedingKGqa  = hparams.attentionKeyLength * hparams.attentionHeadCountKv;
        const int64_t n_embd_head_v = hparams.n_embd_head_v;

        struct ggml_tensor * qq = ggml_permute(ctx, q, 0, 2, 1, 3);
        cb(qq, "q", il);

        struct ggml_tensor * kk =
            ggml_view_3d(ctx, kvCache.getKAtLayer(il),
                    n_embd_head_k, n_kv, n_head_kv,
                    ggml_row_size(kv.k_l[il]->type, n_embd_k_gqa),
                    ggml_row_size(kv.k_l[il]->type, n_embd_head_k),
                    0);
        cb(kk, "k", il);

        struct ggml_tensor * kq = ggml_mul_mat(ctx, kk, qq);
        cb(kq, "kq", il);

        {
            kq = ggml_soft_max_ext(ctx, kq, kq_mask, kq_pos, kq_scale, hparams.f_max_alibi_bias);
            cb(kq, "kq_soft_max_ext", il);
        }

        assert(kv.size == hparams.contextLength);

        // split cached v into n_head heads
        ggml_tensor * vv =
            ggml_view_3d(ctx, kvCache.getVAtLayer(il),
                    n_kv, n_embd_head_v, n_head_kv,
                    ggml_element_size(kv.vLayers[il])*n_ctx,
                    ggml_element_size(kv.vLayers[il])*n_ctx*n_embd_head_v,
                    0);
        cb(vv, "v", il);

        struct ggml_tensor * kqv = ggml_mul_mat(ctx, v, kq);
        cb(kqv, "kqv", il);

        struct ggml_tensor * kqv_merged = ggml_permute(ctx, kqv, 0, 2, 1, 3);
        cb(kqv_merged, "kqv_merged", il);

        struct ggml_tensor * cur = ggml_cont_2d(ctx, kqv_merged, n_embd_head_k*n_head, n_tokens);
        cb(cur, "kqv_merged_cont", il);

        ggml_build_forward_expand(graph, cur);

        cur = ggml_mul_mat(ctx, wo, cur);
        if (woB) {
            cb(cur, "kqv_wo", il);
        }

        if (woB) {
            cur = ggml_add(ctx, cur, woB);
        }

        return cur;
    }
    cb(cur, "kqv_out", il);

    return cur;
}

ggml_tensor* Context::Builder::buildFFN(ggml_context* ctx,
        ggml_tensor* cur, ggml_tensor* upW, ggml_tensor* upB,
        ggml_tensor* gate, ggml_tensor* gate_b, ggml_tensor* downW, ggml_tensor* downB,
        ggml_tensor* act_scales, FfnOpType op, FfnGateType gateType, const BuildCallback& cb, int il) {
    ggml_tensor * tmp = ggml_mul_mat(ctx, upW, cur);
    cb(tmp, "ffn_up", il);

    if (upB) {
        tmp = ggml_add(ctx, tmp, upB);
        cb(tmp, "ffn_up_b", il);
    }

    if (gate) {
        switch (gateType) {
            case FfnGateType::SEQ:
                {
                    cur = ggml_mul_mat(ctx, gate, tmp);
                    cb(cur, "ffn_gate", il);
                } break;
            case FfnGateType::PAR:
                {
                    cur = ggml_mul_mat(ctx, gate, cur);
                    cb(cur, "ffn_gate", il);
                } break;
        }

        if (gate_b) {
            cur = ggml_add(ctx, cur, gate_b);
            cb(cur, "ffn_gate_b", il);
        }
    } else {
        cur = tmp;
    }

    switch (op) {
        case FfnOpType::SILU:
            {
                cur = ggml_silu(ctx, cur);
                cb(cur, "ffn_silu", il);
            } break;
        case FfnOpType::GELU:
            {
                cur = ggml_gelu(ctx, cur);
                cb(cur, "ffn_gelu", il);
                if (act_scales != NULL) {
                    cur = ggml_div(ctx, cur, act_scales);
                    cb(cur, "ffn_act", il);
                }
            } break;
        case FfnOpType::RELU:
            {
                cur = ggml_relu(ctx, cur);
                cb(cur, "ffn_relu", il);
            } break;
        case FfnOpType::RELU_SQR:
            {
                cur = ggml_relu(ctx, cur);
                cb(cur, "ffn_relu", il);

                cur = ggml_sqr(ctx, cur);
                cb(cur, "ffn_sqr(relu)", il);
            } break;
    }

    if (gateType == FfnGateType::PAR) {
        cur = ggml_mul(ctx, cur, tmp);
        cb(cur, "ffn_gate_par", il);
    }

    cur = ggml_mul_mat(ctx, downW, cur);
    cb(cur, "ffn_down", il);

    if (downB) {
        cur = ggml_add(ctx, cur, downB);
        cb(cur, "ffn_down_b", il);
    }

    return cur;
}
 
ggml_tensor* Context::Builder::buildInputOutIds(Context* context, const BuildCallback& cb, int32_t numOutputs) {
    context->mInputs.inputOutIds = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, numOutputs);
    cb(context->mInputs.inputOutIds, "inp_out_ids", -1);
    ggml_set_input(context->mInputs.inputOutIds);
    return context->mInputs.inputOutIds;
}

ggml_cgraph* Context::Builder::buildLlama(Context* context, const Batch& batch, const KvCache& kvCache,
        const Model::Parameters& hparams, bool worstCase, const BuildCallback& cb) {

    numKv = worstCase ? kvCache.size : kvCache.count;
    numOutputs = worstCase ? batch.numTokens : context->mNumOutputs;
    int32_t kvHead = worstCase ? (kvCache.recurrent ? 0 : kvCache.size - batch.numTokens) : kvCache.head;

    ggml_cgraph* gf = ggml_new_graph_custom(ctx, MAX_NODES, false);

    auto* model = context->mModel;

    // mutable variable, needed during the last layer of the computation to skip unused tokens
    int32_t numTokens = batch.numTokens;
    int32_t numHeads = hparams.attentionHeadCount;

    const int64_t attentionLength = hparams.attentionValueLength;
    assert(attentionLength == hparams.attentionKeyLength);
    //assert(attentionLength == mModel->params.n_rot);

    struct ggml_tensor* cur;
    struct ggml_tensor* inputEmbed;
    struct ggml_tensor* inputPosition;

    // B: batch size
    // T: token/embeding dim
    // C: the vocab size/number of tokens

    // inputEmbed: [B x T] F32
    // model->mTensors.token_embed: [C x T] F32, i.e, [32000 x 4096] in llama 7b
    inputEmbed = buildInputEmbed(context, hparams, batch, model->tensors.token_embed, cb);

    // inputPositions - contains the positions: [B x 1] I32
    inputPosition = buildInputPosition(context, batch, cb);

    // kqMask (mask for 1 head, it will be broadcasted to all heads)
    // kqMask: [B x KV_CACHE_SIZE] F32
    ggml_tensor* kqMask = buildInputKqMask(context, batch, cb, true);

    // layerInput: [B x T] F32, The input for next layer.
    auto* inputNextLayer = inputEmbed; 

    for (int il = 0; il < model->params.layerCount; ++il) {
        ggml_tensor* inputThisLayer = inputNextLayer;

        // -> cur: [B x T]
        //    model->mLayer.attn_norm: [1 x T] F32, i.e., [1 x 4096] in llama 7b
        // <- cur: [B x T]
        cur = buildNorm(inputEmbed, hparams, model->layers[il].attn_norm, nullptr, NormType::NORM_RMS, cb, il);
        cb(cur, "attn_norm", il);

        // self-attention
        {
            // compute Q and K and RoPE them

            // -> cur: [B x T] (ggml takes in the transposed input, A x B^T = C^T
            //    model->wq: [T x T], i.e, [4096 x 4096] in llama 7b
            // <- Qcur: [T x B] (ggml transposes the outputs)
            ggml_tensor* Qcur = ggml_mul_mat(ctx, model->layers[il].wq, cur);
            cb(Qcur, "Qcur", il);
            if (model->layers[il].bq) {
                Qcur = ggml_add(ctx, Qcur, model->layers[il].bq);
                cb(Qcur, "Qcur", il);
            }

            // -> cur: [B x T] (ggml takes in the transposed input, A x B^T = C^T
            //    model->wk: [T x T], i.e, [4096 x 4096] in llama 7b
            // <- Kcur: [T x B] (ggml transposes the outputs)
            ggml_tensor* Kcur = ggml_mul_mat(ctx, model->layers[il].wk, cur);
            cb(Kcur, "Kcur", il);
            if (model->layers[il].bk) {
                Kcur = ggml_add(ctx, Kcur, model->layers[il].bk);
                cb(Kcur, "Kcur", il);
            }

            // -> cur: [B x T] (ggml takes in the transposed input, A x B^T = C^T
            //    model->wk: [T x T], i.e, [4096 x 4096] in llama 7b
            // <- Vcur: [T x B] (ggml transposes the outputs)
            struct ggml_tensor* Vcur = ggml_mul_mat(ctx, model->layers[il].wv, cur);
            cb(Vcur, "Vcur", il);
            if (model->layers[il].bv) {
                Vcur = ggml_add(ctx, Vcur, model->layers[il].bv);
                cb(Vcur, "Vcur", il);
            }

            // Rotatry (relative) position encoding.
            // Check out the reference paper, https://openreview.net/forum?id=wHBfxhZu1u
            // YaRN: Efficient Context Window Extension of Large Language Models
            // ??
            // -> Qcur: [HT x H x B]
            //    inputPosition: [B x 1]
            // <- Qcur: [HT x H x B]
            Qcur = ggml_rope_custom(
                ctx, ggml_reshape_3d(ctx, Qcur, hparams.attentionHeadCountKv, numHeads, numTokens), inputPosition,
                hparams.rope.count, int(hparams.rope.scalingTypeTrain), 0, hparams.rope.yarnOrigCtxLength, hparams.rope.freqBaseTrain, hparams.rope.freqScaleTrain,
                extFactor, attnFactor, betaFast, betaSlow);
            cb(Qcur, "Qcur", il);

            // Ditto
            Kcur = ggml_rope_custom(
                ctx, ggml_reshape_3d(ctx, Kcur, hparams.attentionKeyLength, hparams.attentionHeadCountKv, numTokens), inputPosition,
                hparams.rope.count, int(hparams.rope.scalingTypeTrain), 0, hparams.rope.yarnOrigCtxLength, hparams.rope.freqBaseTrain, hparams.rope.freqScaleTrain,
                extFactor, attnFactor, betaFast, betaSlow
            );
            cb(Kcur, "Kcur", il);

            cur = buildKv(model, kvCache, gf,
                model->layers[il].wo, model->layers[il].bo,
                Kcur, Vcur, Qcur, kqMask, nullptr, hparams.contextLength, numTokens, kvHead, numKv, 1.0f / sqrtf(float(hparams.attentionValueLength)), cb, il);
        }

        // ??
        if (il == model->layers.size() - 1) {
            // skip computing output for unused tokens
            ggml_tensor* inputOutIds = buildInputOutIds(context, cb, numOutputs);
            numTokens = numOutputs;
            cur = ggml_get_rows(ctx, cur, inputOutIds);
            inputThisLayer = ggml_get_rows(ctx, inputThisLayer, inputOutIds);
        }

        // Residual forwarding
        // -> cur: [B x T]
        //    inpSA: [B x T]
        // <- cur: [B x T]
        ggml_tensor* ffnInp = ggml_add(ctx, cur, inputThisLayer);
        cb(ffnInp, "ffn_inp", il);

        // feed-forward network
        if (model->layers[il].ffn_gate_inp == nullptr) {
            cur = buildNorm(ffnInp, hparams, model->layers[il].ffn_norm, NULL,
                    NormType::NORM_RMS, cb, il);
            cb(cur, "ffn_norm", il);

            cur = buildFFN(ctx, cur,
                model->layers[il].ffn_up, nullptr,
                model->layers[il].ffn_gate, nullptr,
                model->layers[il].ffn_down, nullptr,
                nullptr, FfnOpType::SILU, FfnGateType::PAR, cb, il);
            cb(cur, "ffn_out", il);
        } else {
            // MoE branch
            cur = buildNorm(ffnInp, hparams,
                model->layers[il].ffn_norm, NULL,
                NormType::NORM_RMS, cb, il);
            cb(cur, "ffn_norm", il);

            ggml_tensor* logits = ggml_mul_mat(ctx, model->layers[il].ffn_gate_inp, cur); // [n_tokens, num_experts]
            cb(logits, "ffn_moe_logits", il);

            ggml_tensor* probs = ggml_soft_max(ctx, logits); // [n_tokens, num_experts]
            cb(probs, "ffn_moe_probs", il);

            // select experts
            ggml_tensor* selected_experts = ggml_top_k(ctx, probs, hparams.expertUsedCount); // [n_tokens, num_experts_per_tok]
            cb(selected_experts->src[0], "ffn_moe_argsort", il);

            ggml_tensor* weights = ggml_get_rows(ctx,
                ggml_reshape_3d(ctx, probs, 1, hparams.expertCount, numTokens), selected_experts);
            cb(weights, "ffn_moe_weights", il);

            weights = ggml_reshape_2d(ctx, weights, hparams.expertUsedCount, numTokens); // [n_tokens, num_experts_per_tok]

            ggml_tensor* weights_sum = ggml_sum_rows(ctx, weights);
            cb(weights_sum, "ffn_moe_weights_sum", il);

            weights = ggml_div(ctx, weights, weights_sum); // [n_tokens, num_experts_per_tok]
            cb(weights, "ffn_moe_weights_norm", il);

            // compute expert outputs
            ggml_tensor* moe_out = nullptr;

            for (int i = 0; i < hparams.expertUsedCount; ++i) {
                ggml_tensor* cur_expert;

                ggml_tensor* cur_up = ggml_mul_mat_id(ctx, model->layers[il].ffn_up_exps, selected_experts, i, cur);
                cb(cur_up, "ffn_moe_up", il);

                ggml_tensor* cur_gate = ggml_mul_mat_id(ctx, model->layers[il].ffn_gate_exps, selected_experts, i, cur);
                cb(cur_gate, "ffn_moe_gate", il);

                cur_gate = ggml_silu(ctx, cur_gate);
                cb(cur_gate, "ffn_moe_silu", il);

                cur_expert = ggml_mul(ctx, cur_up, cur_gate);
                cb(cur_expert, "ffn_moe_gate_par", il);

                cur_expert = ggml_mul_mat_id(ctx, model->layers[il].ffn_down_exps, selected_experts, i, cur_expert); // [n_tokens, n_embd]
                cb(cur_expert, "ffn_moe_down", il);

                cur_expert = ggml_mul(ctx, cur_expert,
                    ggml_view_2d(ctx, weights, 1, numTokens, weights->nb[1], i * weights->nb[0]));
                cb(cur_expert, "ffn_moe_weighted", il);

                if (i == 0) {
                    moe_out = cur_expert;
                }
                else {
                    moe_out = ggml_add(ctx, moe_out, cur_expert);
                    cb(moe_out, "ffn_moe_out", il);
                }
            }

            cur = moe_out;
        }

        // Another residual
        cur = ggml_add(ctx, cur, ffnInp);
        cb(cur, "ffn_out", il);

        // Apply a control vector to the output of this layer.
        ggml_tensor* layerDir = context->control.tensor_for(il);
        if (layerDir != nullptr) {
            cur = ggml_add(ctx, cur, layerDir);
        }
        cb(cur, "l_out", il);

        // input for next layer
        inputNextLayer = cur;
    }

    cur = inputNextLayer;

    cur = buildNorm(cur, hparams,
        model->tensors.output_norm, NULL,
        NormType::NORM_RMS, cb, -1);
    cb(cur, "result_norm", -1);

    // lm_head
    cur = ggml_mul_mat(ctx, model->tensors.output, cur);
    cb(cur, "result_output", -1);

    ggml_build_forward_expand(gf, cur);

    return gf;
}

ggml_cgraph* Context::_buildGraph(const Batch& batch, bool worstCase) {
    // this callback allows us to apply custom logic to each tensor (e.g. ggml-alloc, offloading, etc.)
    auto cb = [&](struct ggml_tensor* cur, const char* name, int il) {
        if (il >= 0) {
            ggml_format_name(cur, "%s-%d", name, il);
        } else {
            ggml_set_name(cur, name);
        }

        //if (!params.offload_kqv) {
        //    if (strncmp(name, "kqv_merged_cont", 12) == 0) {
        //        // all nodes between the KV store and the attention output are run on the CPU
        //        ggml_backend_sched_set_tensor_backend(mSched, cur, mBackendCpu);
        //    }
        //}

        // norm may be automatically assigned to the backend of the previous layer, increasing data transfer between backends
        // FIXME: fix in ggml_backend_sched
        if (batch.numTokens < 32) {
            if (il != -1 && strncmp(name, "norm", 4) == 0) {
                for (auto* backend : mBackends) {
                    if (ggml_backend_buft_supports_backend(mModel->getLayerBufferType(il).bufferType, backend)) {
                        ggml_backend_sched_set_tensor_backend(mSched, cur, backend);
                        break;
                    }
                }
            }
        }
    };

    struct ggml_cgraph* result = NULL;

    Builder builder;

    switch (mModel->getArch()) {
        case Arch::LLAMA:
        {
            result = builder.buildLlama(this, batch, mKvCache,
                mModel->params, worstCase, cb);
        } break;
        default:
            assert(false);
    }

    return result;
}

};

M_END_NAMESPACE

