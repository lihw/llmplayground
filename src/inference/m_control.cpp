
#include "m_inference.h"

#include "m_batch.h"

#include <spdlog/spdlog.h>

#include <cassert>

M_BEGIN_NAMESPACE

namespace infer {

#if 0
bool llama_control_vector_init(struct llama_control_vector & cvec, const llama_model & model) {
    GGML_ASSERT(cvec.tensors.empty());
    GGML_ASSERT(cvec.ctxs.empty());
    GGML_ASSERT(cvec.bufs.empty());

    // count layer buffer types
    std::map<ggml_backend_buffer_type_t, int> buft_layer_count;
    for (int64_t i = 0; i < model.hparams.n_layer; i++) {
        buft_layer_count[model.buft_layer[i].buft]++;
    }

    // allocate contexts
    std::map<ggml_backend_buffer_type_t, ggml_context *> ctx_map;
    for (auto & it : buft_layer_count) {
        int n_layers = it.second;
        struct ggml_init_params params = {
            /*.mem_size   =*/ n_layers * ggml_tensor_overhead(),
            /*.mem_buffer =*/ NULL,
            /*.no_alloc   =*/ true,
        };
        ggml_context * ctx = ggml_init(params);
        if (!ctx) {
            LLAMA_LOG_ERROR("%s: failed to allocate context for control vector\n", __func__);
            return 1;
        }
        ctx_map[it.first] = ctx;
    }

    // make tensors
    cvec.tensors.push_back(nullptr); // there's never a tensor for layer 0
    for (size_t il = 1; il < model.hparams.n_layer; il++) {
        struct ggml_context * ctx = ctx_map.at(model.buft_layer[il].buft);
        ggml_tensor * tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, model.hparams.n_embd);
        cvec.tensors.push_back(tensor);
    }

    // allocate tensors / buffers and zero
    for (auto it : ctx_map) {
        ggml_backend_buffer_type_t buft = it.first;
        ggml_context * ctx = it.second;
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
        if (!buf) {
            LLAMA_LOG_ERROR("%s: failed to allocate buffer for control vector\n", __func__);
            return false;
        }
        ggml_backend_buffer_clear(buf, 0);
        cvec.ctxs.push_back(ctx);
        cvec.bufs.push_back(buf);
    }

    return true;
}

int32_t llama_control_vector_apply(struct llama_context * lctx, const float * data, size_t len, 
        int32_t n_embd, int32_t il_start, int32_t il_end) {
    const llama_model & model = lctx->model;
    llama_control_vector & cvec = lctx->cvec;

    if (data == nullptr) {
        // disable the current control vector (but leave allocated for later)
        cvec.layer_start = -1;
        cvec.layer_end   = -1;
        return 0;
    }

    if (n_embd != (int) model.hparams.n_embd) {
        LLAMA_LOG_ERROR("%s: control vector n_embd does not match model\n", __func__);
        return 1;
    }

    if (cvec.tensors.empty()) {
        if (!llama_control_vector_init(cvec, model)) {
            return 1;
        }
    }

    cvec.layer_start = il_start;
    cvec.layer_end   = il_end;

    for (size_t il = 1; il < model.hparams.n_layer; il++) {
        assert(cvec.tensors[il] != nullptr);

        const size_t off = n_embd * (il - 1); // buffer doesn't have data for layer 0, since it's never present
        if (off + n_embd <= len) {
            ggml_backend_tensor_set(cvec.tensors[il], data + off, 0, n_embd * ggml_element_size(cvec.tensors[il]));
        }
    }

    return 0;
} 
#endif

};

M_END_NAMESPACE

