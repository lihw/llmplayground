
#include "m_inference.h"

#include <spdlog/spdlog.h>

#include <cassert>

M_BEGIN_NAMESPACE

namespace infer {

Context::KvCache::KvCache(Model* model, ggml_type kType, ggml_type vType, uint32_t kvSize)
{
    const auto& hparams = model->params;

    const uint32_t embedingKGqa = hparams.n_embd_k_gqa() + hparams.n_embd_k_s();
    const uint32_t embedingVGqa = hparams.n_embd_v_gqa() + hparams.n_embd_v_s();
    const int64_t  layerCount   = hparams.layerCount;

    mHasShift = false;

    // TODO: find a nicer way to add other recurrent model architectures
    mRecurrent = false;

    // TODO: support mixed reccurent Transformer architectues
    // NOTE: (!a || b) is a logical implication (a -> b)
    assert(!recurrent || embedingKGqa == hparams.n_embd_k_s());
    assert(!recurrent || embedingVGqa == hparams.n_embd_v_s());
    assert( recurrent || embedingKGqa == hparams.n_embd_k_gqa());
    assert( recurrent || embedingVGqa == hparams.n_embd_v_gqa());

    mHead = 0;
    mSize = kv_size;
    mUsed = 0;

    mTypeK = mTypeK;
    mTypeV = mTypeV;

    mCells.clear();
    mCells.resize(mSize);

    if (mRecurrent) {
        // init state copy sources
        for (uint32_t i = 0; i < mCache.size; ++i) {
            mCells[i].src = i;
        }
    }

#ifdef GGML_USE_CLBLAST
    offload = false;
#endif

    // count used buffer types
    std::map<ggml_backend_buffer_type_t, int> layerBufferTypeCount;
    if (offload) {
        for (int64_t i = 0; i < layerCount; ++i) {
            layerBufferTypeCount[model.layerBufferTypes[i].bufferType]++;
        }
    } else {
        layerBufferTypeCount[getDefaultBufferTypeCpu(true)] = layerCount;
    }

    // create a context for each buffer type
    std::map<ggml_backend_buffer_type_t, ggml_context *> ctxMap;
    for (auto & it : layerBufferTypeCount) {
        int numLayers = it.second;
        struct ggml_init_params params = {
            /*.mem_size   =*/ 2u * numLayers * ggml_tensor_overhead(),
            /*.mem_buffer =*/ NULL,
            /*.no_alloc   =*/ true,
        };
        ggml_context * ctx = ggml_init(params);
        if (!ctx) {
            spdlog::error("{}: failed to allocate context for kv cache", LOG_HEAD);
            return ;
        }
        ctxMap[it.first] = ctx;
        mContexts.push_back(ctx);
    }

    mKLayers.reserve(numLayers);
    mVLayers.reserve(numLayers);

    for (int i = 0; i < (int)layerCount; i++) {
        ggml_context * ctx = offload ? ctxMap.at(model->layerBufferTypes[i].bufferType) : mContexts.front();
        gVml_tensor* k = ggml_new_tensor_1d(ctx, mTypeK, embedingKGqa * mSize);
        ggml_tensor* v = ggml_new_tensor_1d(ctx, mTypeV, embedingVGqa * mSize);
        ggml_format_name(k, "cache_k_l%d", i);
        ggml_format_name(v, "cache_v_l%d", i);
        mKLayers.push_back(k);
        mVLayers.push_back(v);
    }

    // allocate tensors and initialize the buffers to avoid NaNs in the padding
    for (auto it : ctxMap) {
        ggml_backend_buffer_type_t bufferType = it.first;
        ggml_context* ctx = it.second;
        ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors_from_buft(ctx, bufferType);
        if (!buffer) {
            spdlog::error("{}: failed to allocate buffer for kv cache", LOG_HEAD);
            return ;
        }
        ggml_backend_buffer_clear(buffer, 0);
        spdlog::info("{}: {:10s} KV buffer size = {:8.2f} MiB\n", LOG_HEAD, ggml_backend_buffer_name(buffer),
                ggml_backend_buffer_get_size(buffer) / 1024.0 / 1024.0);
        mBuffers.push_back(buffer);
    }
}

void Context::KvCache::clear() {
    for (int32_t i = 0; i < (int32_t)mCells.size(); ++i) {
        mCells[i].pos = -1;
        mCells[i].seq_id.clear();
    }
    cache.mHead = 0;
    cache.mUsed = 0;
}

};

#if 0
// find an empty slot of size "n_tokens" in the cache
// updates the cache head
// Note: On success, it's important that cache.head points
// to the first cell of the slot.
static bool llama_kv_cache_find_slot(
           struct llama_kv_cache & cache,
        const struct llama_batch & batch) {
    const uint32_t n_ctx    = cache.size;
    const uint32_t n_tokens = batch.n_tokens;

    if (cache.recurrent) {
        // For recurrent state architectures (like Mamba),
        // each KV cache cell can store the state for a whole sequence.

        llama_seq_id min = cache.size - 1;
        llama_seq_id max = 0;

        for (uint32_t i = 0; i < n_tokens; ++i) {
            for (int32_t j = 0; j < batch.n_seq_id[i]; ++j) {
                llama_seq_id seq_id = batch.seq_id[i][j];
                // make sure it's a valid seq_id
                if ((uint32_t) seq_id < cache.size) {
                    if (seq_id > max) {
                        max = seq_id;
                    }
                    if (seq_id < min) {
                        min = seq_id;
                    }
                    // Assuming the tokens are in-order
                    if (batch.pos[i] != cache.cells[seq_id].pos + 1) {
                        // What should happen when the pos backtracks or skips a value?
                        // Clearing the state mid-batch would require special-casing which isn't done.
                        LLAMA_LOG_WARN("%s: non-consecutive token position %d after %d for sequence %d\n",
                            __func__, batch.pos[i], cache.cells[seq_id].pos, seq_id);
                    }
                    if (cache.cells[seq_id].pos < 0 && 0 <= batch.pos[i]) {
                        cache.used += 1;
                    }
                    cache.cells[seq_id].pos = batch.pos[i];
                    // NOTE: seq_ids are not inserted here; they are handled when the input tensors are set
                } else {
                    // too big seq_id
                    // TODO: would it be possible to resize the KV cache size instead?
                    LLAMA_LOG_ERROR("%s: seq_id=%d >= kv_size=%d Try using a bigger --parallel value\n", __func__, seq_id, cache.size);
                    return false;
                }
            }
        }

        // allow getting the range of used cells, from head to head + n
        cache.head = min;
        cache.n    = max - min + 1;

        // sanity check
        return max >= min;
    }
    // otherwise, one cell per token.

    if (n_tokens > n_ctx) {
        LLAMA_LOG_ERROR("%s: n_tokens=%d > n_ctx=%d\n", __func__, n_tokens, n_ctx);
        return false;
    }

    uint32_t n_tested = 0;

    while (true) {
        if (cache.head + n_tokens > n_ctx) {
            n_tested += n_ctx - cache.head;
            cache.head = 0;
            continue;
        }

        bool found = true;
        for (uint32_t i = 0; i < n_tokens; i++) {
            if (cache.cells[cache.head + i].pos >= 0) {
                found = false;
                cache.head += i + 1;
                n_tested   += i + 1;
                break;
            }
        }

        if (found) {
            break;
        }

        if (n_tested >= n_ctx) {
            //LLAMA_LOG_ERROR("%s: failed to find a slot for %d tokens\n", __func__, n_tokens);
            return false;
        }
    }

    for (uint32_t i = 0; i < n_tokens; i++) {
        cache.cells[cache.head + i].pos = batch.pos[i];

        for (int32_t j = 0; j < batch.n_seq_id[i]; j++) {
            cache.cells[cache.head + i].seq_id.insert(batch.seq_id[i][j]);
        }
    }

    cache.used += n_tokens;

    return true;
}

// find how many cells are currently in use
static uint32_t llama_kv_cache_cell_max(const struct llama_kv_cache & cache) {
    for (uint32_t i = cache.size; i > 0; --i) {
        const llama_kv_cell & cell = cache.cells[i - 1];

        if (cell.pos >= 0 && !cell.is_empty()) {
            return i;
        }
    }

    return 0;
}

static bool llama_kv_cache_seq_rm(
        struct llama_kv_cache & cache,
                 llama_seq_id   seq_id,
                    llama_pos   p0,
                    llama_pos   p1) {
    uint32_t new_head = cache.size;

    if (p0 < 0) p0 = 0;
    if (p1 < 0) p1 = std::numeric_limits<llama_pos>::max();

    // models like Mamba can't have a state partially erased
    if (cache.recurrent) {
        if (seq_id >= (int64_t) cache.size) {
            // could be fatal
            return false;
        }
        if (0 <= seq_id) {
            // partial intersection is invalid
            if ((0 < p0 && p0 <= cache.cells[seq_id].pos) || (0 < p1 && p1 <= cache.cells[seq_id].pos)) {
                return false;
            }
        } else {
            // seq_id is negative, then the range should include everything or nothing
            if (p0 != p1 && (p0 != 0 || p1 != std::numeric_limits<llama_pos>::max())) {
                return false;
            }
        }
    }

    for (uint32_t i = 0; i < cache.size; ++i) {
        if (cache.cells[i].pos >= p0 && cache.cells[i].pos < p1) {
            if (seq_id < 0) {
                cache.cells[i].seq_id.clear();
            } else if (cache.cells[i].has_seq_id(seq_id)) {
                cache.cells[i].seq_id.erase(seq_id);
            } else {
                continue;
            }
            if (cache.cells[i].is_empty()) {
                // keep count of the number of used cells
                if (cache.cells[i].pos >= 0) cache.used--;

                cache.cells[i].pos = -1;
                if (new_head == cache.size) new_head = i;
            }
        }
    }

    // If we freed up a slot, set head to it so searching can start there.
    if (new_head != cache.size && new_head < cache.head) cache.head = new_head;

    return true;
}

static void llama_kv_cache_seq_cp(
        struct llama_kv_cache & cache,
                 llama_seq_id   seq_id_src,
                 llama_seq_id   seq_id_dst,
                    llama_pos   p0,
                    llama_pos   p1) {
    if (p0 < 0) p0 = 0;
    if (p1 < 0) p1 = std::numeric_limits<llama_pos>::max();

    if (cache.recurrent) {
        if ((uint32_t) seq_id_dst < cache.size && (uint32_t) seq_id_src < cache.size) {
            seq_id_src = cache.cells[seq_id_src].src;
            GGML_ASSERT((uint32_t) seq_id_src < cache.size);
            // intent to "copy from"
            // supports copy chains thanks to taking the source of the source
            cache.cells[seq_id_dst].src = seq_id_src;

            // preserve the "keep or clear" status of the copied sequence
            if (cache.cells[seq_id_src].has_seq_id(seq_id_src)) {
                cache.cells[seq_id_dst].seq_id.insert(seq_id_dst);
            } else {
                cache.cells[seq_id_dst].seq_id.erase(seq_id_dst);
            }

            cache.do_copy = true;

            cache.cells[seq_id_dst].pos = cache.cells[seq_id_src].pos;
        }
        return;
    }
    // otherwise, this is the KV cache of a Transformer-like model

    cache.head = 0;

    for (uint32_t i = 0; i < cache.size; ++i) {
        if (cache.cells[i].has_seq_id(seq_id_src) && cache.cells[i].pos >= p0 && cache.cells[i].pos < p1) {
            cache.cells[i].seq_id.insert(seq_id_dst);
        }
    }
}

static void llama_kv_cache_seq_keep(struct llama_kv_cache & cache, llama_seq_id seq_id) {
    uint32_t new_head = cache.size;

    for (uint32_t i = 0; i < cache.size; ++i) {
        if (!cache.cells[i].has_seq_id(seq_id)) {
            if (cache.cells[i].pos >= 0) cache.used--;
            cache.cells[i].pos = -1;
            cache.cells[i].seq_id.clear();
            if (new_head == cache.size) new_head = i;
        } else {
            cache.cells[i].seq_id.clear();
            cache.cells[i].seq_id.insert(seq_id);
        }
    }

    // If we freed up a slot, set head to it so searching can start there.
    if (new_head != cache.size && new_head < cache.head) cache.head = new_head;
}

static void llama_kv_cache_seq_add(
        struct llama_kv_cache & cache,
                 llama_seq_id   seq_id,
                    llama_pos   p0,
                    llama_pos   p1,
                    llama_pos   delta) {
    uint32_t new_head = cache.size;

    if (p0 < 0) p0 = 0;
    if (p1 < 0) p1 = std::numeric_limits<llama_pos>::max();

    if (cache.recurrent) {
        // for Mamba-like models, only the pos needs to be shifted
        if (0 <= seq_id && seq_id < (int64_t) cache.size) {
            llama_kv_cell & cell = cache.cells[seq_id];
            if (cell.has_seq_id(seq_id) && p0 <= cell.pos && cell.pos < p1) {
                cell.pos += delta;
            }
        }
        return;
    }

    for (uint32_t i = 0; i < cache.size; ++i) {
        if (cache.cells[i].has_seq_id(seq_id) && cache.cells[i].pos >= p0 && cache.cells[i].pos < p1) {
            cache.has_shift = true;
            cache.cells[i].pos   += delta;
            cache.cells[i].delta += delta;

            if (cache.cells[i].pos < 0) {
                if (!cache.cells[i].is_empty()) {
                    cache.used--;
                }
                cache.cells[i].pos = -1;
                cache.cells[i].seq_id.clear();
                if (new_head == cache.size) {
                    new_head = i;
                }
            }
        }
    }

    // If we freed up a slot, set head to it so searching can start there.
    // Otherwise we just start the next search from the beginning.
    cache.head = new_head != cache.size ? new_head : 0;
}

static void llama_kv_cache_seq_div(
        struct llama_kv_cache & cache,
                 llama_seq_id   seq_id,
                    llama_pos   p0,
                    llama_pos   p1,
                          int   d) {
    if (p0 < 0) p0 = 0;
    if (p1 < 0) p1 = std::numeric_limits<llama_pos>::max();

    if (cache.recurrent) {
        // for Mamba-like models, only the pos needs to be changed
        if (0 <= seq_id && seq_id < (int64_t) cache.size) {
            llama_kv_cell & cell = cache.cells[seq_id];
            if (cell.has_seq_id(seq_id) && p0 <= cell.pos && cell.pos < p1) {
                cell.pos /= d;
            }
        }
        return;
    }

    for (uint32_t i = 0; i < cache.size; ++i) {
        if (cache.cells[i].has_seq_id(seq_id) && cache.cells[i].pos >= p0 && cache.cells[i].pos < p1) {
            cache.has_shift = true;

            {
                llama_pos p_old = cache.cells[i].pos;
                cache.cells[i].pos   /= d;
                cache.cells[i].delta += cache.cells[i].pos - p_old;
            }
        }
    }
}

static llama_pos llama_kv_cache_seq_pos_max(struct llama_kv_cache & cache, llama_seq_id seq_id) {
    llama_pos result = 0;

    for (uint32_t i = 0; i < cache.size; ++i) {
        if (cache.cells[i].has_seq_id(seq_id)) {
            result = std::max(result, cache.cells[i].pos);
        }
    }

    return result;
}

static void llama_kv_cache_defrag(struct llama_kv_cache & cache) {
    cache.do_defrag = true;
}
#endif

M_END_NAMESPACE
