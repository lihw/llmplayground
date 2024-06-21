/**
 * The model
 * Author: lihw81@gmail.com
*/ 
 
#include <common/m_model.h>

#include <common/m_model_loader.h>
#include <common/m_model_loader_gguf.h>

#include <ggml/ggml.h>
#include <ggml/ggml-backend.h>
#include <spdlog/spdlog.h>

#include <cassert>

M_BEGIN_NAMESPACE

static const std::map<Arch, std::map<Tensor, std::string>> TENSOR_NAMES = {
    {
        Arch::LLAMA,
        {
            { Tensor::TOKEN_EMBD,      "token_embd" },
            { Tensor::OUTPUT_NORM,     "output_norm" },
            { Tensor::OUTPUT,          "output" },
            { Tensor::ROPE_FREQS,      "rope_freqs" },
            { Tensor::ATTN_NORM,       "blk.%d.attn_norm" },
            { Tensor::ATTN_Q,          "blk.%d.attn_q" },
            { Tensor::ATTN_K,          "blk.%d.attn_k" },
            { Tensor::ATTN_V,          "blk.%d.attn_v" },
            { Tensor::ATTN_OUT,        "blk.%d.attn_output" },
            { Tensor::ATTN_ROT_EMBD,   "blk.%d.attn_rot_embd" },
            { Tensor::FFN_GATE_INP,    "blk.%d.ffn_gate_inp" },
            { Tensor::FFN_NORM,        "blk.%d.ffn_norm" },
            { Tensor::FFN_GATE,        "blk.%d.ffn_gate" },
            { Tensor::FFN_DOWN,        "blk.%d.ffn_down" },
            { Tensor::FFN_UP,          "blk.%d.ffn_up" },
            { Tensor::FFN_GATE_EXP,    "blk.%d.ffn_gate.%d" },
            { Tensor::FFN_DOWN_EXP,    "blk.%d.ffn_down.%d" },
            { Tensor::FFN_UP_EXP,      "blk.%d.ffn_up.%d" },
            { Tensor::FFN_GATE_EXPS,   "blk.%d.ffn_gate_exps" },
            { Tensor::FFN_DOWN_EXPS,   "blk.%d.ffn_down_exps" },
            { Tensor::FFN_UP_EXPS,     "blk.%d.ffn_up_exps" },
        },
    }
};

Model::Model() {
}

Model::~Model() {
    for (struct ggml_context * ctx : mContexts) {
        ggml_free(ctx);
    }
    for (ggml_backend_buffer_t buf : mBuffers) {
#ifdef GGML_USE_CUDA
        if (ggml_backend_buffer_get_type(buf) == ggml_backend_cpu_buffer_type()) {
            ggml_backend_cuda_unregister_host_buffer(ggml_backend_buffer_get_base(buf));
        }
#endif
        ggml_backend_buffer_free(buf);
    }
}

struct TensorNameTranslator {
    TensorNameTranslator(Arch arch) : arch(arch) {}

    Arch arch;

    std::string operator()(Tensor tensor) const {
        if (TENSOR_NAMES.at(arch).find(tensor) == TENSOR_NAMES.at(arch).end()) {
            return "__missing__";
        }
        return TENSOR_NAMES.at(arch).at(tensor);
    }

    std::string operator()(Tensor tensor, const std::string & suffix) const {
        if (TENSOR_NAMES.at(arch).find(tensor) == TENSOR_NAMES.at(arch).end()) {
            return "__missing__";
        }
        return TENSOR_NAMES.at(arch).at(tensor) + "." + suffix;
    }

    std::string operator()(Tensor tensor, int bid) const {
        if (TENSOR_NAMES.at(arch).find(tensor) == TENSOR_NAMES.at(arch).end()) {
            return "__missing__";
        }
        char buf[256];
        snprintf(buf, 256, TENSOR_NAMES.at(arch).at(tensor).c_str(), bid);
        return std::string(buf);
    }

    std::string operator()(Tensor tensor, const std::string & suffix, int bid) const {
        if (TENSOR_NAMES.at(arch).find(tensor) == TENSOR_NAMES.at(arch).end()) {
            return "__missing__";
        }
        char buf[256];
        snprintf(buf, 256, TENSOR_NAMES.at(arch).at(tensor).c_str(), bid);
        return std::string(buf) + "." + suffix;
    }

    std::string operator()(Tensor tensor, const std::string & suffix, int bid, int xid) const {
        if (TENSOR_NAMES.at(arch).find(tensor) == TENSOR_NAMES.at(arch).end()) {
            return "__missing__";
        }
        char buf[256];
        snprintf(buf, 256, TENSOR_NAMES.at(arch).at(tensor).c_str(), bid, xid);
        return std::string(buf) + "." + suffix;
    }
};

bool Model::loadParameters(ModelLoader& ml)
{
    const gguf_context* ctx = ml.getContext();

    // get metadata as string
    for (int i = 0; i < gguf_get_n_kv(ctx); i++) {
        gguf_type t = gguf_get_kv_type(ctx, i);
        if (t == GGUF_TYPE_ARRAY) {
            continue;
        }
        const char * n = gguf_get_key(ctx, i);
        const std::string v = ggufKvToStr(ctx, i);
        ggufKv.emplace(n, v);
    }

    if (ml.getArchName() == "llama") {
        arch = Arch::LLAMA;
    } else {
        assert(!"Unsupported model");
        return false;
    }

    // get general kv
    ml.getKey(Kv::GENERAL_NAME, name, false);

    // get hparams kv
    ml.getKey(Kv::VOCAB_SIZE,           params.vocabSize,       false) || ml.getArrayLength(Kv::TOKENIZER_LIST, params.vocabSize);
    ml.getKey(Kv::CONTEXT_LENGTH,       params.contextLength);
    ml.getKey(Kv::EMBEDDING_LENGTH,     params.embedingLength);
    ml.getKey(Kv::FEED_FORWARD_LENGTH,  params.feedForwardLength);
    ml.getKey(Kv::ATTENTION_HEAD_COUNT, params.attentionHeadCount);
    ml.getKey(Kv::BLOCK_COUNT,          params.layerCount);
    ml.getKey(Kv::EXPERT_COUNT,         params.expertCount,      false);
    ml.getKey(Kv::EXPERT_USED_COUNT,    params.expertUsedCount, false);

    assert(params.expertCount <= 60); // FIXME: why 60???
    assert(params.expertUsedCount <= params.expertCount);
    if (params.expertCount > 0) {
        assert(params.expertCount > 0);
    } else {
        assert(params.expertUsedCount == 0);
    }

    // n_head_kv is optional, default to n_head
    params.attentionHeadCountKv = params.attentionHeadCount;
    ml.getKey(Kv::ATTENTION_HEAD_COUNT_KV, params.attentionHeadCountKv, false);

    //bool rope_finetuned = false;
    //ml.get_key(LLM_KV_ROPE_SCALING_FINETUNED, rope_finetuned, false);
    //hparams.rope_finetuned = rope_finetuned;

    //hparams.n_yarn_orig_ctx = hparams.n_ctx_train;
    //ml.get_key(LLM_KV_ROPE_SCALING_ORIG_CTX_LEN, hparams.n_yarn_orig_ctx, false);

    //// rope_freq_base (optional)
    //hparams.rope_freq_base_train = 10000.0f;
    //ml.get_key(LLM_KV_ROPE_FREQ_BASE, hparams.rope_freq_base_train, false);

    //std::string rope_scaling("linear");
    //ml.get_key(LLM_KV_ROPE_SCALING_TYPE, rope_scaling, false);
    //hparams.rope_scaling_type_train = llama_rope_scaling_type_from_string(rope_scaling);
    //GGML_ASSERT(hparams.rope_scaling_type_train != LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED);

    //// rope_freq_scale (inverse of the kv) is optional
    //float ropescale = 0.0f;
    //if (!ml.get_key(LLM_KV_ROPE_SCALING_FACTOR, ropescale, false)) {
    //    // try the old key name
    //    ml.get_key(LLM_KV_ROPE_SCALE_LINEAR, ropescale, false);
    //}
    //hparams.rope_freq_scale_train = ropescale == 0.0f ? 1.0f : 1.0f/ropescale;

    // sanity check for n_rot (optional)
    //{
    //    hparams.n_rot = (hparams.n_head == 0) ? 0 : hparams.embedingLength / hparams.n_head;

    //    ml.get_key(LLM_KV_ROPE_DIMENSION_COUNT, hparams.n_rot, false);

    //    if (model.arch == LLM_ARCH_LLAMA || model.arch == LLM_ARCH_FALCON) {
    //        if (hparams.n_rot != hparams.embedingLength / hparams.n_head) {
    //            throw std::runtime_error(format("invalid n_rot: %u, expected %u", hparams.n_rot, hparams.embedingLength / hparams.n_head));
    //        }
    //    }
    //    // gpt-neox n_rot = rotary_pct * (embedingLength / n_head)
    //    // gpt-j n_rot = rotary_dim
    //}

    params.attentionKeyLength = (params.attentionHeadCount == 0) ? 0 : params.embedingLength / params.attentionHeadCount;
    ml.getKey(Kv::ATTENTION_KEY_LENGTH, params.attentionKeyLength, false);

    params.attentionValueLength = (params.attentionHeadCount == 0) ? 0 : params.embedingLength / params.attentionHeadCount;
    ml.getKey(Kv::ATTENTION_VALUE_LENGTH, params.attentionValueLength, false);

    // arch-specific KVs
    if (arch == Arch::LLAMA) {
        // getKey(Kv::ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);

        if (params.expertCount == 8) {
            switch (params.layerCount) {
            case 32:
                type = Model::Type::MODEL_8x7B;
                break;
            case 56:
                type = Model::Type::MODEL_8x22B;
                break;
            default:
                type = Model::Type::MODEL_UNKNOWN;
            }
        } else {
            switch (params.layerCount) {
            case 22:
                type = Model::Type::MODEL_1B;
                break;
            case 26:
                type = Model::Type::MODEL_3B;
                break;
            case 32:
                type = Model::Type::MODEL_7B;
                break;
            case 40:
                type = Model::Type::MODEL_13B;
                break;
            case 48:
                type = Model::Type::MODEL_34B;
                break;
            case 60:
                type = Model::Type::MODEL_30B;
                break;
            case 80:
                type = params.attentionHeadCount == params.attentionHeadCountKv? Model::Type::MODEL_65B : Model::Type::MODEL_70B;
                break;
            default:
                type = Model::Type::MODEL_UNKNOWN;
            }
        }
    } else {
        assert(!"unknown model arch!");
        return false;
    }

    return true;
}
    
bool Model::loadVocab(ModelLoader& ml) 
{
    return vocab.load(ml);
}

static ggml_backend_buffer_type_t getDefaultBufferTypeCpu(bool host_buffer) {
    ggml_backend_buffer_type_t buft = nullptr;

#if defined(GGML_USE_CUDA)
    // host buffers should only be used when data is expected to be copied to/from the GPU
    if (host_buffer) {
        buft = ggml_backend_cuda_host_buffer_type();
    }
#elif defined(GGML_USE_SYCL)
    if (host_buffer) {
        buft = ggml_backend_sycl_host_buffer_type();
    }
#elif defined(GGML_USE_CPU_HBM)
    buft = ggml_backend_cpu_hbm_buffer_type();
#elif defined(GGML_USE_VULKAN)
    if (host_buffer) {
        buft = ggml_backend_vk_host_buffer_type();
    }
#endif

    if (buft == nullptr) {
        buft = ggml_backend_cpu_buffer_type();
    }
    return buft;

    GGML_UNUSED(host_buffer);
}

#if 0
static ggml_backend_buffer_type_t getDefaultBufferTypeOffload(int gpu) 
{
    ggml_backend_buffer_type_t buft = nullptr;

//#ifdef GGML_USE_METAL
//    buft = ggml_backend_metal_buffer_type();
//#elif defined(GGML_USE_CUDA)
//    buft = ggml_backend_cuda_buffer_type(gpu);
//#elif defined(GGML_USE_VULKAN)
//    buft = ggml_backend_vk_buffer_type(gpu);
//#elif defined(GGML_USE_SYCL)
//    buft = ggml_backend_sycl_buffer_type(gpu);
//#elif defined(GGML_USE_CLBLAST)
//    buft = ggml_backend_opencl_buffer_type();
//#elif defined(GGML_USE_KOMPUTE)
//    buft = ggml_backend_kompute_buffer_type(gpu);
//    if (buft == nullptr) {
//        LLAMA_LOG_WARN("%s: cannot use GPU %d, check `vulkaninfo --summary`\n", __func__, gpu);
//    }
//#endif

    if (buft == nullptr) {
        buft = getDefaultBufferTypeCpu(true);
    }
    return buft;

    GGML_UNUSED(gpu);
}
#endif

bool Model::loadTensors(ModelLoader& ml, int mainGpu, int32_t numGpuLayers, bool useMemoryLock)  
{
    auto LOG_HEAD = "Model::loadTensors()";

    mStartUs = ggml_time_us();

    //ml.get_key(LLM_KV_VOCAB_SIZE,           hparams.n_vocab,       false) || ml.get_arr_n(LLM_KV_TOKENIZER_LIST, hparams.n_vocab);
    //ml.get_key(LLM_KV_CONTEXT_LENGTH,       hparams.n_ctx_train);
    //ml.get_key(LLM_KV_EMBEDDING_LENGTH,     hparams.embedingLength);
    //ml.get_key(LLM_KV_FEED_FORWARD_LENGTH,  hparams.feedForwardLength);
    //ml.get_key(LLM_KV_ATTENTION_HEAD_COUNT, hparams.n_head);
    //ml.get_key(LLM_KV_BLOCK_COUNT,          hparams.n_layer);
    //ml.get_key(LLM_KV_EXPERT_COUNT,         hparams.n_expert,      false);
    //ml.get_key(LLM_KV_EXPERT_USED_COUNT,    hparams.n_expert_used, false);

    // FIXME: we don't support offloading to GPU yet.
    assert(mainGpu == -1 && numGpuLayers == 0);

    const int64_t numLayers = params.layerCount;

    // create one context per buffer type
    size_t ctxSize = ggml_tensor_overhead() * (ml.getTensorCount() + 1); // +1 for models where tok_embd is duplicated as output

    // for more merged tensors
    ctxSize += ggml_tensor_overhead() * params.expertCount * params.layerCount;
    
    // We only have one context and one buffer type. That is CPU.
    auto bufferType = getDefaultBufferTypeCpu(true);

    struct ggml_init_params p = {
            /*.mem_size   =*/ ctxSize,
            /*.mem_buffer =*/ NULL,
            /*.no_alloc   =*/ true,
        };
    {
        ggml_context * ctx = ggml_init(p);
        if (!ctx) {
            throw std::runtime_error("failed to create context");
        }
        mContexts.push_back(ctx); // We only have one context that is to compute in CPU.
        spdlog::info("{}: ggml ctx size = {:7.2} MiB", LOG_HEAD, mContexts.size() * ctxSize / 1024.0 / 1024.0);
    }
    

    // create tensors for the weights
    {
        const int64_t embedingLength    = params.embedingLength;
        const int64_t embedingKGqa      = params.attentionKeyLength * params.attentionHeadCountKv;
        const int64_t embedingVGqa      = params.attentionValueLength * params.attentionHeadCountKv;
        const int64_t embedingGqa       = embedingVGqa;
        const int64_t vocabSize         = params.vocabSize;
        //const int64_t vocabType         = params.vocabTypeCount;
        const int64_t feedForwardLength = params.feedForwardLength;
        const int64_t expertCount       = params.expertCount;

        if (expertCount > 0 && params.expertUsedCount == 0) {
            throw std::runtime_error("model has expert layers but no expert layers are used");
        }

        assert(embedingGqa == embedingKGqa);

        ggml_context * ctxInput        = mContexts[0];
        ggml_context * ctxOutput       = mContexts[0];
        ggml_context * ctxOutputSplit  = mContexts[0];
        auto ctxForLayer               = [&]() { return mContexts[0]; };
        auto ctxForLayerSplit          = [&]() { return mContexts[0]; };

        mLayers.resize(numLayers);

        const auto tn = TensorNameTranslator(arch);
        if (arch == Arch::LLAMA) {
            // input
            mTensors.token_embed = ml.createTensor(ctxInput, tn(Tensor::TOKEN_EMBD, "weight"), { embedingLength, vocabSize });

            // output
            {
                mTensors.output_norm = ml.createTensor(ctxOutput, tn(Tensor::OUTPUT_NORM, "weight"), { embedingLength });
                mTensors.output = ml.createTensor(ctxOutputSplit, tn(Tensor::OUTPUT, "weight"), { embedingLength, vocabSize}, false);
                    // if output is NULL, init from the input tok embed
                    if (mTensors.output == NULL) {
                        mTensors.output = ml.createTensor(ctxOutput, tn(Tensor::TOKEN_EMBD, "weight"), { embedingLength, vocabSize }, true, false);
                        //ml.size_data += ggml_nbytes(mTensors.output);
                    }
            }
            
            // layers
            for (int i = 0; i < numLayers; ++i) {
                ggml_context *ctxLayer = ctxForLayer();
                ggml_context *ctxSplit = ctxForLayerSplit();

                auto &layer = mLayers[i];
                
                /**
                  * embedingLength_head_k = (n_head == 0) ? 0 : embedingLength / n_head;
                  * ml.get_key(LLM_KV_ATTENTION_KEY_LENGTH, embedingLength_head_k, false);

                  * embedingLength_head_v = (n_head == 0) ? 0 : embedingLength / n_head;
                  * ml.get_key(LLM_KV_ATTENTION_VALUE_LENGTH, embedingLength_head_v, false);

                  * n_head_kv = n_head;
                  * ml.get_key(LLM_KV_ATTENTION_HEAD_COUNT_KV, n_head_kv, false);

                  * k: n_emd_k_gqa = n_head_kv * embedingLength_head_k // dimension of key embeddings across all k-v heads

                  * q: n_embed

                  * v: n_emd_v_gqa = n_head_kv * embedingLength_head_v // dimension of value embeddings across all k-v heads

                  * dim(k) == dim(v)
                  *
                  * FIXME: Normally, dim(k) = dim(v) = dim(q) = n_embed / n_head_count. Let us see if so in LLAMA 
                */
                layer.attn_norm = ml.createTensor(ctxLayer, tn(Tensor::ATTN_NORM, "weight", i), { embedingLength });

                layer.wq = ml.createTensor(ctxSplit, tn(Tensor::ATTN_Q, "weight", i), { embedingLength, embedingLength});
                layer.wk = ml.createTensor(ctxSplit, tn(Tensor::ATTN_K, "weight", i), { embedingLength, embedingKGqa});
                layer.wv = ml.createTensor(ctxSplit, tn(Tensor::ATTN_V, "weight", i), { embedingLength, embedingVGqa});
                layer.wo = ml.createTensor(ctxSplit, tn(Tensor::ATTN_OUT, "weight", i), { embedingLength, embedingLength});

                // optional bias tensors
                layer.bq = ml.createTensor(ctxLayer, tn(Tensor::ATTN_Q, "bias", i), { embedingLength}, false);
                layer.bk = ml.createTensor(ctxLayer, tn(Tensor::ATTN_K, "bias", i), { embedingKGqa}, false);
                layer.bv = ml.createTensor(ctxLayer, tn(Tensor::ATTN_V, "bias", i), { embedingVGqa}, false);
                layer.bo = ml.createTensor(ctxLayer, tn(Tensor::ATTN_OUT, "bias", i), { embedingLength}, false);

                layer.ffn_norm = ml.createTensor(ctxLayer, tn(Tensor::FFN_NORM, "weight", i), { embedingLength});

                layer.ffn_gate =
                    ml.createTensor(ctxSplit, tn(Tensor::FFN_GATE, "weight", i), { embedingLength, feedForwardLength });
                layer.ffn_down =
                    ml.createTensor(ctxSplit, tn(Tensor::FFN_DOWN, "weight", i), { feedForwardLength, embedingLength });
                layer.ffn_up = ml.createTensor(ctxSplit, tn(Tensor::FFN_UP, "weight", i), { embedingLength, feedForwardLength });
            }
        }
    }

    ml.areAllTensorsCreated();

    ml.initializeMappings(true, useMemoryLock? &mMemoryLocks: nullptr);
    //model.mappings.reserve(ml.mappings.size());
    
    using BufferMap = std::unordered_map<uint32_t, ggml_backend_buffer_t>;

    // create the backend buffers
    std::vector<std::pair<ggml_context *, BufferMap>> contextBuffers;
    contextBuffers.reserve(1);

    // Ensure we have enough capacity for the maximum backend buffer we will potentially create
    size_t maxBackendBuffers = ml.getFiles().size();
    mBuffers.reserve(maxBackendBuffers);

    BufferMap bufs;
    bufs.reserve(maxBackendBuffers);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(mContexts[0], bufferType);
    if (buf == nullptr) {
        throw std::runtime_error("unable to allocate backend buffer");
        return false;
    }
    mBuffers.push_back(buf);
    if (useMemoryLock && ggml_backend_buffer_is_host(buf)) {
        mMemoryLocks.emplace_back(new MemoryLock);
        auto &mlock = mMemoryLocks.back();
        mlock->initialize(ggml_backend_buffer_get_base(buf));
        mlock->growTo(ggml_backend_buffer_get_size(buf));
    }
    for (uint32_t idx = 0; idx < ml.getFiles().size(); idx++) {
        bufs.emplace(idx, buf);
    }

    if (bufs.empty()) {
        throw std::runtime_error("failed to allocate buffer");
        return false;
    }

    for (auto &b : bufs) {
        // indicate that this buffer contains weights
        // this is used by ggml_backend_sched to improve op scheduling -> ops that use a weight are preferably
        // scheduled to the backend that contains the weight
        ggml_backend_buffer_set_usage(b.second, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    }

    contextBuffers.emplace_back(mContexts[0], bufs);

    // print memory requirements
    for (ggml_backend_buffer_t b : mBuffers) {
        spdlog::info("{}: {:10s} buffer size = {:8.2f} MiB\n",
            LOG_HEAD,
            ggml_backend_buffer_name(b),
            ggml_backend_buffer_get_size(b) / 1024.0 / 1024.0);
    }

    // populate tensors_by_name
    for (ggml_context *c : mContexts) {
        for (auto *cur = ggml_get_first_tensor(c); cur != NULL; cur = ggml_get_next_tensor(c, cur)) {
            mTensorsByName.emplace_back(ggml_get_name(cur), cur);
        }
    }

    // load tensor data
    for (auto &it : contextBuffers) {
        if (!ml.loadData(it.first, it.second, useMemoryLock? &mMemoryLocks : NULL)) {
            return false;
        }
    }

    //if (use_mmap_buffer) {
    //    for (auto &mapping : ml.mappings) {
    //        model.mappings.emplace_back(std::move(mapping));
    //    }
    //}

    // loading time will be recalculate after the first eval, so
    // we take page faults deferred by mmap() into consideration
    mLoadUs = ggml_time_us() - mStartUs;

    return true;
}


M_END_NAMESPACE
