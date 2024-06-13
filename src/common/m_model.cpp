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
            { Tensor::TOKEN_EMBD,      "tokeembedingLength" },
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
    if (ml.getArchName() == "LLAMA") {
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

    //model.split_mode   = split_mode;
    //model.main_gpu     = main_gpu;
    //model.n_gpu_layers = n_gpu_layers;

    const int64_t numLayers = params.layerCount;
    const int64_t gpuStartLayerIndex = std::max(numLayers - numGpuLayers, (int64_t)0);
    //bool use_mmap_buffer = true;

    // there is very little benefit to offloading the input layer, so always keep it on the CPU
    layerBufferTypeInput = getDefaultBufferTypeCpu(true);

    layerBufferTypes.resize(numLayers);
    // assign cpu layers
    for (int64_t i = 0; i < gpuStartLayerIndex; ++i) {
        layerBufferTypes[i] = getDefaultBufferTypeCpu(true);
    }

    //if (split_mode == LLAMA_SPLIT_MODE_LAYER) {
    //    // calculate the split points
    //    int device_count = llama_get_device_count();
    //    bool all_zero = tensor_split == nullptr || std::all_of(tensor_split, tensor_split + device_count, [](float x) { return x == 0.0f; });
    //    std::vector<float> splits(device_count);
    //    if (all_zero) {
    //        // default split, by free memory
    //        for (int i = 0; i < device_count; ++i) {
    //            splits[i] = llama_get_device_memory(i);
    //        }
    //    } else {
    //        std::copy(tensor_split, tensor_split + device_count, splits.begin());
    //    }

    //    // sum and normalize the splits to get the split points
    //    float split_sum = 0.0f;
    //    for (int i = 0; i < device_count; ++i) {
    //        split_sum += splits[i];
    //        splits[i] = split_sum;
    //    }
    //    for (int i = 0; i < device_count; ++i) {
    //        splits[i] /= split_sum;
    //    }

    //    // assign the repeating layers to the devices according to the splits
    //    int act_gpu_layers = std::min(n_gpu_layers, (int)n_layer + 1);
    //    for (int64_t i = i_gpu_start; i < n_layer; ++i) {
    //        int layer_gpu = std::upper_bound(splits.begin(), splits.begin() + device_count, float(i - i_gpu_start)/act_gpu_layers) - splits.begin();
    //        model.buft_layer[i] = llama_default_buffer_type_offload(layer_gpu);
    //    }
    //    // assign the output layer
    //    if (n_gpu_layers > n_layer) {
    //        int layer_gpu = std::upper_bound(splits.begin(), splits.begin() + device_count, float(act_gpu_layers - 1)/act_gpu_layers) - splits.begin();
    //        model.buft_output = llama_default_buffer_type_offload(layer_gpu);
    //    } else {
    //        model.buft_output = llama_default_buffer_type_cpu(true);
    //    }
    //} else {
    {
        ggml_backend_buffer_type_t splitLayerBufferType;
        //if (splitMode == LLAMA_SPLIT_MODE_ROW) {
        //    splitLayerBufferType = llama_default_buffer_type_split(main_gpu, tensor_split);
        //} else {
            // LLAMA_SPLIT_MODE_NONE or LLAMA_SPLIT_MODE_LAYER in backends where it is not supported
        splitLayerBufferType = getDefaultBufferTypeOffload(mainGpu);
        //}
        // assign the repeating layers
        for (int64_t i = gpuStartLayerIndex; i < numLayers; ++i) {
            layerBufferTypes[i] = {
                splitLayerBufferType,
                getDefaultBufferTypeOffload(mainGpu)
            };
        }
        // assign the output layer
        if (numGpuLayers > numLayers) {
            layerBufferTypeOutput = {
                splitLayerBufferType,
                getDefaultBufferTypeOffload(mainGpu)
            };
        } else {
            layerBufferTypeOutput = getDefaultBufferTypeCpu(true);
        }
    }

    // count used buffer types
    std::map<ggml_backend_buffer_type_t, int> layerBufferTypeCount{};
    layerBufferTypeCount[layerBufferTypeInput.bufferType]++;
    layerBufferTypeCount[layerBufferTypeInput.bufferTypeMatrix]++;
    layerBufferTypeCount[layerBufferTypeOutput.bufferType]++;
    layerBufferTypeCount[layerBufferTypeOutput.bufferTypeMatrix]++;
    for (int64_t i = 0; i < params.layerCount; ++i) {
        layerBufferTypeCount[layerBufferTypes[i].bufferTypeMatrix]++;
        layerBufferTypeCount[layerBufferTypes[i].bufferType]++;
    }

    // create one context per buffer type
    size_t ctx_size = ggml_tensor_overhead() * (ml.getTensorCount() + 1); // +1 for models where tok_embd is duplicated as output

    // for moe merged tensors
    ctx_size += ggml_tensor_overhead() * params.expertCount * params.layerCount;

    std::map<ggml_backend_buffer_type_t, ggml_context *> type2contexts;
    for (auto & it : layerBufferTypeCount) {
        struct ggml_init_params p = {
            /*.mem_size   =*/ ctx_size,
            /*.mem_buffer =*/ NULL,
            /*.no_alloc   =*/ true,
        };
        ggml_context * ctx = ggml_init(p);
        if (!ctx) {
            throw std::runtime_error("failed to create context");
            return false;
        }
        type2contexts[it.first] = ctx;
        mContexts.push_back(ctx);
    }

    spdlog::info("{}: ggml ctx size = %7.2f MiB", LOG_HEAD, mContexts.size() * ctx_size/1024.0/1024.0);

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
            return false;
        }

        assert(embedingGqa == embedingKGqa);

        ggml_context * ctxInput        = type2contexts.at(layerBufferTypeInput.bufferType);
        ggml_context * ctxOutput       = type2contexts.at(layerBufferTypeOutput.bufferType);
        //ggml_context * ctxOutputSplit  = type2contexts.at(layerBufferTypeOutput.bufferTypeMatrix);
        auto ctxForLayer               = [&](int i) { return type2contexts.at(layerBufferTypes[i].bufferType); };
        auto ctxForLayerSplit          = [&](int i) { return type2contexts.at(layerBufferTypes[i].bufferTypeMatrix); };

        mLayers.resize(numLayers);

        //const auto tn = LLM_TN(model.arch);
        const auto tn = TensorNameTranslator(arch);
        if (arch == Arch::LLAMA) {
            mTensors.token_embed = ml.createTensor(ctxInput, tn(Tensor::TOKEN_EMBD, "weight"), { embedingLength, vocabSize});

            // output
            {
                mTensors.output_norm = ml.createTensor(ctxOutput, tn(Tensor::OUTPUT_NORM, "weight"), { embedingLength});
            }

            for (int i = 0; i < numLayers; ++i) {
                ggml_context *ctxLayer = ctxForLayer(i);
                ggml_context *ctxSplit = ctxForLayerSplit(i);

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

                //if (expertCount == 0) {
                    layer.ffn_gate =
                        ml.createTensor(ctxSplit, tn(Tensor::FFN_GATE, "weight", i), { embedingLength, feedForwardLength });
                    layer.ffn_down =
                        ml.createTensor(ctxSplit, tn(Tensor::FFN_DOWN, "weight", i), { feedForwardLength, embedingLength });
                    layer.ffn_up = ml.createTensor(ctxSplit, tn(Tensor::FFN_UP, "weight", i), { embedingLength, feedForwardLength });
                //} else {
                //    layer.ffn_gate_inp =
                //        ml.createTensor(ctxLayer, tn(Tensor::FFN_GATE_INP, "weight", i), { embedingLength, expertCount });

                //    layer.ffn_gate_exps = ml.createTensor(
                //        ctxSplit, tn(Tensor::FFN_GATE_EXPS, "weight", i), { embedingLength, feedForwardLength, expertCount }, false);
                //    if (layer.ffn_gate_exps) {
                //        layer.ffn_down_exps = ml.createTensor(
                //            ctxSplit, tn(Tensor::FFN_DOWN_EXPS, "weight", i), { feedForwardLength, embedingLength, expertCount });
                //        layer.ffn_up_exps = ml.createTensor(
                //            ctxSplit, tn(Tensor::FFN_UP_EXPS, "weight", i), { embedingLength, feedForwardLength, expertCount });
                //    } else {
                //        // merge split expert into a single tensor for compatibility with older models
                //        // requires disabling mmap
                //        //use_mmap_buffer = false;

                //        ggml_type type_gate =
                //            ml.require_tensor_meta(tn(Tensor::FFN_GATE_EXP, "weight", i, 0).c_str())->type;
                //        ggml_type type_down =
                //            ml.require_tensor_meta(tn(Tensor::FFN_DOWN_EXP, "weight", i, 0).c_str())->type;
                //        ggml_type type_up =
                //            ml.require_tensor_meta(tn(Tensor::FFN_UP_EXP, "weight", i, 0).c_str())->type;

                //        layer.ffn_gate_exps = ggml_new_tensor_3d(ctxSplit, type_gate, embedingLength, feedForwardLength, expertCount);
                //        layer.ffn_down_exps = ggml_new_tensor_3d(ctxSplit, type_down, feedForwardLength, embedingLength, expertCount);
                //        layer.ffn_up_exps = ggml_new_tensor_3d(ctxSplit, type_up, embedingLength, feedForwardLength, expertCount);

                //        ggml_set_name(layer.ffn_gate_exps, tn(Tensor::FFN_GATE_EXPS, "weight", i).c_str());
                //        ggml_set_name(layer.ffn_down_exps, tn(Tensor::FFN_DOWN_EXPS, "weight", i).c_str());
                //        ggml_set_name(layer.ffn_up_exps, tn(Tensor::FFN_UP_EXPS, "weight", i).c_str());

                //        for (uint32_t x = 0; x < n_expert; ++x) {
                //            // the individual experts are loaded into a view of the merged tensor
                //            ml.createTensor_as_view(ctxSplit,
                //                layer.ffn_gate_exps,
                //                tn(Tensor::FFN_GATE_EXP, "weight", i, x),
                //                { embedingLength, feedForwardLength },
                //                layer.ffn_gate_exps->nb[2] * x);
                //            ml.createTensor_as_view(ctxSplit,
                //                layer.ffn_down_exps,
                //                tn(Tensor::FFN_DOWN_EXP, "weight", i, x),
                //                { feedForwardLength, embedingLength },
                //                layer.ffn_down_exps->nb[2] * x);
                //            ml.createTensor_as_view(ctxSplit,
                //                layer.ffn_up_exps,
                //                tn(Tensor::FFN_UP_EXP, "weight", i, x),
                //                { embedingLength, feedForwardLength },
                //                layer.ffn_up_exps->nb[2] * x);
                //        }
                //    }
                //}
            }
        }
    }

    ml.areAllTensorsCreated();

    //ml.init_mappings(true, useMemoryLock? &model.mlock_mmaps : nullptr);
    //model.mappings.reserve(ml.mappings.size());
    
    using BufferMap = std::unordered_map<uint32_t, ggml_backend_buffer_t>;

    // create the backend buffers
    std::vector<std::pair<ggml_context *, BufferMap>> contextBuffers;
    contextBuffers.reserve(type2contexts.size());

    // Ensure we have enough capacity for the maximum backend buffer we will potentially create
    size_t maxBackendBuffers = type2contexts.size() * ml.getFiles().size();
    mBuffers.reserve(maxBackendBuffers);

    for (auto &it : type2contexts) {
        ggml_backend_buffer_type_t buft = it.first;
        ggml_context *ctx = it.second;

        BufferMap bufs;
        bufs.reserve(maxBackendBuffers);

        // only the mmap region containing the tensors in the model is mapped to the backend buffer
        // this is important for metal with apple silicon: if the entire model could be mapped to a metal buffer,
        // then we could just use metal for all layers this allows using partial offloading when the model size
        // exceeds the metal buffer size, but not the RAM size
        //if (ml.use_mmap && use_mmap_buffer && buft == llama_default_buffer_type_cpu(true)) {
        //    for (uint32_t idx = 0; idx < ml.files.size(); idx++) {
        //        void *addr = nullptr;
        //        size_t first, last;
        //        ml.get_mapping_range(&first, &last, &addr, idx, ctx);
        //        if (first >= last) {
        //            continue;
        //        }
        //        ggml_backend_buffer_t buf = ggml_backend_cpu_buffer_from_ptr((char *)addr + first, last - first);
        //        if (buf == nullptr) {
        //            throw std::runtime_error("unable to allocate backend CPU buffer");
        //        }
        //        model.bufs.push_back(buf);
        //        bufs.emplace(idx, buf);
#ifdef GGML_USE_CUDA
                if (n_layer >= n_gpu_layers) {
                    ggml_backend_cuda_register_host_buffer(
                        ggml_backend_buffer_get_base(buf), ggml_backend_buffer_get_size(buf));
                }
#endif
        //    }
        //}
#ifdef GGML_USE_METAL
        else if (ml.use_mmap && use_mmap_buffer && buft == ggml_backend_metal_buffer_type()) {
            for (uint32_t idx = 0; idx < ml.files.size(); idx++) {
                const size_t max_size = ggml_get_max_tensor_size(ctx);
                void *addr = nullptr;
                size_t first, last;
                ml.get_mapping_range(&first, &last, &addr, idx, ctx);
                if (first >= last) {
                    continue;
                }
                ggml_backend_buffer_t buf =
                    ggml_backend_metal_buffer_from_ptr((char *)addr + first, last - first, max_size);
                if (buf == nullptr) {
                    throw std::runtime_error("unable to allocate backend metal buffer");
                }
                model.bufs.push_back(buf);
                bufs.emplace(idx, buf);
            }
        }
#endif
        //else {
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
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
        //}

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

        contextBuffers.emplace_back(ctx, bufs);
    }

    if (supportGpuOffload()) {
        const int numGpu = std::min(numGpuLayers, int(params.layerCount));

        spdlog::info("{}: offloading {} repeating layers to GPU", LOG_HEAD, numGpu);
        if (numGpuLayers > (int)params.layerCount) {
            spdlog::info("{}: offloading non-repeating layers to GPU", LOG_HEAD);
        }

        const int maxBackendSupportedLayers = params.layerCount + 1;
        const int maxOffloadableLayers = params.layerCount + 1;

        spdlog::info("{}: offloaded %d/%d layers to GPU\n",
            LOG_HEAD,
            std::min(numGpuLayers, maxOffloadableLayers),
            maxBackendSupportedLayers);
    }

    // print memory requirements
    for (ggml_backend_buffer_t buf : mBuffers) {
        spdlog::info("{}: {:10s} buffer size = {:8.2f} MiB\n",
            LOG_HEAD,
            ggml_backend_buffer_name(buf),
            ggml_backend_buffer_get_size(buf) / 1024.0 / 1024.0);
    }

    // populate tensors_by_name
    for (ggml_context *ctx : mContexts) {
        for (auto *cur = ggml_get_first_tensor(ctx); cur != NULL; cur = ggml_get_next_tensor(ctx, cur)) {
            mTensorsByName.emplace_back(ggml_get_name(cur), cur);
        }
    }

    // load tensor data
    for (auto &it : contextBuffers) {
        ggml_context *ctx = it.first;
        auto &bufs = it.second;
        if (!ml.loadData(
                ctx, bufs, useMemoryLock? &mMemoryLocks : NULL)) {
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
