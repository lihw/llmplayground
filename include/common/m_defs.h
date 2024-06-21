/**
* Misc definitions and helper structures
* Author: lihw81@gmail.com
*/


#ifndef M_DEFS_H
#define M_DEFS_H

#include <string>

#define M_NO_COPY_CONSTRUCTOR(Clazz) \
    Clazz(const Clazz&) = delete; \
    Clazz(Clazz&&) = delete; 
#define M_NO_MOVE_CONSTRUCTOR(Clazz) \
    const Clazz& operator=(const Clazz&) = delete; \
    const Clazz& operator=(Clazz&) = delete; 

#define M_BEGIN_NAMESPACE namespace m {

#define M_END_NAMESPACE };

M_BEGIN_NAMESPACE

enum class Kv {
    GENERAL_ARCHITECTURE,
    GENERAL_QUANTIZATION_VERSION,
    GENERAL_ALIGNMENT,
    GENERAL_NAME,
    GENERAL_AUTHOR,
    GENERAL_VERSION,
    GENERAL_URL,
    GENERAL_DESCRIPTION,
    GENERAL_LICENSE,
    GENERAL_SOURCE_URL,
    GENERAL_SOURCE_HF_REPO,

    VOCAB_SIZE,
    CONTEXT_LENGTH,
    EMBEDDING_LENGTH,
    BLOCK_COUNT,
    FEED_FORWARD_LENGTH,
    USE_PARALLEL_RESIDUAL,
    TENSOR_DATA_LAYOUT,
    EXPERT_COUNT,
    EXPERT_USED_COUNT,
    POOLING_TYPE,
    LOGIT_SCALE,

    ATTENTION_HEAD_COUNT,
    ATTENTION_HEAD_COUNT_KV,
    ATTENTION_MAX_ALIBI_BIAS,
    ATTENTION_CLAMP_KQV,
    ATTENTION_KEY_LENGTH,
    ATTENTION_VALUE_LENGTH,
    ATTENTION_LAYERNORM_EPS,
    ATTENTION_LAYERNORM_RMS_EPS,
    ATTENTION_CAUSAL,

    ROPE_DIMENSION_COUNT,
    ROPE_FREQ_BASE,
    ROPE_SCALE_LINEAR,
    ROPE_SCALING_TYPE,
    ROPE_SCALING_FACTOR,
    ROPE_SCALING_ORIG_CTX_LEN,
    ROPE_SCALING_FINETUNED,

    SPLIT_NO,
    SPLIT_COUNT,
    SPLIT_TENSORS_COUNT,

    SSM_INNER_SIZE,
    SSM_CONV_KERNEL,
    SSM_STATE_SIZE,
    SSM_TIME_STEP_RANK,

    TOKENIZER_MODEL,
    TOKENIZER_LIST,
    TOKENIZER_TOKEN_TYPE,
    TOKENIZER_TOKEN_TYPE_COUNT,
    TOKENIZER_SCORES,
    TOKENIZER_MERGES,
    TOKENIZER_BOS_ID,
    TOKENIZER_EOS_ID,
    TOKENIZER_UNK_ID,
    TOKENIZER_SEP_ID,
    TOKENIZER_PAD_ID,
    TOKENIZER_CLS_ID,
    TOKENIZER_MASK_ID,
    TOKENIZER_ADD_BOS,
    TOKENIZER_ADD_EOS,
    TOKENIZER_ADD_PREFIX,
    TOKENIZER_HF_JSON,
    TOKENIZER_RWKV,
    TOKENIZER_PREFIX_ID,
    TOKENIZER_SUFFIX_ID,
    TOKENIZER_MIDDLE_ID,
    TOKENIZER_EOT_ID,
};

enum class Arch {
    LLAMA,
    FALCON,
    BAICHUAN,
    GROK,
    GPT2,
    GPTJ,
    GPTNEOX,
    MPT,
    STARCODER,
    PERSIMMON,
    REFACT,
    BERT,
    NOMIC_BERT,
    BLOOM,
    STABLELM,
    QWEN,
    QWEN2,
    QWEN2MOE,
    PHI2,
    PLAMO,
    CODESHELL,
    ORION,
    INTERNLM2,
    MINICPM,
    GEMMA,
    STARCODER2,
    MAMBA,
    XVERSE,
    COMMAND_R,
    DBRX,
    UNKNOWN,
};

// The model weight data type
// For the different quanzation methonds, check out
// https://github.com/ggerganov/llama.cpp/pull/1684#issuecomment-1579252501
enum class GgufType {
    ALL_F32              = 0,
    MOSTLY_F16           = 1,  // except 1d tensors
    MOSTLY_Q4_0          = 2,  // except 1d tensors
    MOSTLY_Q4_1          = 3,  // except 1d tensors
    MOSTLY_Q4_1_SOME_F16 = 4,  // tok_embeddings.weight and output.weight are F16
    // MOSTLY_Q4_2       = 5,  // support has been removed
    // MOSTLY_Q4_3       = 6,  // support has been removed
    MOSTLY_Q8_0          = 7,  // except 1d tensors
    MOSTLY_Q5_0          = 8,  // except 1d tensors
    MOSTLY_Q5_1          = 9,  // except 1d tensors
    MOSTLY_Q2_K          = 10, // except 1d tensors
    MOSTLY_Q3_K_S        = 11, // except 1d tensors
    MOSTLY_Q3_K_M        = 12, // except 1d tensors
    MOSTLY_Q3_K_L        = 13, // except 1d tensors
    MOSTLY_Q4_K_S        = 14, // except 1d tensors
    MOSTLY_Q4_K_M        = 15, // except 1d tensors
    MOSTLY_Q5_K_S        = 16, // except 1d tensors
    MOSTLY_Q5_K_M        = 17, // except 1d tensors
    MOSTLY_Q6_K          = 18, // except 1d tensors
    MOSTLY_IQ2_XXS       = 19, // except 1d tensors
    MOSTLY_IQ2_XS        = 20, // except 1d tensors
    MOSTLY_Q2_K_S        = 21, // except 1d tensors
    MOSTLY_IQ3_XS        = 22, // except 1d tensors
    MOSTLY_IQ3_XXS       = 23, // except 1d tensors
    MOSTLY_IQ1_S         = 24, // except 1d tensors
    MOSTLY_IQ4_NL        = 25, // except 1d tensors
    MOSTLY_IQ3_S         = 26, // except 1d tensors
    MOSTLY_IQ3_M         = 27, // except 1d tensors
    MOSTLY_IQ2_S         = 28, // except 1d tensors
    MOSTLY_IQ2_M         = 29, // except 1d tensors
    MOSTLY_IQ4_XS        = 30, // except 1d tensors
    MOSTLY_IQ1_M         = 31, // except 1d tensors

    GUESSED = 1024, // not specified in the model file
};

extern const char* getGgufTypeName(GgufType type) noexcept;

enum class KvOverrideType {
    BOOL = 1,
    FLOAT = 2,
    INT = 3,
};

struct KvOverride {
    char key[128];
    enum KvOverrideType tag;
    union {
        int64_t intValue;
        double floatValue;
        bool boolValue;
    };
};

enum class GgufVersion {
    V1,
    V2,
    V3,
};

enum Tensor {
    TOKEN_EMBD,
    TOKEN_EMBD_NORM,
    TOKEN_TYPES,
    POS_EMBD,
    OUTPUT,
    OUTPUT_NORM,
    ROPE_FREQS,
    ATTN_Q,
    ATTN_K,
    ATTN_V,
    ATTN_QKV,
    ATTN_OUT,
    ATTN_NORM,
    ATTN_NORM_2,
    ATTN_OUT_NORM,
    ATTN_ROT_EMBD,
    FFN_GATE_INP,
    FFN_GATE_INP_SHEXP,
    FFN_NORM,
    FFN_GATE,
    FFN_DOWN,
    FFN_UP,
    FFN_ACT,
    FFN_DOWN_EXP,  // split experts for backward compatibility
    FFN_GATE_EXP,
    FFN_UP_EXP,
    FFN_DOWN_EXPS, // merged experts
    FFN_GATE_EXPS,
    FFN_UP_EXPS,
    FFN_DOWN_SHEXP,
    FFN_GATE_SHEXP,
    FFN_UP_SHEXP,
    ATTN_Q_NORM,
    ATTN_K_NORM,
    LAYER_OUT_NORM,
    SSM_IN,
    SSM_CONV1D,
    SSM_X,
    SSM_DT,
    SSM_A,
    SSM_D,
    SSM_OUT,
};

enum class PoolingType {
    UNSPECIFIED = -1,
    NONE = 0,
    MEAN = 1,
    CLS = 2,
};

extern Arch getArchFromString(const std::string& name) noexcept;

extern std::string getKvString(Kv kv, Arch arch) noexcept;

extern std::string getKvString(Kv kv, const std::string& archName) noexcept;

M_END_NAMESPACE

#endif //! M_DEFS_H
