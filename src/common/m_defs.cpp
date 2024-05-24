/** 
* The common stuff
*
* Author: lihw81@gmail.com
*/

#include <common/m_defs.h>

M_BEGIN_NAMESPACE

static const std::map<Kv, const char *> KV_NAMES = {
    { GENERAL_ARCHITECTURE,          "general.architecture"                  },
    { GENERAL_QUANTIZATION_VERSION,  "general.quantization_version"          },
    { GENERAL_ALIGNMENT,             "general.alignment"                     },
    { GENERAL_NAME,                  "general.name"                          },
    { GENERAL_AUTHOR,                "general.author"                        },
    { GENERAL_VERSION,               "general.version"                       },
    { GENERAL_URL,                   "general.url"                           },
    { GENERAL_DESCRIPTION,           "general.description"                   },
    { GENERAL_LICENSE,               "general.license"                       },
    { GENERAL_SOURCE_URL,            "general.source.url"                    },
    { GENERAL_SOURCE_HF_REPO,        "general.source.huggingface.repository" },

    { VOCAB_SIZE,                    "%s.vocab_size"            },
    { CONTEXT_LENGTH,                "%s.context_length"        },
    { EMBEDDING_LENGTH,              "%s.embedding_length"      },
    { BLOCK_COUNT,                   "%s.block_count"           },
    { FEED_FORWARD_LENGTH,           "%s.feed_forward_length"   },
    { USE_PARALLEL_RESIDUAL,         "%s.use_parallel_residual" },
    { TENSOR_DATA_LAYOUT,            "%s.tensor_data_layout"    },
    { EXPERT_COUNT,                  "%s.expert_count"          },
    { EXPERT_USED_COUNT,             "%s.expert_used_count"     },
    { POOLING_TYPE ,                 "%s.pooling_type"          },
    { LOGIT_SCALE,                   "%s.logit_scale"           },

    { ATTENTION_HEAD_COUNT,          "%s.attention.head_count"             },
    { ATTENTION_HEAD_COUNT_KV,       "%s.attention.head_count_kv"          },
    { ATTENTION_MAX_ALIBI_BIAS,      "%s.attention.max_alibi_bias"         },
    { ATTENTION_CLAMP_KQV,           "%s.attention.clamp_kqv"              },
    { ATTENTION_KEY_LENGTH,          "%s.attention.key_length"             },
    { ATTENTION_VALUE_LENGTH,        "%s.attention.value_length"           },
    { ATTENTION_LAYERNORM_EPS,       "%s.attention.layer_norm_epsilon"     },
    { ATTENTION_LAYERNORM_RMS_EPS,   "%s.attention.layer_norm_rms_epsilon" },
    { ATTENTION_CAUSAL,              "%s.attention.causal"                 },

    { ROPE_DIMENSION_COUNT,          "%s.rope.dimension_count"                 },
    { ROPE_FREQ_BASE,                "%s.rope.freq_base"                       },
    { ROPE_SCALE_LINEAR,             "%s.rope.scale_linear"                    },
    { ROPE_SCALING_TYPE,             "%s.rope.scaling.type"                    },
    { ROPE_SCALING_FACTOR,           "%s.rope.scaling.factor"                  },
    { ROPE_SCALING_ORIG_CTX_LEN,     "%s.rope.scaling.original_context_length" },
    { ROPE_SCALING_FINETUNED,        "%s.rope.scaling.finetuned"               },

    { SPLIT_NO,                      "split.no"            },
    { SPLIT_COUNT,                   "split.count"         },
    { SPLIT_TENSORS_COUNT,           "split.tensors.count" },

    { SSM_CONV_KERNEL,               "%s.ssm.conv_kernel"    },
    { SSM_INNER_SIZE,                "%s.ssm.inner_size"     },
    { SSM_STATE_SIZE,                "%s.ssm.state_size"     },
    { SSM_TIME_STEP_RANK,            "%s.ssm.time_step_rank" },

    { TOKENIZER_MODEL,               "tokenizer.ggml.model"              },
    { TOKENIZER_LIST,                "tokenizer.ggml.tokens"             },
    { TOKENIZER_TOKEN_TYPE,          "tokenizer.ggml.token_type"         },
    { TOKENIZER_TOKEN_TYPE_COUNT,    "tokenizer.ggml.token_type_count"   },
    { TOKENIZER_SCORES,              "tokenizer.ggml.scores"             },
    { TOKENIZER_MERGES,              "tokenizer.ggml.merges"             },
    { TOKENIZER_BOS_ID,              "tokenizer.ggml.bos_token_id"       },
    { TOKENIZER_EOS_ID,              "tokenizer.ggml.eos_token_id"       },
    { TOKENIZER_UNK_ID,              "tokenizer.ggml.unknown_token_id"   },
    { TOKENIZER_SEP_ID,              "tokenizer.ggml.seperator_token_id" },
    { TOKENIZER_PAD_ID,              "tokenizer.ggml.padding_token_id"   },
    { TOKENIZER_CLS_ID,              "tokenizer.ggml.cls_token_id"       },
    { TOKENIZER_MASK_ID,             "tokenizer.ggml.mask_token_id"      },
    { TOKENIZER_ADD_BOS,             "tokenizer.ggml.add_bos_token"      },
    { TOKENIZER_ADD_EOS,             "tokenizer.ggml.add_eos_token"      },
    { TOKENIZER_ADD_PREFIX,          "tokenizer.ggml.add_space_prefix"   },
    { TOKENIZER_HF_JSON,             "tokenizer.huggingface.json"        },
    { TOKENIZER_RWKV,                "tokenizer.rwkv.world"              },
    { TOKENIZER_PREFIX_ID,           "tokenizer.ggml.prefix_token_id"    },
    { TOKENIZER_SUFFIX_ID,           "tokenizer.ggml.suffix_token_id"    },
    { TOKENIZER_MIDDLE_ID,           "tokenizer.ggml.middle_token_id"    },
    { TOKENIZER_EOT_ID,              "tokenizer.ggml.eot_token_id"       },
};

static const std::map<Arch, const char *> ARCH_NAMES = {
    { LLAMA,           "llama"      },
    { FALCON,          "falcon"     },
    { GROK,            "grok"       },
    { GPT2,            "gpt2"       },
    { GPTJ,            "gptj"       },
    { GPTNEOX,         "gptneox"    },
    { MPT,             "mpt"        },
    { BAICHUAN,        "baichuan"   },
    { STARCODER,       "starcoder"  },
    { PERSIMMON,       "persimmon"  },
    { REFACT,          "refact"     },
    { BERT,            "bert"       },
    { NOMIC_BERT,      "nomic-bert" },
    { BLOOM,           "bloom"      },
    { STABLELM,        "stablelm"   },
    { QWEN,            "qwen"       },
    { QWEN2,           "qwen2"      },
    { QWEN2MOE,        "qwen2moe"   },
    { PHI2,            "phi2"       },
    { PLAMO,           "plamo"      },
    { CODESHELL,       "codeshell"  },
    { ORION,           "orion"      },
    { INTERNLM2,       "internlm2"  },
    { MINICPM,         "minicpm"    },
    { GEMMA,           "gemma"      },
    { STARCODER2,      "starcoder2" },
    { MAMBA,           "mamba"      },
    { XVERSE,          "xverse"     },
    { COMMAND_R,       "command-r"  },
    { DBRX,            "dbrx"       },
    { UNKNOWN,         "(unknown)"  },
};

Arch getArchFromString(const std::string& name) noexcept {
    for (const auto & kv : ARCH_NAMES) { // NOLINT
        if (kv.second == name) {
            return kv.first;
        }
    }

    return Arch::ARCH_UNKNOWN;
}

M_END_NAMESPACE
