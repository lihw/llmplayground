/** 
* The common stuff
*
* Author: lihw81@gmail.com
*/

#include <common/m_defs.h>

#include <fmt/core.h>

#include <map>

M_BEGIN_NAMESPACE

static const std::map<Kv, const char *> KV_NAMES = {
    { Kv::GENERAL_ARCHITECTURE,          "general.architecture"                  },
    { Kv::GENERAL_QUANTIZATION_VERSION,  "general.quantization_version"          },
    { Kv::GENERAL_ALIGNMENT,             "general.alignment"                     },
    { Kv::GENERAL_NAME,                  "general.name"                          },
    { Kv::GENERAL_AUTHOR,                "general.author"                        },
    { Kv::GENERAL_VERSION,               "general.version"                       },
    { Kv::GENERAL_URL,                   "general.url"                           },
    { Kv::GENERAL_DESCRIPTION,           "general.description"                   },
    { Kv::GENERAL_LICENSE,               "general.license"                       },
    { Kv::GENERAL_SOURCE_URL,            "general.source.url"                    },
    { Kv::GENERAL_SOURCE_HF_REPO,        "general.source.huggingface.repository" },

    { Kv::VOCAB_SIZE,                    "%s.vocab_size"            },
    { Kv::CONTEXT_LENGTH,                "%s.context_length"        },
    { Kv::EMBEDDING_LENGTH,              "%s.embedding_length"      },
    { Kv::BLOCK_COUNT,                   "%s.block_count"           },
    { Kv::FEED_FORWARD_LENGTH,           "%s.feed_forward_length"   },
    { Kv::USE_PARALLEL_RESIDUAL,         "%s.use_parallel_residual" },
    { Kv::TENSOR_DATA_LAYOUT,            "%s.tensor_data_layout"    },
    { Kv::EXPERT_COUNT,                  "%s.expert_count"          },
    { Kv::EXPERT_USED_COUNT,             "%s.expert_used_count"     },
    { Kv::POOLING_TYPE ,                 "%s.pooling_type"          },
    { Kv::LOGIT_SCALE,                   "%s.logit_scale"           },

    { Kv::ATTENTION_HEAD_COUNT,          "%s.attention.head_count"             },
    { Kv::ATTENTION_HEAD_COUNT_KV,       "%s.attention.head_count_kv"          },
    { Kv::ATTENTION_MAX_ALIBI_BIAS,      "%s.attention.max_alibi_bias"         },
    { Kv::ATTENTION_CLAMP_KQV,           "%s.attention.clamp_kqv"              },
    { Kv::ATTENTION_KEY_LENGTH,          "%s.attention.key_length"             },
    { Kv::ATTENTION_VALUE_LENGTH,        "%s.attention.value_length"           },
    { Kv::ATTENTION_LAYERNORM_EPS,       "%s.attention.layer_norm_epsilon"     },
    { Kv::ATTENTION_LAYERNORM_RMS_EPS,   "%s.attention.layer_norm_rms_epsilon" },
    { Kv::ATTENTION_CAUSAL,              "%s.attention.causal"                 },

    { Kv::ROPE_DIMENSION_COUNT,          "%s.rope.dimension_count"                 },
    { Kv::ROPE_FREQ_BASE,                "%s.rope.freq_base"                       },
    { Kv::ROPE_SCALE_LINEAR,             "%s.rope.scale_linear"                    },
    { Kv::ROPE_SCALING_TYPE,             "%s.rope.scaling.type"                    },
    { Kv::ROPE_SCALING_FACTOR,           "%s.rope.scaling.factor"                  },
    { Kv::ROPE_SCALING_ORIG_CTX_LEN,     "%s.rope.scaling.original_context_length" },
    { Kv::ROPE_SCALING_FINETUNED,        "%s.rope.scaling.finetuned"               },

    { Kv::SPLIT_NO,                      "split.no"            },
    { Kv::SPLIT_COUNT,                   "split.count"         },
    { Kv::SPLIT_TENSORS_COUNT,           "split.tensors.count" },

    { Kv::SSM_CONV_KERNEL,               "%s.ssm.conv_kernel"    },
    { Kv::SSM_INNER_SIZE,                "%s.ssm.inner_size"     },
    { Kv::SSM_STATE_SIZE,                "%s.ssm.state_size"     },
    { Kv::SSM_TIME_STEP_RANK,            "%s.ssm.time_step_rank" },

    { Kv::TOKENIZER_MODEL,               "tokenizer.ggml.model"              },
    { Kv::TOKENIZER_LIST,                "tokenizer.ggml.tokens"             },
    { Kv::TOKENIZER_TOKEN_TYPE,          "tokenizer.ggml.token_type"         },
    { Kv::TOKENIZER_TOKEN_TYPE_COUNT,    "tokenizer.ggml.token_type_count"   },
    { Kv::TOKENIZER_SCORES,              "tokenizer.ggml.scores"             },
    { Kv::TOKENIZER_MERGES,              "tokenizer.ggml.merges"             },
    { Kv::TOKENIZER_BOS_ID,              "tokenizer.ggml.bos_token_id"       },
    { Kv::TOKENIZER_EOS_ID,              "tokenizer.ggml.eos_token_id"       },
    { Kv::TOKENIZER_UNK_ID,              "tokenizer.ggml.unknown_token_id"   },
    { Kv::TOKENIZER_SEP_ID,              "tokenizer.ggml.seperator_token_id" },
    { Kv::TOKENIZER_PAD_ID,              "tokenizer.ggml.padding_token_id"   },
    { Kv::TOKENIZER_CLS_ID,              "tokenizer.ggml.cls_token_id"       },
    { Kv::TOKENIZER_MASK_ID,             "tokenizer.ggml.mask_token_id"      },
    { Kv::TOKENIZER_ADD_BOS,             "tokenizer.ggml.add_bos_token"      },
    { Kv::TOKENIZER_ADD_EOS,             "tokenizer.ggml.add_eos_token"      },
    { Kv::TOKENIZER_ADD_PREFIX,          "tokenizer.ggml.add_space_prefix"   },
    { Kv::TOKENIZER_HF_JSON,             "tokenizer.huggingface.json"        },
    { Kv::TOKENIZER_RWKV,                "tokenizer.rwkv.world"              },
    { Kv::TOKENIZER_PREFIX_ID,           "tokenizer.ggml.prefix_token_id"    },
    { Kv::TOKENIZER_SUFFIX_ID,           "tokenizer.ggml.suffix_token_id"    },
    { Kv::TOKENIZER_MIDDLE_ID,           "tokenizer.ggml.middle_token_id"    },
    { Kv::TOKENIZER_EOT_ID,              "tokenizer.ggml.eot_token_id"       },
};

static const std::map<Arch, const char *> ARCH_NAMES = {
    { Arch::LLAMA,           "llama"      },
    { Arch::FALCON,          "falcon"     },
    { Arch::GROK,            "grok"       },
    { Arch::GPT2,            "gpt2"       },
    { Arch::GPTJ,            "gptj"       },
    { Arch::GPTNEOX,         "gptneox"    },
    { Arch::MPT,             "mpt"        },
    { Arch::BAICHUAN,        "baichuan"   },
    { Arch::STARCODER,       "starcoder"  },
    { Arch::PERSIMMON,       "persimmon"  },
    { Arch::REFACT,          "refact"     },
    { Arch::BERT,            "bert"       },
    { Arch::NOMIC_BERT,      "nomic-bert" },
    { Arch::BLOOM,           "bloom"      },
    { Arch::STABLELM,        "stablelm"   },
    { Arch::QWEN,            "qwen"       },
    { Arch::QWEN2,           "qwen2"      },
    { Arch::QWEN2MOE,        "qwen2moe"   },
    { Arch::PHI2,            "phi2"       },
    { Arch::PLAMO,           "plamo"      },
    { Arch::CODESHELL,       "codeshell"  },
    { Arch::ORION,           "orion"      },
    { Arch::INTERNLM2,       "internlm2"  },
    { Arch::MINICPM,         "minicpm"    },
    { Arch::GEMMA,           "gemma"      },
    { Arch::STARCODER2,      "starcoder2" },
    { Arch::MAMBA,           "mamba"      },
    { Arch::XVERSE,          "xverse"     },
    { Arch::COMMAND_R,       "command-r"  },
    { Arch::DBRX,            "dbrx"       },
    { Arch::UNKNOWN,         "(unknown)"  },
};

Arch getArchFromString(const std::string& name) noexcept 
{
    for (const auto & kv : ARCH_NAMES) { // NOLINT
        if (kv.second == name) {
            return kv.first;
        }
    }

    return Arch::UNKNOWN;
}

std::string getKvString(Kv kv, Arch arch) noexcept
{
    return fmt::format(KV_NAMES.at(kv), ARCH_NAMES.at(arch));
}

std::string getKvString(Kv kv, const std::string& archName) noexcept
{
    return fmt::format(KV_NAMES.at(kv), archName);
}

M_END_NAMESPACE
