/** 
* The vocabulary
*
* Author: lihw81@gmail.com
*/

#include <common/m_vocab.h>

#include <spdlog/spdlog.h>

#include <cassert>

M_BEGIN_NAMESPACE

void Vocab::load(llama_model_loader &ml, llama_model &model) noexcept
{
    auto &vocab = model.vocab;

    struct gguf_context *ctx = ml.meta;

    //const auto kv = LLM_KV(model.arch);

    // determine vocab type
    {
        std::string tokenizer_name;

        ml.get_key(LLM_KV_TOKENIZER_MODEL, tokenizer_name);

        if (tokenizer_name == "no_vocab") {
            type = Type::NONE;

            // default special tokens
            specialBosId = -1;
            specialEosId = -1;
            specialUnkId = -1;
            specialSepId = -1;
            specialPadId = -1;
            specialClsId = -1;
            specialMaskId = -1;
            lineFeedId = -1;

            return;
        } else if (tokenizer_name == "llama") {
            type = Type::SPM;

            // default special tokens
            specialBosId = 1;
            specialEosId = 2;
            specialUnkId = 0;
            specialSepId = -1;
            specialPadId = -1;
            specialClsId = -1;
            specialMaskId = -1;

            const int add_space_prefix_keyidx = gguf_find_key(ctx, kv(LLM_KV_TOKENIZER_ADD_PREFIX).c_str());
            if (add_space_prefix_keyidx != -1) {
                addSpacePrefix = gguf_get_val_bool(ctx, add_space_prefix_keyidx);
            }// The default value of add_space_prefix is true.
        } else if (tokenizer_name == "gpt2") {
            type = Type::BPE;

            // read bpe merges and populate bpe ranks
            const int merges_keyidx = gguf_find_key(ctx, kv(LLM_KV_TOKENIZER_MERGES).c_str());
            if (merges_keyidx == -1) {
                throw std::runtime_error("cannot find tokenizer merges in model file\n");
            }

            const int nMerges = gguf_get_arr_n(ctx, merges_keyidx);

            for (int i = 0; i < n_merges; i++) {
                const std::string word = gguf_get_arr_str(ctx, merges_keyidx, i);
                assert(unicode_cpts_from_utf8(word).size() > 0);

                std::string first;
                std::string second;

                const size_t pos = word.find(' ', 1);

                if (pos != std::string::npos) {
                    first = word.substr(0, pos);
                    second = word.substr(pos + 1);
                }

                bpeRanks.emplace(std::make_pair(first, second), i);
            }

            // default special tokens
            specialBosId = 11;
            specialEosId = 11;
            specialUnkId = -1;
            specialSepId = -1;
            specialPadId = -1;
            specialClsId = -1;
            specialMaskId = -1;
        } else if (tokenizer_name == "bert") {
            type = Type::WPM;

            // default special tokens
            specialBosId = -1;
            specialEosId = -1;
            specialUnkId = 100;
            specialSepId = 102;
            specialPadId = 0;
            specialClsId = 101;
            specialMaskId = 103;
            add_space_prefix = false;
        } else {
            spdlog::warn("%s: unknown tokenizer: '%s'", __func__, tokenizer_name.c_str());
            spdlog::warn("%s: using default tokenizer: 'llama'", __func__);

            type = Type::SPM
        }
    }

    const int token_idx = gguf_find_key(ctx, kv(LLM_KV_TOKENIZER_LIST).c_str());
    if (token_idx == -1) {
        throw std::runtime_error("cannot find tokenizer vocab in model file\n");
    }

    const float *scores = nullptr;
    const int score_idx = gguf_find_key(ctx, kv(LLM_KV_TOKENIZER_SCORES).c_str());
    if (score_idx != -1) {
        scores = (const float *)gguf_get_arr_data(ctx, score_idx);
    }

    const int *toktypes = nullptr;
    const int toktype_idx = gguf_find_key(ctx, kv(LLM_KV_TOKENIZER_TOKEN_TYPE).c_str());
    if (toktype_idx != -1) {
        toktypes = (const int *)gguf_get_arr_data(ctx, toktype_idx);
    }

    const uint32_t n_vocab = gguf_get_arr_n(ctx, token_idx);

    vocab.id_to_token.resize(n_vocab);

    for (uint32_t i = 0; i < n_vocab; i++) {
        std::string word = gguf_get_arr_str(ctx, token_idx, i);
        GGML_ASSERT(unicode_cpts_from_utf8(word).size() > 0);

        vocab.token_to_id[word] = i;

        auto &token_data = vocab.id_to_token[i];
        token_data.text = std::move(word);
        token_data.score = scores ? scores[i] : 0.0f;
        token_data.type = toktypes ? (llama_token_type)toktypes[i] : LLAMA_TOKEN_TYPE_NORMAL;
    }
    GGML_ASSERT(vocab.id_to_token.size() == vocab.token_to_id.size());

    // determine the newline token: LLaMA "<0x0A>" == 10 == '\n', Falcon 193 == '\n'
    if (vocab.type == Type::SPM) {
        try {
            linefeed_id = llama_byte_to_token(vocab, '\n');
        } catch (const std::exception &e) {
            spdlog::warn("%s: SPM vocabulary, but newline token not found: %s! Using special_pad_id instead.",
                __func__,
                e.what());
            linefeed_id = special_pad_id;
        }
    } else if (type == Type::WPM) {
        vocab.linefeed_id = vocab.special_pad_id;
    } else {
        const std::vector<int> ids = llama_tokenize_internal(vocab, "\xC4\x8A", false);// U+010A
        assert(!ids.empty() && "model vocab missing newline token");
        vocab.linefeed_id = ids[0];
    }

    // special tokens
    {
        const std::vector<std::pair<enum llm_kv, int32_t &>> special_token_types = {
            { LLM_KV_TOKENIZER_BOS_ID, vocab.special_bos_id },
            { LLM_KV_TOKENIZER_EOS_ID, vocab.special_eos_id },
            { LLM_KV_TOKENIZER_UNK_ID, vocab.special_unk_id },
            { LLM_KV_TOKENIZER_SEP_ID, vocab.special_sep_id },
            { LLM_KV_TOKENIZER_PAD_ID, vocab.special_pad_id },
            { LLM_KV_TOKENIZER_CLS_ID, vocab.special_cls_id },
            { LLM_KV_TOKENIZER_MASK_ID, vocab.special_mask_id },
            { LLM_KV_TOKENIZER_PREFIX_ID, vocab.special_prefix_id },
            { LLM_KV_TOKENIZER_SUFFIX_ID, vocab.special_suffix_id },
            { LLM_KV_TOKENIZER_MIDDLE_ID, vocab.special_middle_id },
            { LLM_KV_TOKENIZER_EOT_ID, vocab.special_eot_id },
        };
        for (const auto &it : special_token_types) {
            const std::string &key = kv(std::get<0>(it));
            int32_t &id = std::get<1>(it);

            uint32_t new_id;
            if (!ml.get_key(std::get<0>(it), new_id, false)) {
                continue;
            }
            if (new_id >= vocab.id_to_token.size()) {
                spdlog::warn(
                    "%s: bad special token: '%s' = %ud, using default id %d\n", __func__, key.c_str(), new_id, id);
            } else {
                id = new_id;
            }
        }

        // Handle add_bos_token and add_eos_token
        {
            bool temp = true;

            if (ml.get_key(LLM_KV_TOKENIZER_ADD_BOS, temp, false)) {
                vocab.special_add_bos = int(temp);
            }
            if (ml.get_key(LLM_KV_TOKENIZER_ADD_EOS, temp, false)) {
                vocab.special_add_eos = int(temp);
            }
        }
    }

    // build special tokens cache
    {
        // TODO: It is unclear (to me) at this point, whether special tokes are guaranteed to be of a deterministic
        // type,
        //  and will always be correctly labeled in 'added_tokens.json' etc.
        // The assumption is, since special tokens aren't meant to be exposed to end user, they are designed
        //  to be unmatchable by the tokenizer, therefore tokens from the vocab, which are unmatchable by the tokenizer
        //  are special tokens.
        // From testing, this appears to correlate 1:1 with special tokens.
        //

        // Counting special tokens and verifying in only one direction
        //  is sufficient to detect difference in those two sets.
        //
        uint32_t special_tokens_count_by_type = 0;
        uint32_t special_tokens_count_from_verification = 0;

        bool special_tokens_definition_mismatch = false;

        for (const auto &t : vocab.token_to_id) {
            const auto &token = t.first;
            const auto &id = t.second;

            // Count all non-normal tokens in the vocab while iterating
            if (vocab.id_to_token[id].type != LLAMA_TOKEN_TYPE_NORMAL) {
                special_tokens_count_by_type++;
            }

            // Skip single character tokens
            if (token.length() > 1) {
                bool is_tokenizable = false;

                // Split token string representation in two, in all possible ways
                //  and check if both halves can be matched to a valid token
                for (unsigned i = 1; i < token.length();) {
                    const auto left = token.substr(0, i);
                    const auto right = token.substr(i);

                    // check if we didnt partition in the middle of a utf sequence
                    auto utf = utf8_len(left.at(left.length() - 1));

                    if (utf == 1) {
                        if (vocab.token_to_id.find(left) != vocab.token_to_id.end()
                            && vocab.token_to_id.find(right) != vocab.token_to_id.end()) {
                            is_tokenizable = true;
                            break;
                        }
                        i++;
                    } else {
                        // skip over the rest of multibyte utf sequence
                        i += utf - 1;
                    }
                }

                if (!is_tokenizable) {
                    // Some tokens are multibyte, but they are utf sequences with equivalent text length of 1
                    //  it's faster to re-filter them here, since there are way less candidates now

                    // Calculate a total "utf" length of a token string representation
                    size_t utf8_str_len = 0;
                    for (unsigned i = 0; i < token.length();) {
                        utf8_str_len++;
                        i += utf8_len(token.at(i));
                    }

                    // And skip the ones which are one character
                    if (utf8_str_len > 1) {
                        // At this point what we have left are special tokens only
                        vocab.special_tokens_cache[token] = id;

                        // Count manually found special tokens
                        special_tokens_count_from_verification++;

                        // If this manually found special token is not marked as such, flag a mismatch
                        if (vocab.id_to_token[id].type == LLAMA_TOKEN_TYPE_NORMAL) {
                            special_tokens_definition_mismatch = true;
                        }
                    }
                }
            }
        }

        if (special_tokens_definition_mismatch
            || special_tokens_count_from_verification != special_tokens_count_by_type) {
            spdlog::warn("%s: mismatch in special tokens definition ( %u/%zu vs %u/%zu ).\n",
                __func__,
                special_tokens_count_from_verification,
                vocab.id_to_token.size(),
                special_tokens_count_by_type,
                vocab.id_to_token.size());
        } else {
            LLAMA_LOG_INFO("%s: special tokens definition check successful ( %u/%zu ).\n",
                __func__,
                special_tokens_count_from_verification,
                vocab.id_to_token.size());
        }
    }
}

M_END_NAMESPACE
