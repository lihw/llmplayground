/** 
* The vocabulary
*
* Author: lihw81@gmail.com
*/

#include <common/m_vocab.h>

#include <common/m_model_loader.h>

#include <spdlog/spdlog.h>

#include <cassert>

M_BEGIN_NAMESPACE

bool Vocab::load(ModelLoader& ml, Model& model) noexcept
{
    gguf_context *ctx = ml.getContext();

    //const auto kv = LLM_KV(model.arch);

    // determine vocab type
    {
        std::string tokenizerName;

        ml.getKey(Kv::TOKENIZER_MODEL, tokenizerName);

        if (tokenizerName == "no_vocab") {
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
        } else if (tokenizerName == "llama") {
            type = Type::SPM;

            // default special tokens
            specialBosId = 1;
            specialEosId = 2;
            specialUnkId = 0;
            specialSepId = -1;
            specialPadId = -1;
            specialClsId = -1;
            specialMaskId = -1;

            const int addSpacePrefixKeyIdx = gguf_find_key(ctx, getKvString(Kv::TOKENIZER_ADD_PREFIX, ml.getArchName()).c_str());
            if (addSpacePrefixKeyIdx != -1) {
                addSpacePrefix = gguf_get_val_bool(ctx, addSpacePrefixKeyIdx);
            }// The default value of add_space_prefix is true.
        } else if (tokenizerName == "gpt2") {
            type = Type::BPE;

            // read bpe merges and populate bpe ranks
            const int mergesKeyIdx = gguf_find_key(ctx, Kv::TOKENIZER_MERGES.c_str());
            if (mergesKeyIdx == -1) {
                spdlog::error("cannot find tokenizer merges in model file\n");
                return false;
            }

            const int nMerges = gguf_get_arr_n(ctx, mergesKeyIdx);

            for (int i = 0; i < nMerges; i++) {
                const std::string word = gguf_get_arr_str(ctx, mergesKeyIdx, i);
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
        } else if (tokenizerName == "bert") {
            type = Type::WPM;

            // default special tokens
            specialBosId = -1;
            specialEosId = -1;
            specialUnkId = 100;
            specialSepId = 102;
            specialPadId = 0;
            specialClsId = 101;
            specialMaskId = 103;
            addSpacePrefix = false;
        } else {
            spdlog::warn("%s: unknown tokenizer: '%s'", __func__, tokenizerName.c_str());
            spdlog::warn("%s: using default tokenizer: 'llama'", __func__);

            type = Type::SPM
        }
    }

    const int tokenIdx = gguf_find_key(ctx, getKvString(Kv::TOKENIZER_LIST, ml.getArchName()).c_str());
    if (tokenIdx == -1) {
        spdlog::error("cannot find tokenizer vocab in model file");
        return false;
    }

    const float *scores = nullptr;
    const int scoreIdx = gguf_find_key(ctx, getKvString(Kv::TOKENIZER_SCORES, ml.getArchName()).c_str());
    if (scoreIdx != -1) {
        scores = (const float *)gguf_get_arr_data(ctx, scoreIdx);
    }

    const int *toktypes = nullptr;
    const int toktypeIdx = gguf_find_key(ctx, getKvString(Kv::TOKENIZER_TOKEN_TYPE, ml.getArchName()).c_str());
    if (toktypeIdx != -1) {
        toktypes = (const int *)gguf_get_arr_data(ctx, toktypeIdx);
    }

    const uint32_t numVocab = gguf_get_arr_n(ctx, tokenIdx);
    idToToken.resize(numVocab);
    for (uint32_t i = 0; i < numVocab; i++) {
        std::string word = gguf_get_arr_str(ctx, tokenIdx, i);
        assert(unicode_cpts_from_utf8(word).size() > 0);

        tokenToId[word] = i;

        auto& tokenData = idToToken[i];
        tokenData.text = std::move(word);
        tokenData.score = scores ? scores[i] : 0.0f;
        tokenData.type = toktypes ? (TokenType)toktypes[i] : TokenType::NORMAL;
    }
    assert(idToToken.size() == tokenToId.size());

    // determine the newline token: LLaMA "<0x0A>" == 10 == '\n', Falcon 193 == '\n'
    if (type == Type::SPM) {
        try {
            lineFeedId = convertByteToToken('\n');
        } catch (const std::exception &e) {
            spdlog::warn("%s: SPM vocabulary, but newline token not found: %s! Using special_pad_id instead.",
                __func__, e.what());
            lineFeedId = specialPadId;
        }
    } else if (type == Type::WPM) {
        lineFeedId = specialPadId;
    } else {
        const std::vector<int> ids = llama_tokenize_internal("\xC4\x8A", false);// U+010A
        assert(!ids.empty() && "model vocab missing newline token");
        lineFeedId = ids[0];
    }

    // special tokens
    {
        const std::vector<std::pair<Kv, int32_t &>> specialTokenTypes = {
            { Kv::TOKENIZER_BOS_ID, specialBosId },
            { Kv::TOKENIZER_EOS_ID, specialEosId },
            { Kv::TOKENIZER_UNK_ID, specialUnkId },
            { Kv::TOKENIZER_SEP_ID, specialSepId },
            { Kv::TOKENIZER_PAD_ID, specialPadId },
            { Kv::TOKENIZER_CLS_ID, specialClsId },
            { Kv::TOKENIZER_MASK_ID, specialMaskId },
            { Kv::TOKENIZER_PREFIX_ID, specialPrefixId },
            { Kv::TOKENIZER_SUFFIX_ID, specialSuffixId },
            { Kv::TOKENIZER_MIDDLE_ID, specialMiddleId },
            { Kv::TOKENIZER_EOT_ID, specialEotId },
        };
        for (const auto &it : specialTokenTypes) {
            const std::string &key = getKvString(std::get<0>(it), ml.getArchName());
            int32_t &id = std::get<1>(it);

            uint32_t newId;
            if (!ml.getKey(std::get<0>(it), newId, false)) {
                continue;
            }
            if (newId >= idToToken.size()) {
                spdlog::warn(
                    "%s: bad special token: '%s' = %ud, using default id %d\n", __func__, key.c_str(), newId, id);
            } else {
                id = newId;
            }
        }

        // Handle add_bos_token and add_eos_token
        {
            bool temp = true;

            if (ml.getKey(Kv::TOKENIZER_ADD_BOS, temp, false)) {
                specialAddBos = int(temp);
            }
            if (ml.getKey(Kv::TOKENIZER_ADD_EOS, temp, false)) {
                specialAddEos = int(temp);
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
        uint32_t specialTokensCountByType = 0;
        uint32_t specialTokensCountFromVerification = 0;

        bool specialTokensDefinitionMismatch = false;

        for (const auto &[token, id] : tokenToId) {
            // Count all non-normal tokens in the vocab while iterating
            if (idToToken[id].type != TokenType::NORMAL) {
                specialTokensCountByType++;
            }

            // Skip single character tokens
            if (token.length() > 1) {
                bool isTokenizable = false;

                // Split token string representation in two, in all possible ways
                //  and check if both halves can be matched to a valid token
                for (unsigned i = 1; i < token.length();) {
                    const auto left = token.substr(0, i);
                    const auto right = token.substr(i);

                    // check if we didnt partition in the middle of a utf sequence
                    auto utf = utf8Len(left.at(left.length() - 1));

                    // If the token can be split into two short ones.
                    if (utf == 1) {
                        if (tokenToId.find(left) != tokenToId.end()
                            && tokenToId.find(right) != tokenToId.end()) {
                            isTokenizable = true;
                            break;
                        }
                        i++;
                    } else {
                        // skip over the rest of multibyte utf sequence
                        i += utf - 1;
                    }
                }

                if (!isTokenizable) {
                    // Some tokens are multibyte, but they are utf sequences with equivalent text length of 1
                    //  it's faster to re-filter them here, since there are way less candidates now

                    // Calculate a total "utf" length of a token string representation
                    size_t utf8StrLen = 0; // The number of utf characters
                    for (unsigned i = 0; i < token.length();) {
                        utf8StrLen++;
                        i += utf8Len(token.at(i));
                    }

                    // And skip the ones which are one character
                    if (utf8StrLen > 1) {
                        // At this point what we have left are special tokens only
                        specialTokensCache[token] = id;

                        // Count manually found special tokens
                        specialTokensCountFromVerification++;

                        // If this manually found special token is not marked as such, flag a mismatch
                        if (idToToken[id].type == TokenType::NORMAL) {
                            specialTokensDefinitionMismatch = true;
                        }
                    }
                }
            }
        }

        if (specialTokensDefinitionMismatch
            || specialTokensCountFromVerification != specialTokensCountByType) {
            spdlog::warn("%s: mismatch in special tokens definition ( %u/%zu vs %u/%zu ).\n",
                __func__,
                specialTokensCountFromVerification, 
                idToToken.size(),
                specialTokensCountByType,
                idToToken.size());
        } else {
            spdlog::info("%s: special tokens definition check successful ( %u/%zu ).\n",
                __func__,
                specialTokensCountFromVerification, 
                idToToken.size());
        }
    }
}

M_END_NAMESPACE
