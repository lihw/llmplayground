/** 
* The abstract interface of tokenizer class
*
* Author: lihw81@gmail.com
*/

#include "m_tokenizer.h"

#include <common/m_vocab.h>
#include <common/unicode.h>
#include <common/m_misc.h>
#include "m_tokenizer_bpe.h"
#include "m_tokenizer_spm.h"

#include <spdlog/spdlog.h>
#include <fmt/core.h>

#include <regex>

#include <cassert>

M_BEGIN_NAMESPACE

int tokenize(const std::string& text, 
            const Vocab& vocab, 
            bool addSpecial,
            bool parseSpecial,
            std::vector<TokenId>& out_tokens) noexcept
{
    const auto LOG_HEAD = "tokenize()";

    if (text.empty()) {
        spdlog::warn("{}: empty input text", LOG_HEAD);
        return 0;
    }

    size_t numTokens = text.length() + 2 * (addSpecial? 1 : 0);
    out_tokens.reserve(numTokens);
        
    if (vocab.type != Vocab::Type::BPE && vocab.type != Vocab::Type::SPM) {
        spdlog::error("{}: unsupported tokenizer {}", LOG_HEAD, int(vocab.type));
        return -1;
    }
   
    auto findSpecial = [](const std::string& text, const std::unordered_map<Token, TokenId>& specials, size_t pos) noexcept -> std::pair<size_t, TokenId> {
        for (const auto& [delimiter, id]: specials) {
            size_t found = text.find(delimiter, pos);
            if((found != std::string::npos) ) {
                return {found + delimiter.length(), id};
            }
        }
        return {std::string::npos, -1};
    };
    
    auto splitWords = [](const std::string& text) noexcept {
        std::vector<std::string> words;
        std::regex pattern("<\\|startoftext\\|>|<\\|endoftext\\|>|'s|'t|'re|'ve|'m|'ll|'d|\\w+|\\d+|\\S+");
        std::smatch match;

        std::string::const_iterator searchStart(text.cbegin());
        while (std::regex_search(searchStart, text.cend(), match, pattern)) {
            words.push_back(match.str());
            searchStart = match.suffix().first;
        }
        
        return words;
    };
        
    if (vocab.type == Vocab::Type::SPM) {
        if (addSpecial && vocab.specialAddBos != 0) {
            assert(vocab.specialBosId != -1);
            out_tokens.push_back(vocab.specialBosId);
        }
    }
    else if (vocab.type == Vocab::Type::BPE) {
        if (addSpecial && vocab.specialAddBos == 1) {
            assert(vocab.specialBosId != -1);
            out_tokens.push_back(vocab.specialBosId);
        }
    }
    
    size_t start = 0;

    auto tokenizeInternal = [&](std::string& fragment) -> void {
        if (vocab.type == Vocab::Type::SPM) {
            // without adding this leading whitespace, we do not get the same results as the original SPM tokenizer
            if (start == 0 && addSpecial) {
                fragment = " " + fragment;
            }
            escapeWhitespace(fragment);
            TokenizerSpm spm;
            spm.tokenize(fragment, vocab, out_tokens);
        }
        else if (vocab.type == Vocab::Type::BPE) {
            TokenizerBpe bpe;
            auto words = splitWords(fragment);
            for (auto& w : words) {
                bpe.tokenize(w, vocab, out_tokens);
            }
        }
        else {
            spdlog::error("{}: unsupported tokenizer type {} for tokenizing the text chunks", LOG_HEAD, int(vocab.type));
        }
    };

    // Split the text into chunks separated by the special tokens
    if (parseSpecial) {
        auto result = findSpecial(text, vocab.specialTokensCache, start);
        auto found = result.first;

        while (found != std::string::npos) {
            if (found > start) {
                auto fragment = text.substr(start, found - start);
                tokenizeInternal(fragment);
            }

            // Take the special dlimeter as a token too.
            out_tokens.push_back(result.second);

            start = result.first;
            result = findSpecial(text, vocab.specialTokensCache, start);
            found = result.first;
        }

        // Process the remaining words
        auto fragment = text.substr(start);
        tokenizeInternal(fragment);
    } else {
        std::string fragment = text;
        tokenizeInternal(fragment);
    }
        
    if (vocab.type == Vocab::Type::SPM) {
        if (addSpecial && vocab.specialAddEos == 1) {
            assert(vocab.specialEosId != -1);
            out_tokens.push_back(vocab.specialEosId);
        }
    } else if (vocab.type == Vocab::Type::BPE) {
        assert(vocab.specialAddEos != -1);
    } else {
        spdlog::error("%s: unsupported tokenizer type %d for dealing the tailing token", __func__, int(vocab.type));
        return 0;
    }
    
    if (spdlog::get_level() <= spdlog::level::level_enum::info) {
        spdlog::info("The output tokens:");
        for (auto& t : out_tokens) {
            auto tok = fmt::format("('{}', {}), ", vocab.idToToken[t].text, t);
            fprintf(stdout, tok.c_str());
        }
        fprintf(stdout, "\n\n");
    }

    return int(out_tokens.size());
}

M_END_NAMESPACE

