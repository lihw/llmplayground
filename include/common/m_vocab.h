/** 
* The vocabulary
*
* Author: lihw81@gmail.com
*/

#ifndef M_VOCAB_H
#define M_VOCAB_H

#include <common/m_defs.h>

#include <string>
#include <unordered_map>
#include <vector>
#include <map>

#include <cassert>

M_BEGIN_NAMESPACE

enum class VocabType {
    NONE = 0, // For models without vocab
    SPM  = 1, // LLaMA tokenizer based on byte-level BPE with byte fallback
    BPE  = 2, // GPT-2 tokenizer based on byte-level BPE
    WPM  = 3, // BERT tokenizer based on WordPiece
};

struct Vocab {
    using Id    = int32_t;
    using Token = std::string;
    //using TType  = llama_token_type;

    //enum llama_vocab_type type = LLAMA_VOCAB_TYPE_SPM;

    std::unordered_map<Token, Id> tokenToId;
    std::vector<TokenData>       idToToken;

    std::unordered_map<Token, Id> specialTokensCache;

    std::map<std::pair<std::string, std::string>, int> bpeRanks;

    // Default LLaMA special tokens
    Id specialBosId  = 1;
    Id specialEosId  = 2;
    Id specialUnkId  = 0;
    Id specialSepId  = -1;
    Id specialPadId  = -1;
    Id specialClsId  = -1;
    Id specialMaskId = -1;

    Id specialAddBos = -1; // -1 unknown, 1 add, 0 don't add.
    Id specialAddEos = -1; // -1 unknown, 1 add, 0 don't add.

    Id lineFeedId       = 13;
    Id specialPrefixId = -1;
    Id specialSuffixId = -1;
    Id specialMiddleId = -1;
    Id specialEotId    = -1;

    bool add_space_prefix = true;

    int findBpeRank(const std::string & leftBytes, const std::string& rightBytes) const {
        assert(leftBytes.find(' ') == std::string::npos); // Both left and right bytes shouldn't contain spaces and newlines
        assert(leftBytes.find('\n') == std::string::npos);
        assert(rightBytes.find(' ') == std::string::npos);
        assert(rightBytes.find('\n') == std::string::npos);

        auto it = bpeRanks.find(std::make_pair(leftBytes, rightBytes));
        if (it == bpeRanks.end()) {
            return -1;
        }

        return it->second;
    }
};

M_END_NAMESPACE

#endif //!M_VOCAB_H
