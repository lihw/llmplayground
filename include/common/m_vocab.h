/** 
* The vocabulary
*
* Author: lihw81@gmail.com
*/

#ifndef M_VOCAB_H
#define M_VOCAB_H

#include <common/m_defs.h>
#include <common/m_token.h>

#include <string>
#include <unordered_map>
#include <vector>
#include <map>

#include <cassert>

M_BEGIN_NAMESPACE

class Model;
class ModelLoader;

struct Vocab {
    using Id    = int32_t;
    using Token = std::string;
    using TType = TokenType;

    enum class Type {
        NONE = 0, // For models without vocab
        SPM  = 1, // LLaMA tokenizer based on byte-level BPE with byte fallback
        BPE  = 2, // GPT-2 tokenizer based on byte-level BPE
        WPM  = 3, // BERT tokenizer based on WordPiece
    };

    Type type = Type::SPM;

    std::unordered_map<Token, Id> tokenToId;
    std::vector<TokenData>        idToToken;

    std::unordered_map<Token, Id> specialTokensCache;

    // FIXME: bpe ranks should be special to BPE vocab only.
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

    bool addSpacePrefix = true;

    Type getType() const {
        return type;
    }

    // Ditto
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

    bool load(ModelLoader &ml) noexcept;

    TokenId byteToToken(uint8_t ch) const noexcept;
};

M_END_NAMESPACE

#endif //!M_VOCAB_H
