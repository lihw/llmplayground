/** 
* The abstract interface of tokenizer class
*
* Author: lihw81@gmail.com
*/

#ifndef M_TOKENIZER_H
#define M_TOKENIZER_H

#include <common/m_defs.h>
#include <common/m_token.h>
#include <common/m_vocab.h>

#include <vector>
#include <string>

M_BEGIN_NAMESPACE

class Tokenizer {
    M_NO_COPY_CONSTRUCTOR(Tokenizer)
    M_NO_MOVE_CONSTRUCTOR(Tokenizer)

public:
    explicit Tokenizer() {};

    virtual ~Tokenizer() {};

    /**
     * Tokenize the input text into a list of tokens with given Vocab
     */
    virtual int tokenize(const std::string& text, 
            const Vocab& vocab, 
            std::vector<TokenId>& out_tokens) noexcept = 0;

protected:
    /**
     * Byte in BPE 
    */
    struct Byte {
        using index = int;
        index prev;
        index next;
        const char *text;
        size_t length;
    };

    /**
     * Split the text into words
     */
    std::vector<std::string> pretokenize(const std::string& text, const std::vector<std::string>& specials) noexcept;

    static inline size_t utf8Len(char src) noexcept {
        const size_t lookup[] = { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 4 };
        uint8_t highbits = static_cast<uint8_t>(src) >> 4;
        return lookup[highbits];
    }
};

extern int tokenize(const std::string& text, 
            const Vocab& vocab, 
            bool addSpecial,
            bool parseSpecial,
            std::vector<TokenId>& out_tokens) noexcept;

extern void detokenize(const std::vector<TokenId>& tokens,
            const Vocab& vocab,
            bool addSpecial,
            std::string& out_text) noexcept;

M_END_NAMESPACE

#endif // !M_TOKENIZER_H
