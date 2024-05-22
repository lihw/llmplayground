/** 
* The abstract interface of tokenizer class
*
* Author: lihw81@gmail.com
*/

#ifndef M_TOKENIZER_H
#define M_TOKENIZER_H

#include <common/m_defs.h>
#include <common/m_token.h>

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
    virtual int tokenize(const std::string& text, const Vocab& vocab, std::vector<Token>& out_tokens) noexcept = 0;

protected:
    /**
     * Split the text into words
     */
    std::vector<std::string> pretokenize(const std::string& text, const std::vector<std::string>& specials) noexcept;
};

M_END_NAMESPACE

#endif // !M_TOKENIZER_H
