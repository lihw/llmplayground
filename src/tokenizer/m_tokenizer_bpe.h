/** 
* The byte pair encoding tokenizer
*
* Author: lihw81@gmail.com
*/

#include "m_tokenizer.h"

#include <common/m_token.h>

M_BEGIN_NAMESPACE

class TokenizerBpe : public Tokenizer {
    M_NO_COPY_CONSTRUCTOR(TokenizerBpe)
    M_NO_MOVE_CONSTRUCTOR(TokenizerBpe)

public:
    explicit TokenizerBpe() noexcept; 

    virtual ~TokenizerBpe() override;

    virtual int tokenize(const std::string& text, const Vocab& vocab, std::vector<Token>& out_tokens) noexcept final;

private:
    std::vector<std::string> tokenize(const std::string& word) noexcept;
};

M_END_NAMESPACE

