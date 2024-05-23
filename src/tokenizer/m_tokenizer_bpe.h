/** 
* The byte pair encoding tokenizer
*
* Author: lihw81@gmail.com
*/

#include "m_tokenizer.h"

#include <common/m_token.h>

#include <queue>

M_BEGIN_NAMESPACE

// BPE tokenizer
// adapted from FIXME: mllm and llama.cpp
// tried to simplify unicode stuff, so most likely does not work 100% correctly!

class TokenizerBpe : public Tokenizer {
    M_NO_COPY_CONSTRUCTOR(TokenizerBpe)
    M_NO_MOVE_CONSTRUCTOR(TokenizerBpe)

public:
    explicit TokenizerBpe() noexcept; 

    virtual ~TokenizerBpe() override;

    virtual int tokenize(const std::string& text, const Vocab& vocab, std::vector<Token>& out_tokens) noexcept final;

private:
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
     * Byte pair in BPE
    */
    struct Bigram {
        struct Comparator {
            bool operator()(const Bigram& l, const Bigram& r) const {
                return l.rank > r.rank || (l.rank == r.rank && l.left > r.left);
            }
        };

        using QueueStorage = std::vector<Bigram>;
        using Queue = std::priority_queue<Bigram, QueueStorage, Comparator>;
        Byte::index left;
        Byte::index right;
        std::string text;
        int rank;
        size_t length;
    };

private:
    std::vector<Byte> tokenize(const std::string& word, const Vocab& vocab) noexcept;
};

M_END_NAMESPACE

