/**
 * The sentence-piece tokenization
 * 
 * Author: lihw81@gmail.com
*/
#ifndef M_TOKENIZER_SPM_H
#define M_TOKENIZER_SPM_H

#include "m_tokenizer.h"

#include <common/m_vocab.h>

#include <queue>

M_BEGIN_NAMESPACE

// SPM tokenizer
// adapted from FIXME: mllm and llama.cpp
// tried to simplify unicode stuff, so most likely does not work 100% correctly!

class TokenizerSpm : public Tokenizer {
    M_NO_COPY_CONSTRUCTOR(TokenizerSpm)
    M_NO_MOVE_CONSTRUCTOR(TokenizerSpm)

public:
    explicit TokenizerSpm() noexcept; 

    virtual ~TokenizerSpm() override;

    virtual int tokenize(const std::string& text, const Vocab& vocab, std::vector<TokenId>& out_tokens) noexcept final;

private:

    /**
     * Byte pair in BPE
    */
    struct Bigram {
        struct Comparator
        {
            bool operator()(Bigram& l, Bigram& r)
            {
                return (l.score < r.score) || (l.score == r.score && l.left > r.left);
            }
        };
        using QueueStorage = std::vector<Bigram>;
        using Queue = std::priority_queue<Bigram, QueueStorage, Comparator>;
        int left;
        int right;
        float score;
        size_t length;
    };

private:
  void resegment(const Tokenizer::Byte &byte,
                 const std::vector<Tokenizer::Byte> &bytes,
                 std::unordered_map<std::string, std::pair<int, int>>& revMerge,
                 const Vocab &vocab, std::vector<Vocab::Id> &output);
};


M_END_NAMESPACE


#endif // !M_TOKENIZER_SPM_H