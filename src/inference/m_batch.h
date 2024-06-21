/** 
* The batch
*
* Author: lihw81@gmail.com
*/

#ifndef M_BATCH_H
#define M_BATCH_H

#include <common/m_defs.h>
#include <common/m_token.h>

M_BEGIN_NAMESPACE

typedef int32_t Pos;
typedef int32_t SeqId;

// Input data for decoding
// A Batch object can contain input about one or many sequences
// The provided arrays (i.e. token, embd, pos, etc.) must have size of n_tokens
//
struct Batch {
    //! the token ids of the input (used when embd is NULL)
    std::vector<Token>                       tokens;
    //! token embeddings (i.e. float vector of size n_embd) (used when token is NULL)
    std::vector<float>                       embeds;
    //! the positions of the respective token in the sequence
    std::vector<Pos>                         pos;
    std::vector<std::vector<SeqId>>          seqIds;
    //! if zero, the logits (and/or the embeddings) for the respective token will not be output
    std::vector<int8_t>                      logits; // TODO: rename this to "output"

    // NOTE: helpers for smooth API transition - can be deprecated in the future
    //       for future-proof code, use the above fields instead and ignore everything below
    //
    // pos[i] = all_pos_0 + i*all_pos_1
    //
    //llama_pos    all_pos_0;  // used if pos == NULL
    //llama_pos    all_pos_1;  // used if pos == NULL
    //llama_seq_id all_seq_id; // used if seq_id == NULL
    //

    void add(TokenId tokenId, PosEncoding pos, const std::vector<SeqId>& seqIds, 
        bool logits) noexcept;
};

M_END_NAMESPACE

#endif // !M_BATCH_H

