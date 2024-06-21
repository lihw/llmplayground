/** 
* The batch
*
* Author: lihw81@gmail.com
*/

#include "m_batch.h"

#include <cassert>

M_BEGIN_NAMESPACE

Batch::Batch() 
    : numTokens{0}
{
}
    
Batch::~Batch() noexcept
{
}

void Batch::add(TokenId tokenId, Pos pos1, const std::vector<SeqId>& seqIds1, bool logits1) noexcept
{
    tokens[numTokens] = tokenId;
    pos[numTokens]    = pos1;
    seqIds[numTokens]  = seqIds1;
    logits[numTokens] = logits1;

    numTokens++;
}

Batch* createBatch(int32_t numTokens, int32_t embedLength, int32_t maxNumSeq)
{
    Batch* batch = new Batch;
    // { 0, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, 0, 0, 0, };

    if (embedLength) {
        batch->embeds.resize(numTokens * embedLength);
    } else {
        batch->tokens.resize(numTokens);
    }

    batch->pos.resize(numTokens);

    batch->seqIds.resize(numTokens);  // FIXME: allocation has one extra slot.
    for (auto& seqId : batch->seqIds) {
        seqId.reserve(maxNumSeq);
    }
    /*
    batch.n_seq_id = (int32_t *)       malloc(sizeof(int32_t)        * n_tokens_alloc);
    batch.seq_id   = (llama_seq_id **) malloc(sizeof(llama_seq_id *) * (n_tokens_alloc + 1));
    for (int i = 0; i < n_tokens_alloc; ++i) {
        batch.seq_id[i] = (llama_seq_id *) malloc(sizeof(llama_seq_id) * n_seq_max);
    }
    batch.seq_id[n_tokens_alloc] = nullptr;
    */
    batch->logits.resize(numTokens);

    return batch;
}
    

#if 0

Batch::Batch(int32_t numTokens, int32_t numEmbeds, int32_t maxSeq) 
{
    assert(numTokens > 0);
        
    if (numEmbedings > 0) {
        embeds.resize(numEmbeds);
    } else {
        tokens.resize(numTokens);
    } 

    pos.resize(numTokens);
    seqIds.resize(numTokens + 1);
    for (size_t i = 0; i < seqIds.size() - 1; i++) {
        seqIds[i].resize(maxSeq);
    }

    logits.resize(numTokens);

    return batch;
}

void Batch::add(TokenId tokenId, PosEncoding pos, const std::vector<SeqId>& seqIds, 
        bool logits) noexcept
{
    tokens.push_back(tokenId);
    pos.push_back(pos);
    batch.seqId[tokens.size() - 1] = seqIds;
    logits.push_back(logits);
}

#endif
    
M_END_NAMESPACE
