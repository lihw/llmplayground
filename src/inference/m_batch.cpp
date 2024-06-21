/** 
* The batch
*
* Author: lihw81@gmail.com
*/

#include "m_batch.h"

#include <cassert>

M_BEGIN_NAMESPACE

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
    
}

M_END_NAMESPACE
