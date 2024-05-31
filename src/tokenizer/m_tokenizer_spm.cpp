/**
 * The sentence-piece tokenization
 * 
 * Author: lihw81@gmail.com
*/

#include "m_tokenizer_spm.h"

#include <vector>
#include <string>

M_BEGIN_NAMESPACE

TokenizerSpm::TokenizerSpm() noexcept
{
}

TokenizerSpm::~TokenizerSpm() 
{
}

int TokenizerSpm::tokenize(const std::string& text, const Vocab& vocab, std::vector<TokenId>& out_tokens) noexcept
{
    // Split the entire sentence into tokens instead of tokenizing
    // words (pre-tokenization) as in BPE.
    std::vector<Byte> bytes;
    size_t offset = 0;
    size_t index = 0;
    while (offset < text.size()) {
        Byte b;
        size_t charLen = std::min(text.size() - offset, size_t(utf8Len(text[offset])));
        b.text = text.c_str() + offset;
        b.length = charLen;
        offset += b.length;
        b.prev = int(index) - 1;
        b.next = offset == text.size() ? -1 : int(index) + 1;
        index++;
        bytes.emplace_back(b);
    }

    Bigram::Queue workQueue;

    // The bigram -> its original position in the text.
    std::unordered_map<std::string, std::pair<int, int>> revMerge;

    auto addNewBigram = [&vocab, &bytes, &workQueue, &revMerge](int left, int right) {
        if (left == -1 || right == -1) {
            return;
        }

        const std::string text = std::string(bytes[left].text, size_t(bytes[left].length + bytes[right].length));
        auto token = vocab.tokenToId.find(text);

        if (token == vocab.tokenToId.end()) {
            return;
        }

        if (static_cast<size_t>((*token).second) >= vocab.idToToken.size()) {
            return;
        }

        const auto &tokData = vocab.idToToken[(*token).second];

        Bigram bigram;
        bigram.left = left;
        bigram.right = right;
        bigram.score = tokData.score;
        bigram.length = text.size();

        workQueue.push(bigram);

        // Do we need to support "not used"
        revMerge[text] = std::make_pair(left, right);
    };

    // seed the work queue with all possible 2-character tokens.
    for (int i = 1; i < (int)bytes.size(); i++) {
        addNewBigram(i - 1, i);
    }

    // keep substituting the highest frequency pairs for as long as we can.
    // It is a greedy algorithm that keep the longest bigram/token around

    // At first
    //   o   o   o   o   o    <- bigram
    //  / \ / \ / \ / \ / \ ...  
    // a   b   c   d   e   f
    //
    // When (b, c) is picked,
    //   o   o   o   o    <- bigram
    //  / \ / \ / \ / \  ...
    // a   bc  d   e   f
    // A new bigram (bc, d) is added. The old bigram (a, b) is updated to (a, bc)
    // The old bigram (c, d) since c.length = 0, it becomes invalid and won't be processed. 

    while (!workQueue.empty()) {
        auto bigram = workQueue.top();
        workQueue.pop();

        auto &leftByte = bytes[bigram.left];
        auto &rightByte = bytes[bigram.right];

        // if one of the symbols already got merged, skip it.
        if (leftByte.length == 0 || rightByte.length == 0 || leftByte.length + rightByte.length != bigram.length) {
            continue;
        }

        // merge the right sym into the left one
        leftByte.length += rightByte.length;
        rightByte.length = 0;

        // LLAMA_LOG_INFO("left = '%*s' size = %zu\n", (int) left_sym.n, left_sym.text, bigram.size);

        // remove the right sym from the chain
        leftByte.next = rightByte.next;
        if (rightByte.next >= 0) {
            bytes[rightByte.next].prev = bigram.left;
        }

        // find more substitutions
        addNewBigram(leftByte.prev, bigram.left);
        addNewBigram(bigram.left, rightByte.next);
    }

    for (int i = 0; i != -1; i = bytes[i].next) {
        auto &byte = bytes[i];
        resegment(byte, bytes, revMerge, vocab, out_tokens);
    }

    return int(out_tokens.size());
}

void TokenizerSpm::resegment(const Tokenizer::Byte &byte,
    const std::vector<Tokenizer::Byte> &bytes,
    std::unordered_map<std::string, std::pair<int, int>>& revMerge,
    const Vocab &vocab,
    std::vector<Vocab::Id> &output)
{
    auto text = std::string(byte.text, byte.length);
    auto token = vocab.tokenToId.find(text);

    if (token != vocab.tokenToId.end()) {
        output.push_back((*token).second);
        return;
    }
    
    // If the token is not found in vocab.

    const auto p = revMerge.find(text);

    if (p == revMerge.end()) {
        // output any symbols that did not form tokens as bytes.
        output.reserve(output.size() + byte.length);
        for (int j = 0; j < (int)byte.length; ++j) {
            Vocab::Id tokenId = vocab.byteToToken(byte.text[j]);
            output.push_back(tokenId);
        }
        return;
    }

    assert(!"There can't be any bigram that doesn't exist in the vocab!");

    resegment(bytes[p->second.first], bytes, revMerge, vocab, output);
    resegment(bytes[p->second.second], bytes, revMerge, vocab, output);
}

M_END_NAMESPACE
