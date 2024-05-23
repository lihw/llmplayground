/**
 * The byte pair encoding tokenizer
 *
 * Author: lihw81@gmail.com
 */

#include "m_tokenizer_bpe.h"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <stdexcept>
#include <iterator>

M_BEGIN_NAMESPACE

int TokenizerBpe::tokenize(const std::string &text, const Vocab &vocab, std::vector<TokenId> &out_tokens) noexcept
{
    // FIXME:
    // Currently we don't have any specials
    std::vector<std::string> specials = {};
    std::vector<std::string> words = pretokenize(text, specials);

    // Create tokens from each word
    out_tokens.clear();

    for (auto &word : words) { 
        auto bytes = tokenizeInternal(word, vocab); 

        // Tokenize the bytes
        for (auto& b : bytes) {
            const std::string str = std::string(b.text, b.length);
            const auto token = vocab.tokenToId.find(str);

            // Roll back to multi-char tokens if not found.
            if (token == vocab.tokenToId.end()) {
                for (auto j = str.begin(); j != str.end(); ++j) {
                    std::string byteStr(1, *j);
                    auto tokenMultibyte = vocab.tokenToId.find(byteStr);
                    if (tokenMultibyte == vocab.tokenToId.end()) {
                        spdlog::error("ERROR: byte not found in vocab");
                        return -1;
                    }
                    out_tokens.push_back((*tokenMultibyte).second);
                }
            } else {
                out_tokens.push_back((*token).second);
            }
        }
    }

    // FIXME: The utf8 handling?

    // FIXME: the end of word?
    return (int)out_tokens.size();
}

std::vector<TokenizerBpe::Byte> TokenizerBpe::tokenizeInternal(const std::string &word, const Vocab& vocab) noexcept
{
    std::vector<Byte> bytes;

    Bigram::Queue tasks;

    auto addNewBigram = [&tasks, &bytes, &vocab](int left, int right) {
        if (left == -1 || right == -1) { return; }

        std::string leftSymbol = std::string(bytes[left].text, bytes[left].length);
        std::string rightSymbol = std::string(bytes[right].text, bytes[right].length);

        int rank = vocab.findBpeRank(leftSymbol, rightSymbol);

        if (rank < 0) { return; }

        Bigram bigram;

        bigram.left = left;
        bigram.right = right;
        bigram.text = leftSymbol + rightSymbol;
        bigram.length = leftSymbol.size() + rightSymbol.size();
        bigram.rank = rank;

        tasks.push(bigram);
    };

    int index = 0;
    size_t offset = 0;

    // Create BPE bytes from each word
    while (offset < word.size()) {
        Byte b;
        size_t charLen = std::min(word.size() - offset, (size_t)utf8_len(word[offset]));
        b.text = word.c_str() + offset;
        b.length = charLen;
        offset += b.length;
        b.prev = index - 1;
        b.next = offset == word.size() ? -1 : index + 1;
        index++;
        bytes.emplace_back(b);
    }

    for (int i = 1; i < (int)bytes.size(); i++) { addNewBigram(i - 1, i); }

    // Merge bytes into longest ones.
    while (!tasks.empty()) {
        auto bigram = tasks.top();
        tasks.pop();

        auto &leftByte = bytes[bigram.left];// The left byte
        auto &rightByte = bytes[bigram.right];// The right byte

        if (leftByte.length == 0 || rightByte.length == 0) { continue; }

        std::string leftToken = std::string(leftByte.text, leftByte.length);// The left token
        std::string rightToken = std::string(rightByte.text, rightByte.length);// Ditto

        if (leftToken + rightToken != bigram.text) {
            continue;// Skip the outdated bigram
        }

        // Remove the right byte from the sequence.
        leftByte.next = rightByte.next;
        if (rightByte.next >= 0) { bytes[rightByte.next].prev = bigram.left; }

        addNewBigram(leftByte.prev, bigram.left);// left + bigram
        addNewBigram(bigram.left, rightByte.next);// bigram + right
    }

    // Add the finished tokens to the final list keeping correct order for next and prev
    std::vector<Byte> out;
    std::copy_if(bytes.begin(), bytes.end(), std::back_inserter(out), [](const Byte &b) { return b.length > 0; });

    return out;
}

M_END_NAMESPACE
