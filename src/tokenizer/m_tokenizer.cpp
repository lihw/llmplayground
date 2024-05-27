/** 
* The abstract interface of tokenizer class
*
* Author: lihw81@gmail.com
*/

#include "m_tokenizer.h"

#include <common/m_vocab.h>
#include <common/unicode.h>

#include <regex>

#include <cassert>

M_BEGIN_NAMESPACE

std::vector<std::string> Tokenizer::pretokenize(const std::string& text, const std::vector<std::string>& specials) noexcept {

    size_t start = 0;

    auto findSpecial = [](const std::string& text, const std::vector<std::string>& specials, size_t pos) noexcept -> std::pair<size_t, std::string> {
        for (const std::string &delimiter : specials) {
            size_t found = text.find(delimiter, pos);
            if((found != std::string::npos) ) {
                return {found, delimiter};
            }
        }
        return {std::string::npos, ""};
    };

    auto splitWords = [](const std::string& text) noexcept {
        std::vector<std::string> words;
        std::regex pattern("<\\|startoftext\\|>|<\\|endoftext\\|>|'s|'t|'re|'ve|'m|'ll|'d|\\w+|\\d+|\\S+");
        std::smatch match;

        std::string::const_iterator searchStart(text.cbegin());
        while (std::regex_search(searchStart, text.cend(), match, pattern)) {
            words.push_back(match.str());
            searchStart = match.suffix().first;
        }
        
        return words;
    };
    
    // Split the text into chunks seperated with given special delimeters.
    // And then split the chunk with gpt2-defined spaces
    // GPT2 system regex:  's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
    std::vector<std::string> out;

    auto result = findSpecial(text, specials, start);
    auto found = result.first;
    while (found != std::string::npos) {
        if (found > start) {
            const std::vector<std::string>& words = splitWords(text.substr(start, found - start));
            out.insert(out.end(), words.begin(), words.end());
        }

        // Take the special delimeter as a token too.
        if (result.second == "\n") {
            out.push_back("<0x0A>");
        } else {
            out.push_back(result.second);
        }

        start += result.second.length();
        result = findSpecial(text, specials, start);
        found = result.first;
    }

    return out;
}

TokenId Tokenizer::byteToToken(const Vocab& vocab, uint8_t ch) {
    assert(vocab.getType() != Vocab::Type::NONE);
    static const char * hex = "0123456789ABCDEF";
    switch (vocab.getType()) {
        case Vocab::Type::SPM: {
            const char buf[7] = { '<', '0', 'x', hex[ch >> 4], hex[ch & 15], '>', 0 };
            auto token = vocab.tokenToId.find(buf);
            if (token != vocab.tokenToId.end()) {
                return (*token).second;
            }
            // Try to fall back to just the byte as a string
            const char buf2[2] = { (char)ch, 0 };
            return vocab.tokenToId.at(buf2);
        }
        case Vocab::Type::WPM: 
        case Vocab::Type::BPE: 
            return vocab.tokenToId.at(unicode_byte_to_utf8(ch));
        default:
            assert(false);
    }
}

M_END_NAMESPACE

