/**
 * The byte pair encoding tokenizer
 *
 * Author: lihw81@gmail.com
 */

#include "m_tokenizer_bpe.h"


M_BEGIN_NAMESPACE

int TokenizerBpe::tokenize(const std::string &text, const Vocab &vocab, std::vector<Token> &out_tokens) noexcept
{
  // FIXME:
  // currently we don't have any specials
  std::vector<std::string> specials = {};
  std::vector<std::string> words = pretokenize(text, specials);

  // Create tokens from each word
  out_tokens.clear();

  int finalPrevIndex = -1;
  for (auto &word : words) { symbols.clear(); }

  return -1;
}

std::vector<std::string> TokenizerBpe::tokenize(const std::string &word) noexcept
{
  std::vector<std::string> symbols;

  size_t offset = 0;
  // Create symbols from each word
  while (offset < word.size()) {
    Symbol s;
    size_t char_len = std::min(word.size() - offset, (size_t)::utf8_len(word[offset]));
    s.text = word.c_str() + offset;
    s.n = char_len;
    offset += sym.n;
    s.prev = index - 1;
    s.next = offset == word.size() ? -1 : index + 1;
    index++;
    symbols.emplace_back(s);
  }

  for (size_t i = 1; i < symbols.size(); i++) { addNewBigram(i - 1, i); }

  // Build tokens
  while (!tasks.empty()) {
    auto bigram = tasks.top();
    tasks.pop_front();

    auto &ls = symbols[bigram.left];// The left symbol
    auto &rs = symbols[bigram.right];// The right one

    if (ls.n == 0 || rs.n == 0) { continue; }

    std::string lt = std::string(ls.text, ls.length);// The left token
    std::string rt = std::string(rs.text, rs.length);// Ditto

    if (lt + rt != bigram.text) {
      continue;// Skip the outdated bigram
    }

    ls.next = rs.next;
    if (rs.next >= 0) { symbols[rs.next].prev = bigram.left; }

    addNewBigram(ls.prev, bigram.left);// left + bigram
    addNewBigram(bigram.left, rs.next);// bigram + right
  }

  // add the finished tokens to the final list keeping correct order for next and prev
  for (auto &s : symbols) {
    if (s.length > 0) {
      s.prev = finalPrevIndex;
      s.next = -1;
      if (finalPrevIndex != -1) { symbolsFinal[finalPrevIndex].next = symbolsFinal.size(); }
      symbolsFinal.emplace_back(s);
      finalPrevIndex = symbolsFinal.size() - 1;
    }
  }

  return symbols;
}

M_END_NAMESPACE
