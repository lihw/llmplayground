#include <catch2/catch_test_macros.hpp>


#include <common/tokenizer.hpp>


TEST_CASE("tokenizer", "[bpe]")
{
    std::vector<m::TokenId> out_tokens;
    int numTokens = m::tokenize("hello, world", vocab, true, true, out_tokens);
}
