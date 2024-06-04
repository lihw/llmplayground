#include <catch2/catch_test_macros.hpp>

#include <common/m_model_loader.h>
#include <common/m_token.h>
#include <common/m_vocab.h>

#include "../src/tokenizer/m_tokenizer_spm.h"

#include <spdlog/spdlog.h>

#include <vector>

TEST_CASE("tokenizer", "[bpe]")
{
    spdlog::set_pattern("%v");

    const char* MODEL_FILE = "Llama-2-7b-GGUF/llama-2-7b.gguf";

    m::Model model = m::loadModel(MODEL_FILE);
    if (!model.isValid()) {
        spdlog::error("{} is invalid", MODEL_FILE);
        REQUIRE(model.isValid() == true);
    }

    m::Vocab vocab;
    bool vocabLoaded = vocab.load(*model.ml);
    if (!vocabLoaded) {
        spdlog::error("fail to load vocab");
        REQUIRE(vocabLoaded == true);
    }

    std::vector<m::TokenId> out_tokens;
    int numTokens = m::tokenize("hello, world", vocab, true, true, out_tokens);
    REQUIRE(numTokens > 0);
}
