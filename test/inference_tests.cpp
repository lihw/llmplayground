#include <catch2/catch_test_macros.hpp>

#include <common/m_model_loader.h>
#include <common/m_token.h>
#include <common/m_vocab.h>
#include <common/m_model.h>

#include "../src/inference/m_inference.h"
#include "../src/inference/m_batch.h"
#include "../src/tokenizer/m_tokenizer_spm.h"

#include <spdlog/spdlog.h>

#include <unordered_map>
#include <vector>

TEST_CASE("tokenizer", "[spm]")
{
    spdlog::set_pattern("%v");

    const char* MODEL_FILE = "Llama-2-7b-GGUF/llama-2-7b.gguf";

    auto* model = m::loadModel(MODEL_FILE, m::ModelLoader::Parameters());
    REQUIRE(model->isValid() == true);

    const std::string text = "Hello my name is";

    std::vector<m::TokenId> tokens;
    m::Vocab& vocab = model->vocab;
    m::tokenize(text, vocab, true, false, tokens);

    auto* context = m::infer::createContext(model, m::infer::Context::Parameters());
    
    // total length of the sequence including the prompt
    const size_t length = 32;
    const size_t contextSize = context->params.contextSize; // Retrieve the attention context length
    const size_t n_kv_req = tokens.size() + (contextSize - tokens.size());
    spdlog::info("length = {}, context_size = {}, n_kv_req = {}", length, contextSize, n_kv_req);

    REQUIRE(contextSize == 2048);
    REQUIRE(n_kv_req > contextSize);
    // LOG_TEE("%s: error: n_kv_req > n_ctx, the required KV cache size is not big enough\n", __func__);
    // LOG_TEE("%s:        either reduce n_len or increase n_ctx\n", __func__);

    auto* batch = m::createBatch(512, 0, 1);
    // evaluate the initial prompt
    for (size_t i = 0; i < tokens.size(); i++) {
        batch->add(tokens[i], m::Pos(i), { 0 }, false);
    }
    // decode will output logits only for the last token of the prompt
    batch->logits[batch->numTokens - 1] = true;

    REQUIRE(m::infer::decode(context, batch) == 0);

#if 0
    for (auto& [text, tokens] : tests) {
        std::vector<m::TokenId> out_tokensBos;
        std::vector<m::TokenId> out_tokensNoBos;
        m::tokenize(text, vocab, true, false, out_tokensBos);
        m::tokenize(text, vocab, false, false, out_tokensNoBos);

        if (tokens.size() == 0) {
            REQUIRE(out_tokensBos.size() == 0);
            REQUIRE(out_tokensNoBos.size() == 0);
            continue;
        }

        REQUIRE(out_tokensNoBos == tokens);
        REQUIRE(out_tokensBos[0] == vocab.specialBosId);
        REQUIRE(std::vector<m::TokenId>(out_tokensBos.begin() + 1, out_tokensBos.end()) == tokens);
    }
#endif

#if 0
    for (auto& [text, tokens] : tests) {
        std::vector<m::TokenId> out_tokensBos;
        m::tokenize(text, vocab, true, false, out_tokensBos);

        std::string out_text;
        m::detokenize(out_tokensBos, vocab, true, out_text);
        
        spdlog::info("{}", out_text);
    }
#endif

}
