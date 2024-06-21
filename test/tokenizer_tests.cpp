#include <catch2/catch_test_macros.hpp>

#include <common/m_model_loader.h>
#include <common/m_token.h>
#include <common/m_vocab.h>
#include <common/m_model.h>

#include "../src/tokenizer/m_tokenizer_spm.h"

#include <spdlog/spdlog.h>

#include <unordered_map>
#include <vector>

TEST_CASE("tokenizer", "[spm]")
{
    spdlog::set_pattern("%v");

    const char* MODEL_FILE = "Llama-2-7b-GGUF/llama-2-7b.gguf";

    m::Model* model = m::loadModel(MODEL_FILE);
    REQUIRE(model->isValid() == true);

    m::Vocab& vocab = model->vocab;

    static std::vector<std::pair<std::string, std::vector<m::TokenId>>> tests = {
        { ""                      , {  }, },
        { " "                     , {     259, }, },
        { "  "                    , {    1678, }, },
        { "   "                   , {     268, }, },
        { "\t"                    , {   29871,     12, }, },
        { "\n"                    , {   29871,     13, }, },
        { "\t\n"                  , {   29871,     12,     13, }, },
        { "Hello world"           , {   15043,   3186, }, },
        { " Hello world"          , {   29871,  15043,   3186, }, },
        { "Hello World"           , {   15043,   2787, }, },
        { " Hello World"          , {   29871,  15043,   2787, }, },
        { " Hello World!"         , {   29871,  15043,   2787,  29991, }, },
        { "Hello, world!"         , {   15043,  29892,   3186,  29991, }, },
        { " Hello, world!"        , {   29871,  15043,  29892,   3186,  29991, }, },
        { " this is 🦙.cpp"        , {   29871,    445,    338,  29871,    243,    162,    169,    156,  29889,   8223, }, },
        { "w048 7tuijk dsdfhu"    , {     281,  29900,  29946,  29947,  29871,  29955,   9161,  13535,  18031,   2176,   6905, }, },
        { "нещо на Български"     , {    1538,   4851,    665,   1386,  29713,   1305, }, },
        { "កាន់តែពិសេសអាចខលចេញ"   , {   29871,  31849,  31324,  31934,    228,    162,    142,    228,    161,    146,    228,    162,    133,    228,    161,    153,    228,    161,    186,  31708,    228,    162,    132,  31708,    228,    161,    165,  31324,    228,    161,    136,    228,    161,    132,    228,    161,    158,    228,    161,    136,    228,    162,    132,    228,    161,    140, }, },
        { "🚀 (normal) 😶‍🌫️ (multiple emojis concatenated) ✅ (only emoji that has its own token)", {   29871,    243,    162,    157,    131,    313,   8945,  29897,  29871,    243,    162,    155,    185,  30722,    243,    162,    143,    174,  30598,    313,  20787,    953,   3848,    275,  16125,    630,  29897,  29871,  31681,    313,   6194,    953,  29877,   2397,    393,    756,    967,   1914,   5993,  29897, }, },
        { "Hello"                 , {   15043, }, },
        { " Hello"                , {   29871,  15043, }, },
        { "  Hello"               , {     259,  15043, }, },
        { "   Hello"              , {    1678,  15043, }, },
        { "    Hello"             , {     268,  15043, }, },
        { "    Hello\n    Hello"  , {     268,  15043,     13,   1678,  15043, }, },
        { " ("                    , {   29871,  313, }, },
    };

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
