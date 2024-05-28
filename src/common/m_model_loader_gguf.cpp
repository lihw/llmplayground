/** 
* Load the GGUF models from files
*
* Author: lihw81@gmail.com
*/

#include <common/m_model_loader_gguf.h>

#include <spdlog/spdlog.h>
#include <fmt/core.h>

#include <map>

#include <stdint.h>
#include <inttypes.h>

#define PATH_MAX 1024

M_BEGIN_NAMESPACE

static int ggufSplitPath(char* path, size_t maxlen, const char * pathPrefix, int splitNo, int splitCount) noexcept {
    static const char * const SPLIT_PATH_FORMAT = "%s-%05d-of-%05d.gguf";
    if (snprintf(path, maxlen, SPLIT_PATH_FORMAT, pathPrefix, splitNo + 1, splitCount)) {
        return int(strlen(path));
    }
    return 0;
}

static int ggufSplitPrefix(char * dest, size_t maxlen, const char * splitPath, int splitNo, int splitCount) noexcept {
    std::string strSplitPath(splitPath);
    char postfix[32];
    snprintf(postfix, 32, "-%05d-of-%05d.gguf", splitNo + 1, splitCount);
    std::string strPostfix(postfix);

    // check if dest ends with postfix
    size_t sizePrefix = strSplitPath.size() - strPostfix.size();
    if (sizePrefix > 0 && strSplitPath.find(strPostfix, sizePrefix) != std::string::npos) {
        snprintf(dest, std::min((size_t) sizePrefix + 1, maxlen), "%s", splitPath);
        return int(sizePrefix);
    }

    return 0;
}

static const char * ggufGetVerName(GgufVersion version) {
    switch (version) {
        case GgufVersion::V1: return "GGUF V1 (support until nov 2023)";
        case GgufVersion::V2: return "GGUF V2";
        case GgufVersion::V3: return "GGUF V3 (latest)";
    }

    return "unknown";
}

static std::string ggufGetTensorShape(const std::vector<int64_t>& ne) {
    char buf[256];
    snprintf(buf, sizeof(buf), "%5" PRId64, ne.at(0));
    for (size_t i = 1; i < ne.size(); i++) {
        snprintf(buf + strlen(buf), sizeof(buf) - strlen(buf), ", %5" PRId64, ne.at(i));
    }
    return buf;
}

ModelLoaderGguf::ModelLoaderGguf() 
    : ModelLoader()
{
}

bool ModelLoaderGguf::load(const std::string &file, bool useMmap) noexcept 
{
    if (!ModelLoader::load(file, useMmap)) {
        return false;
    }

    ggml_context* ctx = NULL;
    gguf_init_params params = {
        /*.no_alloc = */ true,
        /*.ctx      = */ &ctx,
    };

    mMeta = gguf_init_from_file(file.c_str(), params);
    if (!mMeta) {
        spdlog::error("%s: failed to load model from %s\n", __func__, file.c_str());
        return false;
    }

    getKey(Kv::GENERAL_ARCHITECTURE, mArchName, false);
    //llm_kv = LLM_KV(getArchFromName(mArchNam));

    // Save tensors data offset of the main file.
    // For subsidiary files, `meta` tensor data offset must not be used,
    // so we build a unified tensors index for weights.
    for (ggml_tensor* cur = ggml_get_first_tensor(ctx); cur; cur = ggml_get_next_tensor(ctx, cur)) {
        mWeights.emplace_back(0, cur->name, mMeta, cur);
    }
    //files.emplace_back(new llama_file(file.c_str(), "rb"));
    //contexts.emplace_back(ctx);
        
    uint16_t numSplits = 0;
    getKey(Kv::SPLIT_COUNT, numSplits, false);
        
    // Load additional GGML contexts
    if (numSplits > 1) {
        uint16_t idx = 0;
        getKey(Kv::SPLIT_NO, idx);
        if (idx != 0) {
            spdlog::error("illegal split file: %d, model must be loaded with the first split", idx);
            return false;
        }

        char splitPrefix[PATH_MAX] = {0};
        if (!ggufSplitPrefix(splitPrefix, sizeof(splitPrefix), file.c_str(), idx, numSplits)) {
            spdlog::error("invalid split file: %s", file.c_str());
            return false;
        }

        spdlog::info("%s: loading additional %d GGUFs\n", __func__, numSplits);

        char splitPath[PATH_MAX] = {0};
        for (idx = 1; idx < numSplits; idx++) {
            ggufSplitPath(splitPath, sizeof(splitPath), splitPrefix, idx, numSplits);

            gguf_init_params splitParams = {
                /*.no_alloc = */ true,
                /*.ctx      = */ &ctx,
            };
            gguf_context * ctxGguf = gguf_init_from_file(splitPath, splitParams);
            if (!ctxGguf) {
                spdlog::error("%s: failed to load GGUF split from %s\n", __func__, splitPath);
                return false;
            }

            // Save tensors data offset info of the shard.
            for (ggml_tensor * cur = ggml_get_first_tensor(ctx); cur; cur = ggml_get_next_tensor(ctx, cur)) {
                mWeights.emplace_back(idx, cur->name, ctx_gguf, cur);
            }
            //files.emplace_back(new llama_file(split_path, "rb"));
            mContexts.emplace_back(ctx);

            gguf_free(ctxGguf);
        }

        getKey(Kv::SPLIT_TENSORS_COUNT, mNumTensors);

        // sanity check
        {
            const int numTensorsLoaded = (int)mWeights.size();
            if (mNumTensors != numTensorsLoaded) {
                spdlog::error("corrupted model: %d tensors expected but %d found", mNumTensors, numTensorsLoaded);
                return false;
            }
        }

        spdlog::info("%s: additional %d GGUFs metadata loaded.\n",  __func__, numSplits - 1);
    } 


    mNumKv      = gguf_get_n_kv(mMeta);
    mNumTensors = mWeights.size();
    mVersion    = (GgufVersion)(gguf_get_version(mMeta));

    for (auto & w : mWeights) {
        mNumElements += ggml_nelements(w.tensor);
        mNumBytes    += ggml_nbytes(w.tensor);
    }

    spdlog::info("%s: loaded meta data with %d key-value pairs and %d tensors from %s (version %s)\n",
                __func__, mNumKv, mNumTensors, file.c_str(), ggufGetVerName(mVersion));
        
    // determine file type based on the number of tensors for each quantization and print meta data
    // TODO: make optional
    {
        GgufType ftype;

        std::map<enum ggml_type, uint32_t> typeCount;

        uint32_t maxNumType = 0;
        enum ggml_type typeMax = GGML_TYPE_F32;

        for (size_t i = 0; i < mNumTensors; i++) {
            const ggml_tensor* tensor = mWeights.at(i).tensor;
            enum ggml_type type = tensor->type;

            typeCount[type]++;

            if (maxNumType < typeCount[type]) {
                maxNumType = typeCount[type];
                typeMax = type;
            }

            const uint16_t sid = mWeights.at(i).idx;
            spdlog::info("%s: - tensor %4d, split %2d: %32s %-8s [ %s ]\n", __func__, i, sid, 
                    ggml_get_name(tensor), ggml_type_name(type), ggufGetTensorShape(tensor).c_str());
        }

        switch (typeMax) {
            case GGML_TYPE_F32:     ftype = GgufType::ALL_F32;        break;
            case GGML_TYPE_F16:     ftype = GgufType::MOSTLY_F16;     break;
            case GGML_TYPE_Q4_0:    ftype = GgufType::MOSTLY_Q4_0;    break;
            case GGML_TYPE_Q4_1:    ftype = GgufType::MOSTLY_Q4_1;    break;
            case GGML_TYPE_Q5_0:    ftype = GgufType::MOSTLY_Q5_0;    break;
            case GGML_TYPE_Q5_1:    ftype = GgufType::MOSTLY_Q5_1;    break;
            case GGML_TYPE_Q8_0:    ftype = GgufType::MOSTLY_Q8_0;    break;
            case GGML_TYPE_Q2_K:    ftype = GgufType::MOSTLY_Q2_K;    break;
            case GGML_TYPE_Q3_K:    ftype = GgufType::MOSTLY_Q3_K_M;  break;
            case GGML_TYPE_Q4_K:    ftype = GgufType::MOSTLY_Q4_K_M;  break;
            case GGML_TYPE_Q5_K:    ftype = GgufType::MOSTLY_Q5_K_M;  break;
            case GGML_TYPE_Q6_K:    ftype = GgufType::MOSTLY_Q6_K;    break;
            case GGML_TYPE_IQ2_XXS: ftype = GgufType::MOSTLY_IQ2_XXS; break;
            case GGML_TYPE_IQ2_XS:  ftype = GgufType::MOSTLY_IQ2_XS;  break;
            case GGML_TYPE_IQ2_S:   ftype = GgufType::MOSTLY_IQ2_S;   break;
            case GGML_TYPE_IQ3_XXS: ftype = GgufType::MOSTLY_IQ3_XXS; break;
            case GGML_TYPE_IQ1_S:   ftype = GgufType::MOSTLY_IQ1_S;   break;
            case GGML_TYPE_IQ1_M:   ftype = GgufType::MOSTLY_IQ1_M;   break;
            case GGML_TYPE_IQ4_NL:  ftype = GgufType::MOSTLY_IQ4_NL;  break;
            case GGML_TYPE_IQ4_XS:  ftype = GgufType::MOSTLY_IQ4_XS;  break;
            case GGML_TYPE_IQ3_S:   ftype = GgufType::MOSTLY_IQ3_S;   break;
            default:
                    {
                        spdlog::warn("%s: unknown type %s\n", __func__, ggml_type_name(typeMax));
                        ftype = GgufType::ALL_F32;
                    } break;
        }

        // this is a way to mark that we have "guessed" the file type
        ftype = (GgufType) (ftype | GgufType::GUESSED);

        {
            const int kid = gguf_find_key(mMeta, "general.file_type");
            if (kid >= 0) {
                ftype = (GgufType)gguf_get_val_u32(mMeta, kid);
            }
        }

        spdlog::info("%s: Dumping metadata keys/values. Note: KV overrides do not apply in this output.\n", __func__);
        for (int i = 0; i < mNumKv; i++) {
            const char* name            = gguf_get_key(mMeta, i);
            const enum gguf_type type   = gguf_get_kv_type(mMeta, i);
            const std::string type_name =
                type == GGUF_TYPE_ARRAY
                ? fmt::format("%s[%s,%d]", gguf_type_name(type), gguf_type_name(gguf_get_arr_type(mMeta, i)), gguf_get_arr_n(mMeta, i))
                : gguf_type_name(type);

            std::string value          = gguf_kv_to_str(mMeta, i);
            const size_t MAX_VALUE_LEN = 40;
            if (value.size() > MAX_VALUE_LEN) {
                value = fmt::format("%s...", value.substr(0, MAX_VALUE_LEN - 3).c_str());
            }

            spdlog::info("%s: - kv %3d: %42s %-16s = %s\n", __func__, i, name, type_name.c_str(), value.c_str());
        }

        // print type counts
        for (auto & kv : typeCount) {
            if (kv.second == 0) {
                continue;
            }

            spdlog::info("%s: - type %4s: %4d tensors\n", __func__, ggml_type_name(kv.first), kv.second);
        }
    }

    //if (!llama_mmap::SUPPORTED) {
    //    spdlog::warn("%s: mmap is not supported on this platform\n", __func__);
    //    useMmap = false;
    //}

    //mUseMmap = useMmap;

    return true;
}

ModelLoaderGguf::~ModelLoaderGguf() 
{
    if (mMeta) {
        gguf_free(mMeta);
    }
}

M_END_NAMESPACE
