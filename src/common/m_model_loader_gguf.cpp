/** 
* Load the GGUF models from files
*
* Author: lihw81@gmail.com
*/

#include <common/m_model_loader_gguf.h>

#include <common/m_misc.h>
#include <common/m_model.h>
#include <common/m_vocab.h>

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

/*
static std::string ggufGetTensorShape(const std::vector<int64_t>& ne) {
    char buf[256];
    snprintf(buf, sizeof(buf), "%5" PRId64, ne.at(0));
    for (size_t i = 1; i < ne.size(); i++) {
        snprintf(buf + strlen(buf), sizeof(buf) - strlen(buf), ", %5" PRId64, ne.at(i));
    }
    return buf;
}
*/
static std::string ggufGetTensorShape(const struct ggml_tensor * t) 
{
    char buf[256];
    snprintf(buf, sizeof(buf), "%5" PRId64, t->ne[0]);
    for (int i = 1; i < GGML_MAX_DIMS; i++) {
        snprintf(buf + strlen(buf), sizeof(buf) - strlen(buf), ", %5" PRId64, t->ne[i]);
    }
    return buf;
}

static std::string ggufDataToStr(enum gguf_type type, const void * data, int i) {
    switch (type) {
        case GGUF_TYPE_UINT8:   return std::to_string(((const uint8_t  *)data)[i]);
        case GGUF_TYPE_INT8:    return std::to_string(((const int8_t   *)data)[i]);
        case GGUF_TYPE_UINT16:  return std::to_string(((const uint16_t *)data)[i]);
        case GGUF_TYPE_INT16:   return std::to_string(((const int16_t  *)data)[i]);
        case GGUF_TYPE_UINT32:  return std::to_string(((const uint32_t *)data)[i]);
        case GGUF_TYPE_INT32:   return std::to_string(((const int32_t  *)data)[i]);
        case GGUF_TYPE_UINT64:  return std::to_string(((const uint64_t *)data)[i]);
        case GGUF_TYPE_INT64:   return std::to_string(((const int64_t  *)data)[i]);
        case GGUF_TYPE_FLOAT32: return std::to_string(((const float    *)data)[i]);
        case GGUF_TYPE_FLOAT64: return std::to_string(((const double   *)data)[i]);
        case GGUF_TYPE_BOOL:    return ((const bool *)data)[i] ? "true" : "false";
        default:                return fmt::format("unknown type %d", int(type));
    }
}

std::string ggufKvToStr(const struct gguf_context * ctx_gguf, int i) 
{
    const enum gguf_type type = gguf_get_kv_type(ctx_gguf, i);

    switch (type) {
        case GGUF_TYPE_STRING:
            return gguf_get_val_str(ctx_gguf, i);
        case GGUF_TYPE_ARRAY:
            {
                const enum gguf_type arr_type = gguf_get_arr_type(ctx_gguf, i);
                int arr_n = gguf_get_arr_n(ctx_gguf, i);
                const void * data = gguf_get_arr_data(ctx_gguf, i);
                std::stringstream ss;
                ss << "[";
                for (int j = 0; j < arr_n; j++) {
                    if (arr_type == GGUF_TYPE_STRING) {
                        std::string val = gguf_get_arr_str(ctx_gguf, i, j);
                        // escape quotes
                        replaceAll(val, "\\", "\\\\");
                        replaceAll(val, "\"", "\\\"");
                        ss << '"' << val << '"';
                    } else if (arr_type == GGUF_TYPE_ARRAY) {
                        ss << "???";
                    } else {
                        ss << ggufDataToStr(arr_type, data, j);
                    }
                    if (j < arr_n - 1) {
                        ss << ", ";
                    }
                }
                ss << "]";
                return ss.str();
            }
        default:
            return ggufDataToStr(type, gguf_get_val_data(ctx_gguf, i), 0);
    }
}

ModelLoaderGguf::ModelLoaderGguf() 
    : ModelLoader()
{
}

bool ModelLoaderGguf::load(const std::string& file) noexcept
{
    constexpr auto LOG_HEAD = "ModelLoaderGguf:load()";

    ggml_context* ctx = NULL;
    gguf_init_params p = {
        /*.no_alloc = */ true,
        /*.ctx      = */ &ctx,
    };

    mMeta = gguf_init_from_file(file.c_str(), p);
    if (!mMeta) {
        spdlog::error("{}: failed to load model from {}\n", LOG_HEAD, file.c_str());
        return false;
    }

    getKey(Kv::GENERAL_ARCHITECTURE, mArchName, false);

    // Save tensors data offset of the main file.
    // For subsidiary files, `meta` tensor data offset must not be used,
    // so we build a unified tensors index for weights.
    for (ggml_tensor* cur = ggml_get_first_tensor(ctx); cur; cur = ggml_get_next_tensor(ctx, cur)) {
        mWeights.emplace_back(uint16_t(0), cur->name, mMeta, cur);
    }

    mFiles.emplace_back(new File(file.c_str(), "rb"));
    mContexts.emplace_back(ctx);

    uint16_t numSplits = 0;
    getKey(Kv::SPLIT_COUNT, numSplits, false);

    // Load additional GGML contexts
    if (numSplits > 1) {
        uint16_t idx = 0;
        getKey(Kv::SPLIT_NO, idx);
        if (idx != 0) {
            spdlog::error("{}: illegal split file: {}, model must be loaded with the first split", LOG_HEAD, idx);
            return false;
        }

        char splitPrefix[PATH_MAX] = { 0 };
        if (!ggufSplitPrefix(splitPrefix, sizeof(splitPrefix), file.c_str(), idx, numSplits)) {
            spdlog::error("{}: invalid split file: {}", LOG_HEAD, file);
            return false;
        }

        spdlog::info("{}: loading additional {} GGUFs", LOG_HEAD, numSplits);

        char splitPath[PATH_MAX] = { 0 };
        for (idx = 1; idx < numSplits; idx++) {
            ggufSplitPath(splitPath, sizeof(splitPath), splitPrefix, idx, numSplits);

            gguf_init_params splitParams = {
                /*.no_alloc = */ true,
                /*.ctx      = */ &ctx,
            };
            gguf_context* ctxGguf = gguf_init_from_file(splitPath, splitParams);
            if (!ctxGguf) {
                spdlog::error("{}: failed to load GGUF split from {}", LOG_HEAD, splitPath);
                return false;
            }

            // Save tensors data offset info of the shard.
            for (ggml_tensor* cur = ggml_get_first_tensor(ctx); cur; cur = ggml_get_next_tensor(ctx, cur)) {
                mWeights.emplace_back(idx, cur->name, ctxGguf, cur);
            }
            mFiles.emplace_back(new File(splitPath, "rb"));
            mContexts.emplace_back(ctx);

            gguf_free(ctxGguf);
        }

        getKey(Kv::SPLIT_TENSORS_COUNT, mNumTensors);

        // sanity check
        {
            const int numTensorsLoaded = (int)mWeights.size();
            if (int(mNumTensors) != numTensorsLoaded) {
                spdlog::error("corrupted model: %d tensors expected but %d found", mNumTensors, numTensorsLoaded);
                return false;
            }
        }

        spdlog::info("{}: additional {} GGUFs metadata loaded.\n", LOG_HEAD, numSplits - 1);
    }


    mNumKeyValues = gguf_get_n_kv(mMeta);
    mNumTensors = mWeights.size();
    mVersion = (GgufVersion)(gguf_get_version(mMeta));

    for (auto& w : mWeights) {
        mNumElements += ggml_nelements(w.tensor);
        mNumBytes += ggml_nbytes(w.tensor);
    }

    spdlog::info("{}: loaded meta data with {} key-value pairs and {} tensors from {} (version {})\n",
        LOG_HEAD, mNumKeyValues, mNumTensors, file, ggufGetVerName(mVersion));

    // determine file type based on the number of tensors for each quantization and print meta data
    // TODO: make optional
    {
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
            spdlog::info("{}: tensor {:d}, split {:2d}: {:28s} {:6s} [{}]", LOG_HEAD, i, sid,
                ggml_get_name(tensor), ggml_type_name(type), ggufGetTensorShape(tensor).c_str());
        }

        for (auto& [type, number] : typeCount) {
            if (number == 0) {
                continue;
            }
            spdlog::info("{}: type {:4s}: {:4d} tensors", LOG_HEAD, ggml_type_name(type), number);
        }
        spdlog::info("{}: total:     {:4d} tensors\n", LOG_HEAD, mNumTensors);


        spdlog::info("{}: dumping metadata keys/values. Note: KV overrides do not apply in this output.", LOG_HEAD);
        for (int i = 0; i < mNumKeyValues; i++) {
            const char* name = gguf_get_key(mMeta, i);
            const enum gguf_type type = gguf_get_kv_type(mMeta, i);
            const std::string typeName =
                type == GGUF_TYPE_ARRAY
                ? fmt::format("{}[{},{}]", gguf_type_name(type), gguf_type_name(gguf_get_arr_type(mMeta, i)), gguf_get_arr_n(mMeta, i))
                : gguf_type_name(type);

            std::string value = ggufKvToStr(mMeta, i);
            const size_t MAX_VALUE_LEN = 40;
            if (strncmp(name, "general.file_type", 16) == 0) {
                uint32_t v = ((uint32_t*)gguf_get_val_data(mMeta, i))[0];
                value = getGgufTypeName(GgufType(v));
            }
            else if (value.size() > MAX_VALUE_LEN) {
                value = fmt::format("{}...", value.substr(0, MAX_VALUE_LEN - 3));
            }

            spdlog::info("{}: {:3d}: {:42s} {:<16s} = {}", LOG_HEAD, i, name, typeName, value);
        }
        spdlog::info("{}: total {} key/values\n", LOG_HEAD, mNumKeyValues);
    }


#if defined _WIN32
    mUseMmap = params.useMmap;
#elif defined _POSIX_MAPPED_FILES
    mUseMmap = params.useMmap;
#else
    if (params.useMmap) {
        spdlog::warn("{}: mmap is not supported on this platform", LOG_HEAD);
        mUseMmap = false;
    }
#endif

    return true;
}

ModelLoaderGguf::~ModelLoaderGguf() 
{
    if (mMeta) {
        gguf_free(mMeta);
    }
}

Model* ModelLoaderGguf::build() noexcept
{
    const auto LOG_HEAD = "ModelLoaderGguf::build()";

    Model* model = new Model;

    //
    // load parameters
    //
    if (!model->loadParameters(*this)) {
        spdlog::error("{}: error loading model parameters", LOG_HEAD);
        return nullptr;
    }
    spdlog::info("{}: model parameters loaded", LOG_HEAD);

    //
    // load model vocab
    // 
    model->loadVocab(*this);
    if (params.vocabOnly) {
        spdlog::info("{}: vocab only - skipping tensors", LOG_HEAD);
        return model;
    }
    spdlog::info("{}: model vocab loaded", LOG_HEAD);

    //
    // load tensors
    //
    if (!model->loadTensors(*this, params.mainGpu, params.numGpuLayers, params.useMemoryLock)) {
        spdlog::error("{}: error loading model parameters", LOG_HEAD);
        return nullptr;
    }
    spdlog::info("{}: model tensors loaded", LOG_HEAD);

    /*
    (
            ml, model, params.n_gpu_layers, params.split_mode,  params.main_gpu, params.tensor_split, params.use_mlock,
            params.progress_callback, params.progress_callback_user_data
        )) {
            return nullptr;
        }
    } catch (const std::exception & err) {
        spdlog::error("{}: error loading model: {}\n", LOG_HEAD, err.what());
        return nullptr;
    }
    */

    return model;
}

M_END_NAMESPACE
