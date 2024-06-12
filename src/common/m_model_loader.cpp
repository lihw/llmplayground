/** 
* Load the GGUF models from files
*
* Author: lihw81@gmail.com
*/

#include <common/m_model_loader.h>

#include <common/m_model_loader_gguf.h>

#include <spdlog/spdlog.h>
#include <fmt/core.h>

#include <array>

M_BEGIN_NAMESPACE

ModelLoader::ModelLoader() 
{
}

ModelLoader::~ModelLoader() 
{
    for (auto* ctx : mContexts) {
        ggml_free(ctx);
    }
}

const char* ModelLoader::getTensorName(int i) const noexcept 
{
    return mWeights.at(size_t(i)).tensor->name;
}

const ModelLoader::Weight* ModelLoader::getWeight(const char * name) const noexcept 
{
    const auto LOG_HEAD = "ModelLoader::getWeight()";

    for (auto& weight : mWeights) {
        if (strcmp(name, weight.tensor->name) == 0) {
            return &weight;
        }
    }
    spdlog::error("{}: tensor '{}' not found", LOG_HEAD, name);
    return nullptr;
}

ggml_tensor* ModelLoader::getTensorMeta(const char* name) const noexcept 
{
    const auto * weight = getWeight(name);
    if (!weight) {
        spdlog::error("%s: tensor '%s' not found", __func__, name);
        return nullptr;
    }
    return weight->tensor;
}

ggml_tensor* ModelLoader::getTensorMeta(int i) const noexcept
{
    return getTensorMeta(getTensorName(i));
}

ggml_tensor* ModelLoader::createTensorFor(ggml_context * ctx, const ggml_tensor * cur) 
{
    struct ggml_tensor* tensor = ggml_dup_tensor(ctx, cur);
    ggml_set_name(tensor, ggml_get_name(cur));

    mNumCreated++;

    return tensor;
}

const ggml_tensor* ModelLoader::checkTensorDims(const std::string & name, const std::vector<int64_t> & ne, bool required) const {
    const auto LOG_HEAD = "ModelLoader::checkTensorDims";

    const struct ggml_tensor * cur = getTensorMeta(name.c_str());

    if (cur == nullptr) {
        if (required) {
            throw std::runtime_error(fmt::format("{}: tensor '{}' not found", LOG_HEAD, name));
        }
        
        return nullptr;
    }

    {
        bool ok = true;
        for (size_t i = 0; i < GGML_MAX_DIMS; ++i) {
            if ((i < ne.size() && ne[i] != cur->ne[i]) || (i >= ne.size() && cur->ne[i] != 1)) {
                ok = false;
                break;
            }
        }

        auto formatTensorShape1 = [](const std::vector<int64_t> &ne) -> std::string {
                char buf[256];
                snprintf(buf, sizeof(buf), "%5" PRId64, ne.at(0));
                for (size_t i = 1; i < ne.size(); i++) {
                    snprintf(buf + strlen(buf), sizeof(buf) - strlen(buf), ", %5" PRId64, ne.at(i));
                }
                return buf;
            };
        auto formatTensorShape2 = [](const struct ggml_tensor* t) -> std::string {
                char buf[256];
                snprintf(buf, sizeof(buf), "%5" PRId64, t->ne[0]);
                for (int i = 1; i < GGML_MAX_DIMS; i++) {
                    snprintf(buf + strlen(buf), sizeof(buf) - strlen(buf), ", %5" PRId64, t->ne[i]);
                }
                return buf;
            };

        if (!ok)
        {
            throw std::runtime_error(
                    fmt::format("{}: tensor '{}' has wrong shape; expected {}, got {}",
                        LOG_HEAD, name,
                        formatTensorShape1(ne),
                        formatTensorShape2(cur)));
        }
    }

    return cur;
}

ggml_tensor * ModelLoader::createTensor(struct ggml_context * ctx, const std::string & name, const std::vector<int64_t> & ne, bool required) {
    const struct ggml_tensor * cur = checkTensorDims(name, ne, required);

    if (cur == NULL) {
        return NULL;
    }

    return createTensorFor(ctx, cur);
}

ggml_tensor * ModelLoader::createTensorAsView(ggml_context * ctx, ggml_tensor * base, const std::string & name, const std::vector<int64_t> & ne, size_t offset, bool required) {
    const auto LOG_HEAD = "ModelLoader::createTensorAsView()";

    const struct ggml_tensor * cur = checkTensorDims(name, ne, required);

    if (cur == NULL) {
        return NULL;
    }

    if (cur->type != base->type) {
        throw std::runtime_error(fmt::format("{}: tensor '{}' has wrong type; expected {}, got {}", __func__, name.c_str(), ggml_type_name(base->type), ggml_type_name(cur->type)));
        return nullptr;
    }

    std::array<int64_t, GGML_MAX_DIMS> dims;
    for (size_t i = 0; i < GGML_MAX_DIMS; ++i) {
        dims[i] = i < ne.size() ? ne[i] : 1;
    }

    struct ggml_tensor * tensor = ggml_view_4d(ctx, base,
            dims[0], dims[1], dims[2], dims[3],
            cur->nb[1], cur->nb[2], cur->nb[3],
            offset);

    ggml_set_name(tensor, name.c_str());

    mNumCreated++;

    return tensor;
}

bool ModelLoader::areAllTensorsCreated() const  {
    const auto LOG_HEAD = "ModelLoader::areAllTensorsCreated()";

    if (mNumCreated != mNumTensors) {
        throw std::runtime_error(fmt::format("{}: wrong number of tensors; expected {}, got {}", 
                LOG_HEAD, mNumTensors, mNumCreated));
        return false;
    }
    return true;
}

#if 0
void init_mappings(bool prefetch = true, llama_mlocks * mlock_mmaps = nullptr) {
    if (use_mmap) {
        mappings.reserve(files.size());
        mmaps_used.reserve(files.size());
        for (const auto & file : files) {
            std::unique_ptr<llama_mmap> mapping(new llama_mmap(file.get(), prefetch ? -1 : 0, ggml_is_numa()));
            mmaps_used.emplace_back(mapping->size, 0);
            if (mlock_mmaps) {
                std::unique_ptr<llama_mlock> mlock_mmap(new llama_mlock());
                mlock_mmap->init(mapping->addr);
                mlock_mmaps->emplace_back(std::move(mlock_mmap));
            }
            mappings.emplace_back(std::move(mapping));
        }
    }

    // compute the total size of all tensors for progress reporting
    for (auto & w : weights) {
        size_data += ggml_nbytes(w.tensor);
    }
}

void get_mapping_range(size_t * first, size_t * last, void ** addr, int idx, ggml_context * ctx) const {
    GGML_ASSERT(!mappings.empty());
    const auto & mapping = mappings.at(idx);

    *first = mapping->size;
    *last  = 0;
    *addr = mapping->addr;
    for (ggml_tensor * tensor = ggml_get_first_tensor(ctx); tensor; tensor = ggml_get_next_tensor(ctx, tensor)) {
        try {
            const auto * weight = get_weight(ggml_get_name(tensor));
            if (!weight) {
                continue;
            }
            if (weight->idx != idx) {
                continue;
            }
            *first = std::min(*first, weight->offs);
            *last  = std::max(*last,  weight->offs + ggml_nbytes(tensor));
        } catch(...) {
            // the tensor is not in the model
        }
    }
}

// for backwards compatibility, does not support ggml-backend
void load_data_for(struct ggml_tensor * cur) const {
    const auto & w = require_weight(ggml_get_name(cur));

    if (use_mmap) {
        const auto & mapping = mappings.at(w.idx);
        if (cur->data == nullptr) {
            cur->data = (uint8_t *)mapping->addr + w.offs;
        } else {
            memcpy(cur->data, (uint8_t *)mapping->addr + w.offs, ggml_nbytes(cur));
        }
    } else {
        GGML_ASSERT(cur->data != nullptr);
        GGML_ASSERT(w.idx < files.size());
        const auto & file = files.at(w.idx);
        file->seek(w.offs, SEEK_SET);
        file->read_raw(cur->data, ggml_nbytes(cur));
    }
}

size_t size_done = 0;
size_t size_data = 0;
std::vector<std::pair<size_t, size_t>> mmaps_used;

// Returns false if cancelled by progress_callback
bool load_all_data(
        struct ggml_context * ctx,
        llama_buf_map & bufs_mmap,
        llama_mlocks * lmlocks,
        llama_progress_callback progress_callback,
        void * progress_callback_user_data) {
    GGML_ASSERT(size_data != 0 && "call init_mappings() first");

    std::vector<no_init<uint8_t>> read_buf;
    for (struct ggml_tensor * cur = ggml_get_first_tensor(ctx); cur != NULL; cur = ggml_get_next_tensor(ctx, cur)) {
        const auto * weight = get_weight(ggml_get_name(cur));
        if (weight == nullptr) {
            // this can happen with split experts models
            continue;
        }

        if (progress_callback) {
            if (!progress_callback((float) size_done / size_data, progress_callback_user_data)) {
                return false;
            }
        }

        size_t n_size = ggml_nbytes(cur);

        if (use_mmap) {
            const auto & mapping = mappings.at(weight->idx);
            ggml_backend_buffer_t buf_mmap = nullptr;
            if (bufs_mmap.count(weight->idx)) {
                buf_mmap = bufs_mmap.at(weight->idx);
            }
            GGML_ASSERT(buf_mmap || cur->data); // either we have a buffer to allocate the tensor in, or it is already allocated
            if (buf_mmap && cur->data == nullptr) {
                ggml_backend_tensor_alloc(buf_mmap, cur, (uint8_t *) mapping->addr + weight->offs);
                if (lmlocks) {
                    const auto & lmlock = lmlocks->at(weight->idx);
                    lmlock->grow_to(weight->offs + ggml_nbytes(cur));
                }

                auto & mmap_used = mmaps_used[weight->idx];
                mmap_used.first  = std::min(mmap_used.first,  weight->offs);
                mmap_used.second = std::max(mmap_used.second, weight->offs + n_size);
            } else {
                ggml_backend_tensor_set(cur, (uint8_t *) mapping->addr + weight->offs, 0, n_size);
            }
        } else {
            GGML_ASSERT(weight->idx < files.size());
            const auto & file = files.at(weight->idx);
            if (ggml_backend_buffer_is_host(cur->buffer)) {
                file->seek(weight->offs, SEEK_SET);
                file->read_raw(cur->data, ggml_nbytes(cur));
            } else {
                read_buf.resize(ggml_nbytes(cur));
                file->seek(weight->offs, SEEK_SET);
                file->read_raw(read_buf.data(), ggml_nbytes(cur));
                ggml_backend_tensor_set(cur, read_buf.data(), 0, n_size);
            }
        }

        size_done += n_size;
    }

    // check if this is the last call and do final cleanup
    if (size_done >= size_data) {
        // unmap offloaded tensors and metadata
        if (use_mmap) {
            for (uint32_t idx = 0; idx < mappings.size(); idx++) {
                const auto & mmap_used = mmaps_used.at(idx);
                auto & mapping = mappings.at(idx);
                mapping->unmap_fragment(0, mmap_used.first);
                if (mmap_used.second != 0) {
                    mapping->unmap_fragment(mmap_used.second, mapping->size);
                }
            }
        }
        if (progress_callback) {
            // Even though the model is done loading, we still honor
            // cancellation since we need to free allocations.
            return progress_callback(1.0f, progress_callback_user_data);
        }
    }

    return true;
}

#endif

#if 0

Model::~Model()
{
    delete ml;
}

Model loadModel(const char* file) noexcept 
{
    unsigned curPercentage = 0;
    auto progress_callback_user_data = &curPercentage;
    auto progress_callback = [](float progress, void * ctx) {
        unsigned * cur_percentage_p = (unsigned *) ctx;
        unsigned percentage = (unsigned) (100 * progress);
        while (percentage > *cur_percentage_p) {
            *cur_percentage_p = percentage;
            spdlog::info(".");
            if (percentage >= 100) {
                spdlog::info("\n");
            }
        }
        return true;
    };

    const char* suffix = strrchr(file, '.');
    if (strncmp(suffix, ".gguf", 5) == 0) {
        ModelLoaderGguf* modelLoader = new ModelLoaderGguf();
        if (!modelLoader->load(std::string(file))) {
            spdlog::error("{}: failed to load model file {}", __func__, file);
            delete modelLoader;
            return Model();
        }

        return Model(modelLoader);
    } 

    spdlog::error("{}: unsupported model format {}", __func__, file);
    return Model();
}

#endif

M_END_NAMESPACE
