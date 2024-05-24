/** 
* Load the GGUF models from files
*
* Author: lihw81@gmail.com
*/

#include "m_model_loader.h"

M_BEGIN_NAMESPACE



struct llama_model_loader {
    int n_kv      = 0;
    int n_tensors = 0;
    int n_created = 0;

    int64_t n_elements = 0;
    size_t  n_bytes    = 0;

    bool use_mmap = false;

    llama_files files;
    llama_ftype ftype;
    llama_fver  fver;

    llama_mmaps mappings;

    // Holds information on a model weight

    std::unordered_map<std::string, struct llama_model_kv_override> kv_overrides;



    llama_model_loader(const std::string & fname, bool use_mmap, const struct llama_model_kv_override * param_overrides_p) {
        int trace = 0;
        if (getenv("LLAMA_TRACE")) {
            trace = atoi(getenv("LLAMA_TRACE"));
        }

        if (param_overrides_p != nullptr) {
            for (const struct llama_model_kv_override *p = param_overrides_p; p->key[0] != 0; p++) {
                kv_overrides.insert({std::string(p->key), *p});
            }
        }

        struct ggml_context * ctx = NULL;
        struct gguf_init_params params = {
            /*.no_alloc = */ true,
            /*.ctx      = */ &ctx,
        };

        meta = gguf_init_from_file(fname.c_str(), params);
        if (!meta) {
            throw std::runtime_error(format("%s: failed to load model from %s\n", __func__, fname.c_str()));
        }

        get_key(llm_kv(LLM_KV_GENERAL_ARCHITECTURE), arch_name, false);
        llm_kv = LLM_KV(llm_arch_from_string(arch_name));

        // Save tensors data offset of the main file.
        // For subsidiary files, `meta` tensor data offset must not be used,
        // so we build a unified tensors index for weights.
        for (ggml_tensor * cur = ggml_get_first_tensor(ctx); cur; cur = ggml_get_next_tensor(ctx, cur)) {
            weights.emplace_back(0, cur->name, meta, cur);
        }
        files.emplace_back(new llama_file(fname.c_str(), "rb"));
        contexts.emplace_back(ctx);

        uint16_t n_split = 0;
        get_key(llm_kv(LLM_KV_SPLIT_COUNT), n_split, false);

        // Load additional GGML contexts
        if (n_split > 1) {
            uint16_t idx = 0;
            get_key(llm_kv(LLM_KV_SPLIT_NO), idx);
            if (idx != 0) {
                throw std::runtime_error(format("illegal split file: %d, model must be loaded with the first split", idx));
            }

            char split_prefix[PATH_MAX] = {0};
            if (!llama_split_prefix(split_prefix, sizeof(split_prefix), fname.c_str(), idx, n_split)) {
                throw std::runtime_error(format("invalid split file: %s", fname.c_str()));
            }

            if (trace > 0) {
                LLAMA_LOG_INFO("%s: loading additional %d GGUFs\n", __func__, n_split);
            }

            char split_path[PATH_MAX] = {0};
            for (idx = 1; idx < n_split; idx++) {
                llama_split_path(split_path, sizeof(split_path), split_prefix, idx, n_split);

                struct gguf_init_params split_params = {
                    /*.no_alloc = */ true,
                    /*.ctx      = */ &ctx,
                };
                struct gguf_context * ctx_gguf = gguf_init_from_file(split_path, split_params);
                if (!ctx_gguf) {
                    throw std::runtime_error(format("%s: failed to load GGUF split from %s\n", __func__, split_path));
                }

                // Save tensors data offset info of the shard.
                for (ggml_tensor * cur = ggml_get_first_tensor(ctx); cur; cur = ggml_get_next_tensor(ctx, cur)) {
                    weights.emplace_back(idx, cur->name, ctx_gguf, cur);
                }
                files.emplace_back(new llama_file(split_path, "rb"));
                contexts.emplace_back(ctx);

                gguf_free(ctx_gguf);
            }

            get_key(llm_kv(LLM_KV_SPLIT_TENSORS_COUNT), n_tensors);

            // sanity check
            {
                const int n_tensors_loaded = (int) weights.size();
                if (n_tensors != n_tensors_loaded) {
                    throw std::runtime_error(format("corrupted model: %d tensors expected but %d found", n_tensors, n_tensors_loaded));
                }
            }

            LLAMA_LOG_INFO("%s: additional %d GGUFs metadata loaded.\n",  __func__, n_split - 1);
        }

        n_kv      = gguf_get_n_kv(meta);
        n_tensors = weights.size();

        fver = (enum llama_fver) gguf_get_version(meta);

        for (auto & w : weights) {
            n_elements += ggml_nelements(w.tensor);
            n_bytes    += ggml_nbytes(w.tensor);
        }

        LLAMA_LOG_INFO("%s: loaded meta data with %d key-value pairs and %d tensors from %s (version %s)\n",
                __func__, n_kv, n_tensors, fname.c_str(), llama_file_version_name(fver));

        // determine file type based on the number of tensors for each quantization and print meta data
        // TODO: make optional
        {
            std::map<enum ggml_type, uint32_t> n_type;

            uint32_t n_type_max = 0;
            enum ggml_type type_max = GGML_TYPE_F32;

            for (int i = 0; i < n_tensors; i++) {
                const ggml_tensor * tensor = weights.at(i).tensor;
                enum ggml_type type = tensor->type;

                n_type[type]++;

                if (n_type_max < n_type[type]) {
                    n_type_max = n_type[type];
                    type_max   = type;
                }

                if (trace > 0) {
                    const uint16_t sid = weights.at(i).idx;
                    LLAMA_LOG_INFO("%s: - tensor %4d, split %2d: %32s %-8s [ %s ]\n", __func__, i, sid, ggml_get_name(tensor), ggml_type_name(type), llama_format_tensor_shape(tensor).c_str());
                }
            }

            switch (type_max) {
                case GGML_TYPE_F32:     ftype = LLAMA_FTYPE_ALL_F32;        break;
                case GGML_TYPE_F16:     ftype = LLAMA_FTYPE_MOSTLY_F16;     break;
                case GGML_TYPE_Q4_0:    ftype = LLAMA_FTYPE_MOSTLY_Q4_0;    break;
                case GGML_TYPE_Q4_1:    ftype = LLAMA_FTYPE_MOSTLY_Q4_1;    break;
                case GGML_TYPE_Q5_0:    ftype = LLAMA_FTYPE_MOSTLY_Q5_0;    break;
                case GGML_TYPE_Q5_1:    ftype = LLAMA_FTYPE_MOSTLY_Q5_1;    break;
                case GGML_TYPE_Q8_0:    ftype = LLAMA_FTYPE_MOSTLY_Q8_0;    break;
                case GGML_TYPE_Q2_K:    ftype = LLAMA_FTYPE_MOSTLY_Q2_K;    break;
                case GGML_TYPE_Q3_K:    ftype = LLAMA_FTYPE_MOSTLY_Q3_K_M;  break;
                case GGML_TYPE_Q4_K:    ftype = LLAMA_FTYPE_MOSTLY_Q4_K_M;  break;
                case GGML_TYPE_Q5_K:    ftype = LLAMA_FTYPE_MOSTLY_Q5_K_M;  break;
                case GGML_TYPE_Q6_K:    ftype = LLAMA_FTYPE_MOSTLY_Q6_K;    break;
                case GGML_TYPE_IQ2_XXS: ftype = LLAMA_FTYPE_MOSTLY_IQ2_XXS; break;
                case GGML_TYPE_IQ2_XS:  ftype = LLAMA_FTYPE_MOSTLY_IQ2_XS;  break;
                case GGML_TYPE_IQ2_S:   ftype = LLAMA_FTYPE_MOSTLY_IQ2_S;   break;
                case GGML_TYPE_IQ3_XXS: ftype = LLAMA_FTYPE_MOSTLY_IQ3_XXS; break;
                case GGML_TYPE_IQ1_S:   ftype = LLAMA_FTYPE_MOSTLY_IQ1_S;   break;
                case GGML_TYPE_IQ1_M:   ftype = LLAMA_FTYPE_MOSTLY_IQ1_M;   break;
                case GGML_TYPE_IQ4_NL:  ftype = LLAMA_FTYPE_MOSTLY_IQ4_NL;  break;
                case GGML_TYPE_IQ4_XS:  ftype = LLAMA_FTYPE_MOSTLY_IQ4_XS;  break;
                case GGML_TYPE_IQ3_S:   ftype = LLAMA_FTYPE_MOSTLY_IQ3_S;   break;
                default:
                    {
                        LLAMA_LOG_WARN("%s: unknown type %s\n", __func__, ggml_type_name(type_max));
                        ftype = LLAMA_FTYPE_ALL_F32;
                    } break;
            }

            // this is a way to mark that we have "guessed" the file type
            ftype = (llama_ftype) (ftype | LLAMA_FTYPE_GUESSED);

            {
                const int kid = gguf_find_key(meta, "general.file_type");
                if (kid >= 0) {
                    ftype = (llama_ftype) gguf_get_val_u32(meta, kid);
                }
            }

            LLAMA_LOG_INFO("%s: Dumping metadata keys/values. Note: KV overrides do not apply in this output.\n", __func__);

            for (int i = 0; i < n_kv; i++) {
                const char * name           = gguf_get_key(meta, i);
                const enum gguf_type type   = gguf_get_kv_type(meta, i);
                const std::string type_name =
                    type == GGUF_TYPE_ARRAY
                    ? format("%s[%s,%d]", gguf_type_name(type), gguf_type_name(gguf_get_arr_type(meta, i)), gguf_get_arr_n(meta, i))
                    : gguf_type_name(type);

                std::string value          = gguf_kv_to_str(meta, i);
                const size_t MAX_VALUE_LEN = 40;
                if (value.size() > MAX_VALUE_LEN) {
                    value = format("%s...", value.substr(0, MAX_VALUE_LEN - 3).c_str());
                }
                replace_all(value, "\n", "\\n");

                LLAMA_LOG_INFO("%s: - kv %3d: %42s %-16s = %s\n", __func__, i, name, type_name.c_str(), value.c_str());
            }

            // print type counts
            for (auto & kv : n_type) {
                if (kv.second == 0) {
                    continue;
                }

                LLAMA_LOG_INFO("%s: - type %4s: %4d tensors\n", __func__, ggml_type_name(kv.first), kv.second);
            }
        }

        if (!llama_mmap::SUPPORTED) {
            LLAMA_LOG_WARN("%s: mmap is not supported on this platform\n", __func__);
            use_mmap = false;
        }

        this->use_mmap = use_mmap;
    }

    ~llama_model_loader() {
        if (meta) {
            gguf_free(meta);
        }
        for (auto * ctx : contexts) {
            ggml_free(ctx);
        }
    }

    template<typename T>
    typename std::enable_if<std::is_integral<T>::value, bool>::type
    get_arr_n(const std::string & key, T & result, const bool required = true) {
        const int kid = gguf_find_key(meta, key.c_str());

        if (kid < 0) {
            if (required) {
                throw std::runtime_error(format("key not found in model: %s", key.c_str()));
            }
            return false;
        }

        struct GGUFMeta::ArrayInfo arr_info =
            GGUFMeta::GKV<GGUFMeta::ArrayInfo>::get_kv(meta, kid);


        result = arr_info.length;
        return true;
    }

    template<typename T>
    typename std::enable_if<std::is_integral<T>::value, bool>::type
    get_arr_n(const enum llm_kv kid, T & result, const bool required = true) {
        return get_arr_n(llm_kv(kid), result, required);
    }

    template<typename T>
    bool get_key(const std::string & key, T & result, const bool required = true) {
        auto it = kv_overrides.find(key);

        const struct llama_model_kv_override * override =
            it != kv_overrides.end() ? &it->second : nullptr;

        const bool found = GGUFMeta::GKV<T>::set(meta, key, result, override);

        if (required && !found) {
            throw std::runtime_error(format("key not found in model: %s", key.c_str()));
        }

        return found;
    }

    template<typename T>
    bool get_key(const enum llm_kv kid, T & result, const bool required = true) {
        return get_key(llm_kv(kid), result, required);
    }

    std::string get_arch_name() const {
        return arch_name;
    }

    enum llm_arch get_arch() const {
        return llm_kv.arch;
    }

    const char * get_tensor_name(int i) const {
        return weights.at(i).tensor->name;
    }

    const llama_tensor_weight * get_weight(const char * name) const {
        for (const auto & weight : weights) {
            if (strcmp(name, weight.tensor->name) == 0) {
                return &weight;
            }
        }
        return nullptr;
    }

    const llama_tensor_weight & require_weight(const char * name) const {
        const llama_tensor_weight * weight = get_weight(name);
        if (!weight) {
            throw std::runtime_error(format("%s: tensor '%s' not found", __func__, name));
        }
        return *weight;
    }

    struct ggml_tensor * get_tensor_meta(const char * name) const {
        const auto * weight = get_weight(name);
        if (!weight) {
            return nullptr;
        }
        return weight->tensor;
    }

    struct ggml_tensor * require_tensor_meta(const char * name) const {
        struct ggml_tensor * tensor = get_tensor_meta(name);
        if (!tensor) {
            throw std::runtime_error(format("%s: tensor '%s' not found", __func__, name));
        }
        return tensor;
    }

    struct ggml_tensor * get_tensor_meta(int i) const {
        return get_tensor_meta(get_tensor_name(i));
    }

    struct ggml_tensor * create_tensor_for(struct ggml_context * ctx, const struct ggml_tensor * cur) {
        struct ggml_tensor * tensor = ggml_dup_tensor(ctx, cur);
        ggml_set_name(tensor, ggml_get_name(cur));

        n_created++;

        return tensor;
    }

    const struct ggml_tensor * check_tensor_dims(const std::string & name, const std::vector<int64_t> & ne, bool required) const {
        const struct ggml_tensor * cur = get_tensor_meta(name.c_str());

        if (cur == NULL) {
            if (!required) {
                return NULL;
            }
            throw std::runtime_error(format("%s: tensor '%s' not found", __func__, name.c_str()));
        }

        {
            bool is_ok = true;
            for (size_t i = 0; i < GGML_MAX_DIMS; ++i) {
                if ((i < ne.size() && ne[i] != cur->ne[i]) || (i >= ne.size() && cur->ne[i] != 1)) {
                    is_ok = false;
                    break;
                }
            }
            if (!is_ok) {
                throw std::runtime_error(
                        format("%s: tensor '%s' has wrong shape; expected %s, got %s",
                            __func__, name.c_str(),
                            llama_format_tensor_shape(ne).c_str(),
                            llama_format_tensor_shape(cur).c_str()));
            }
        }

        return cur;
    }

    struct ggml_tensor * create_tensor(struct ggml_context * ctx, const std::string & name, const std::vector<int64_t> & ne, bool required = true) {
        const struct ggml_tensor * cur = check_tensor_dims(name, ne, required);

        if (cur == NULL) {
            return NULL;
        }

        return create_tensor_for(ctx, cur);
    }

    struct ggml_tensor * create_tensor_as_view(struct ggml_context * ctx, struct ggml_tensor * base, const std::string & name, const std::vector<int64_t> & ne, size_t offset, bool required = true) {
        const struct ggml_tensor * cur = check_tensor_dims(name, ne, required);

        if (cur == NULL) {
            return NULL;
        }

        if (cur->type != base->type) {
            throw std::runtime_error(format("%s: tensor '%s' has wrong type; expected %s, got %s", __func__, name.c_str(), ggml_type_name(base->type), ggml_type_name(cur->type)));
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

        n_created++;

        return tensor;
    }

    void done_getting_tensors() const {
        if (n_created != n_tensors) {
            throw std::runtime_error(format("%s: wrong number of tensors; expected %d, got %d", __func__, n_tensors, n_created));
        }
    }

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
};