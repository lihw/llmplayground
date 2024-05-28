
#ifndef M_GGUF_H
#define M_GGUF_H

#include <common/m_defs.h>

#include <ggml/ggml.h>
#include <spdlog/spdlog.h>
#include <fmt/core.h>

#include <inttypes.h>

M_BEGIN_NAMESPACE

namespace GGUFMeta
{
    template <typename T, gguf_type gt_, T (*gfun)(const gguf_context *, const int)>
    struct GKV_Base_Type
    {
        static constexpr gguf_type gt = gt_;

        static T getter(const gguf_context *ctx, const int kid)
        {
            return gfun(ctx, kid);
        }
    };

    template <typename T>
    struct GKV_Base;

    template <>
    struct GKV_Base<bool> : GKV_Base_Type<bool, GGUF_TYPE_BOOL, gguf_get_val_bool>
    {
    };
    template <>
    struct GKV_Base<uint8_t> : GKV_Base_Type<uint8_t, GGUF_TYPE_UINT8, gguf_get_val_u8>
    {
    };
    template <>
    struct GKV_Base<uint16_t> : GKV_Base_Type<uint16_t, GGUF_TYPE_UINT16, gguf_get_val_u16>
    {
    };
    template <>
    struct GKV_Base<uint32_t> : GKV_Base_Type<uint32_t, GGUF_TYPE_UINT32, gguf_get_val_u32>
    {
    };
    template <>
    struct GKV_Base<uint64_t> : GKV_Base_Type<uint64_t, GGUF_TYPE_UINT64, gguf_get_val_u64>
    {
    };
    template <>
    struct GKV_Base<int8_t> : GKV_Base_Type<int8_t, GGUF_TYPE_INT8, gguf_get_val_i8>
    {
    };
    template <>
    struct GKV_Base<int16_t> : GKV_Base_Type<int16_t, GGUF_TYPE_INT16, gguf_get_val_i16>
    {
    };
    template <>
    struct GKV_Base<int32_t> : GKV_Base_Type<int32_t, GGUF_TYPE_INT32, gguf_get_val_i32>
    {
    };
    template <>
    struct GKV_Base<int64_t> : GKV_Base_Type<int64_t, GGUF_TYPE_INT64, gguf_get_val_i64>
    {
    };
    template <>
    struct GKV_Base<float> : GKV_Base_Type<float, GGUF_TYPE_FLOAT32, gguf_get_val_f32>
    {
    };
    template <>
    struct GKV_Base<double> : GKV_Base_Type<double, GGUF_TYPE_FLOAT64, gguf_get_val_f64>
    {
    };
    template <>
    struct GKV_Base<const char *> : GKV_Base_Type<const char *, GGUF_TYPE_STRING, gguf_get_val_str>
    {
    };

    template <>
    struct GKV_Base<std::string>
    {
        static constexpr gguf_type gt = GGUF_TYPE_STRING;

        static std::string getter(const gguf_context *ctx, const int kid)
        {
            return gguf_get_val_str(ctx, kid);
        }
    };

    struct ArrayInfo
    {
        const gguf_type gt;
        const size_t length;
        const void *data;
    };

    template <>
    struct GKV_Base<ArrayInfo>
    {
    public:
        static constexpr gguf_type gt = GGUF_TYPE_ARRAY;
        static ArrayInfo getter(const gguf_context *ctx, const int k)
        {
            return ArrayInfo{
                gguf_get_arr_type(ctx, k),
                size_t(gguf_get_arr_n(ctx, k)),
                gguf_get_arr_data(ctx, k),
            };
        }
    };

    template <typename T>
    class GKV : public GKV_Base<T>
    {
        GKV() = delete;

    public:
        static T get_kv(const gguf_context *ctx, const int k)
        {
            const enum gguf_type kt = gguf_get_kv_type(ctx, k);

            if (kt != GKV::gt)
            {
                throw std::runtime_error(format("key %s has wrong type %s but expected type %s",
                                                gguf_get_key(ctx, k), gguf_type_name(kt), gguf_type_name(GKV::gt)));
            }
            return GKV::getter(ctx, k);
        }

        static const char *override_type_to_str(const KvOverrideType ty)
        {
            switch (ty)
            {
            case KvOverrideType::BOOL:
                return "bool";
            case KvOverrideType::INT:
                return "int";
            case KvOverrideType::FLOAT:
                return "float";
            }
            return "unknown";
        }

        static bool validate_override(const KvOverrideType expected_type, const KvOverride *ovrd)
        {
            if (!ovrd)
            {
                return false;
            }
            if (ovrd->tag == expected_type)
            {
                LLAMA_LOG_INFO("%s: Using metadata override (%5s) '%s' = ",
                               __func__, override_type_to_str(ovrd->tag), ovrd->key);
                switch (ovrd->tag)
                {
                case KvOverrideType::BOOL:
                {
                    spdlog::info("%s\n", ovrd->boolValue ? "true" : "false");
                }
                break;
                case KvOverrideType::INT:
                {
                    spdlog::info("%" PRId64 "\n", ovrd->intValue);
                }
                break;
                case KvOverrideType::FLOAT:
                {
                    spdlog::info("%.6f\n", ovrd->floatValue);
                }
                break;
                default:
                    // Shouldn't be possible to end up here, but just in case...
                    throw std::runtime_error(
                        fmt::format("Unsupported attempt to override %s type for metadata key %s\n",
                               override_type_to_str(ovrd->tag), ovrd->key));
                }
                return true;
            }
            LLAMA_LOG_WARN("%s: Warning: Bad metadata override type for key '%s', expected %s but got %s\n",
                           __func__, ovrd->key, override_type_to_str(expected_type), override_type_to_str(ovrd->tag));
            return false;
        }

        template <typename OT>
        static typename std::enable_if<std::is_same<OT, bool>::value, bool>::type
        try_override(OT &target, const KvOverride *ovrd)
        {
            if (validate_override(KvOverrideType::BOOL, ovrd))
            {
                target = ovrd->boolValue;
                return true;
            }
            return false;
        }

        template <typename OT>
        static typename std::enable_if<!std::is_same<OT, bool>::value && std::is_integral<OT>::value, bool>::type
        try_override(OT &target, const KvOverride *ovrd)
        {
            if (validate_override(KvOverrideType::INT, ovrd))
            {
                target = ovrd->intValue;
                return true;
            }
            return false;
        }

        template <typename OT>
        static typename std::enable_if<std::is_floating_point<OT>::value, bool>::type
        try_override(T &target, const KvOverride *ovrd)
        {
            if (validate_override(KvOverrideType::FLOAT, ovrd))
            {
                target = ovrd->floatValue;
                return true;
            }
            return false;
        }

        template <typename OT>
        static typename std::enable_if<std::is_same<OT, std::string>::value, bool>::type
        try_override(T &target, const KvOverride *ovrd)
        {
            (void)target;
            (void)ovrd;
            if (!ovrd)
            {
                return false;
            }
            // Currently, we should never end up here so it would be a bug if we do.
            throw std::runtime_error(fmt::format("Unsupported attempt to override string type for metadata key %s\n",
                                            ovrd ? ovrd->key : "NULL"));
        }

        static bool set(const gguf_context *ctx, const int k, T &target, const KvOverride *ovrd = nullptr)
        {
            if (try_override<T>(target, ovrd))
            {
                return true;
            }
            if (k < 0)
            {
                return false;
            }
            target = get_kv(ctx, k);
            return true;
        }

        static bool set(const gguf_context *ctx, const char *key, T &target, const KvOverride *ovrd = nullptr)
        {
            return set(ctx, gguf_find_key(ctx, key), target, ovrd);
        }

        static bool set(const gguf_context *ctx, const std::string &key, T &target, const KvOverride *ovrd = nullptr)
        {
            return set(ctx, key.c_str(), target, ovrd);
        }
    };
}

M_END_NAMESPACE

#endif