/**
 * @file
 * @brief Layouts and their manipulations for shared tiles.
 */

#pragma once

#include <concepts>

namespace kittens {
namespace ducks {
/**
* @namespace st_shape
* 
* @brief A namespace for template metaprogramming with shared tile layouts.
*/
namespace st_shape {

struct st_16x16 {
    static constexpr int rows = 16;
    static constexpr int cols = 16;

    template<typename _T>
    static constexpr int bytes_per_thread() {
        if constexpr (sizeof(_T) == 2 || sizeof(_T) == 4) {
            return 16;
        } else {
            static_assert(false, "Unsupported type");
        }
    }

    template<typename _T>
    __device__ __forceinline__ static const uint32_t swizzle (int2 coord) {
        const int r = coord.x, c = coord.y;
        using T = _T;

        const uint32_t offset = sizeof(T)*(r*cols + c);

        if constexpr (sizeof(T) == 2 || sizeof(T) == 4) {
            return offset;
        } else {
            static_assert(false, "Unsupported type");
        }
        
    }
};

struct st_16x16_swizzled {
    static constexpr int rows = 16;
    static constexpr int cols = 16;

    template<typename _T>
    static constexpr int bytes_per_thread() {
        if constexpr (sizeof(_T) == 2) {
            return 4;
        } else if constexpr (sizeof(_T) == 4) {
            return 16;
        } else {
            static_assert(false, "Unsupported type");
        }
    }

    template<typename _T>
    __device__ __forceinline__ static const uint32_t swizzle (int2 coord) {
        const int r = coord.x, c = coord.y;
        using T = _T;

        const uint32_t offset = sizeof(T)*(r*cols + c);

        if constexpr (sizeof(T) == 2) {
            const int swizzle = ((offset % 512) >> 7) << 3;
            const int swizzled_offset = offset ^ swizzle;
            return swizzled_offset;
        } else if constexpr (sizeof(T) == 4) {
            return offset;
        } else {
            static_assert(false, "Unsupported type");
        }
        
    }
};

struct st_32x32 {
    static constexpr int rows = 32;
    static constexpr int cols = 32;

    template<typename _T>
    static constexpr int bytes_per_thread() {
        if constexpr (sizeof(_T) == 2 || sizeof(_T) == 4) {
            return 16;
        } else {
            static_assert(false, "Unsupported type");
        }
    }

    template<typename _T>
    __device__ __forceinline__ static const uint32_t swizzle (int2 coord) {
        const int r = coord.x, c = coord.y;
        using T = _T;

        const uint32_t offset = sizeof(T)*(r*cols + c);

        if constexpr (sizeof(T) == 2) {
            const int first_swizzle = ((offset % 1024) >> 9) << 5;
            const int second_swizzle = ((offset % 2048) >> 10) << 4;
            const int swizzled_offset = offset ^ first_swizzle ^ second_swizzle;
            return swizzled_offset;
        } else if constexpr (sizeof(T) == 4) {
            return offset;
        } else {
            static_assert(false, "Unsupported type");
        }
    }
};

struct st_16x32 {
    static constexpr int rows = 16;
    static constexpr int cols = 32;

    template<typename _T>
    static constexpr int bytes_per_thread() {
        if constexpr (sizeof(_T) == 2 || sizeof(_T) == 4) {
            return 16;
        } else {
            static_assert(false, "Unsupported type");
        }
    }

    template<typename _T>
    __device__ __forceinline__ static const uint32_t swizzle (int2 coord) {
        const int r = coord.x, c = coord.y;
        using T = _T;

        const uint32_t offset = sizeof(T)*(r*cols + c);

        if constexpr (sizeof(T) == 2) {
            const int swizzle = ((offset % 1024) >> 9) << 5;
            const int swizzled_offset = offset ^ swizzle;
            return swizzled_offset;
        } else if constexpr (sizeof(T) == 4) {
            return offset;
        } else {
            static_assert(false, "Unsupported type");
        }
    }
};

struct st_32x16 {
    static constexpr int rows = 32;
    static constexpr int cols = 16;

    template<typename _T>
    static constexpr int bytes_per_thread() {
        if constexpr (sizeof(_T) == 2 || sizeof(_T) == 4) {
            return 16;
        } else {
            static_assert(false, "Unsupported type");
        }
    }

    template<typename _T>
    __device__ __forceinline__ static const uint32_t swizzle (int2 coord) {
        const int r = coord.x, c = coord.y;
        using T = _T;

        const uint32_t offset = sizeof(T)*(r*cols + c);

        if constexpr (sizeof(T) == 2) {
            const int swizzle = ((offset % 1024) >> 9) << 4;
            const int swizzled_offset = offset ^ swizzle;
            return swizzled_offset;
        } else if constexpr (sizeof(T) == 4) {
            return offset;
        } else {
            static_assert(false, "Unsupported type");
        }
    }
};

struct st_8x32 {
    static constexpr int rows = 8;
    static constexpr int cols = 32;

    template<typename _T>
    static constexpr int bytes_per_thread() {
        if constexpr (sizeof(_T) == 2 || sizeof(_T) == 4) {
            return 16;
        } else {
            static_assert(false, "Unsupported type");
        }
    }

    template<typename _T>
    __device__ __forceinline__ static const uint32_t swizzle (int2 coord) {
        const int r = coord.x, c = coord.y;
        using T = _T;

        const uint32_t offset = sizeof(T)*(r*cols + c);

        if constexpr (sizeof(T) == 2 || sizeof(T) == 4) {
            return offset;
        } else {
            static_assert(false, "Unsupported type");
        }
    }
};

struct st_16x128 {
    static constexpr int rows = 16;
    static constexpr int cols = 128;

    template<typename _T>
    static constexpr int bytes_per_thread() {
        if constexpr (sizeof(_T) == 1) {
            return 16;
        } else {
            static_assert(false, "Unsupported type");
        }
    }

    template<typename _T>
    __device__ __forceinline__ static const uint32_t swizzle (int2 coord) {
        const int r = coord.x, c = coord.y;
        using T = _T;

        const uint32_t offset = sizeof(T)*(r*cols + c);

        if constexpr (sizeof(T) == 1) {
            const int swizzle = ((offset % (16*128)) >> 8) << 4;
            const int swizzled_offset = offset ^ swizzle;
            return swizzled_offset;
        } else {
            static_assert(false, "Unsupported type");
        }
    }
};

#ifdef KITTENS_UDNA1
/**
 * @brief 16x32 shared subtile, flat row-major (no swizzle) -- gfx1250.
 *
 * The XOR swizzle used on CDNA collides with the wave-32 `ds_load_b32` lane
 * mapping on gfx1250 and dropped half the elements when first ported. The
 * flat layout is the correctness baseline on gfx1250 and matches the
 * subtile-major addressing used by `ops/warp/memory/tile/global_to_shared.cuh`
 * when `KITTENS_UDNA1` is defined.
 */
struct st_16x32_nosz {
    static constexpr int rows = 16;
    static constexpr int cols = 32;

    template<typename _T>
    static constexpr int bytes_per_thread() {
        if constexpr (sizeof(_T) == 2 || sizeof(_T) == 4) {
            return 16;
        } else {
            static_assert(false, "Unsupported type");
        }
    }

    template<typename _T>
    __device__ __forceinline__ static const uint32_t swizzle(int2 coord) {
        const int r = coord.x, c = coord.y;
        return sizeof(_T) * (r * cols + c);
    }
};

/**
 * @brief 16x32 shared subtile with LDS bank-conflict padding -- gfx1250.
 *
 * Inserts `PAD_AMOUNT` pad elements every `PAD_INTERVAL` logical elements
 * (matching the recommended 16 B padding every 256 B for `bf16` and `f16`).
 * The padded offset is computed as `flat + (flat / PAD_INTERVAL) * PAD_AMOUNT`
 * where `flat` is the row-major byte offset divided by `sizeof(T)`.
 *
 * The shape's `swizzle()` returns the padded *byte* offset directly so
 * downstream callers do not need to know about the layout. Storage size in
 * elements is `rows*cols + (rows*cols / PAD_INTERVAL) * PAD_AMOUNT`; the
 * library accommodates this through `st::data[]` being sized off `rows*cols`
 * for the unpadded baseline -- kernels using padded layouts should allocate a
 * raw `bf16` buffer sized via `padded_elems` and view it as a flat slab.
 */
template<int PAD_INTERVAL = 128, int PAD_AMOUNT = 8>
struct st_16x32_padded {
    static constexpr int rows          = 16;
    static constexpr int cols          = 32;
    static constexpr int pad_interval  = PAD_INTERVAL;
    static constexpr int pad_amount    = PAD_AMOUNT;

    template<typename _T>
    static constexpr int bytes_per_thread() {
        if constexpr (sizeof(_T) == 2 || sizeof(_T) == 4) {
            return 16;
        } else {
            static_assert(false, "Unsupported type");
        }
    }

    // Named "swizzle" for API compat with CDNA shapes, but applies padding (not XOR swizzle).
    template<typename _T>
    __device__ __forceinline__ static const uint32_t swizzle(int2 coord) {
        const int r = coord.x, c = coord.y;
        const int flat = r * cols + c;
        const int padded_flat = flat + (flat / PAD_INTERVAL) * PAD_AMOUNT;
        return sizeof(_T) * padded_flat;
    }

    static constexpr int padded_elems(int total_elems) {
        return total_elems + (total_elems / PAD_INTERVAL) * PAD_AMOUNT;
    }
};

template<typename T> struct is_st_16x32_padded_inst : std::false_type {};
template<int I, int A> struct is_st_16x32_padded_inst<st_16x32_padded<I, A>> : std::true_type {};

template<typename T>
concept all = std::is_same_v<T, st_16x16> ||
              std::is_same_v<T, st_16x16_swizzled> ||
              std::is_same_v<T, st_32x32> ||
              std::is_same_v<T, st_16x32> ||
              std::is_same_v<T, st_32x16> ||
              std::is_same_v<T, st_8x32>  ||
              std::is_same_v<T, st_16x128> ||
              std::is_same_v<T, st_16x32_nosz> ||
              is_st_16x32_padded_inst<T>::value;
#else
template<typename T>
concept all = std::is_same_v<T, st_16x16> || 
              std::is_same_v<T, st_16x16_swizzled> || 
              std::is_same_v<T, st_32x32> || 
              std::is_same_v<T, st_16x32> || 
              std::is_same_v<T, st_32x16> || 
              std::is_same_v<T, st_8x32>  ||
              std::is_same_v<T, st_16x128>;
#endif


} // namespace st_shape
} // namespace ducks
} // namespace kittens