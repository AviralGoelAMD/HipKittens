/**
 * @file
 * @brief Scheduling primitives for gfx1250.
 *
 * Wraps the gfx1250 scheduling controls -- expert SCHED_MODE, wave priority,
 * `s_sleep` -- behind a `kittens::sched::*` API. The expert RAII guard is the
 * primary entry point: constructing it sets SCHED_MODE to the requested
 * value and the destructor restores the default (mode 0, normal scheduling).
 */

#pragma once

#ifdef KITTENS_UDNA1

#include "../../../common/common.cuh"

namespace kittens {
namespace sched {

/**
 * @brief gfx1250 SCHED_MODE values.
 *
 * - `normal`   : full hardware counter checks (default).
 * - `limited`  : disables `VA_VDST` and `VM_VSRC` checks. Lets VMEM<->VALU
 *                pack tighter when the programmer has manually proven
 *                independence (e.g. async loads issued ahead of WMMA).
 * - `full`     : disables most VALU/VMEM counter checks. AMD documents this
 *                as experimental / unsafe by default.
 */
enum class mode : int {
    normal          = 0,
    full            = 1,
    limited         = 2,
    limited_nostall = 2 | (1 << 4), // limited + DISABLE_VALU_STALL (bit[4])
};

// `s_setreg_b32 hwreg(MODE_REG=1, offset=4, size=5), value`
// 5 bits to cover bits [4:0] of SCHED_MODE (including DISABLE_VALU_STALL at bit[4]).
constexpr int SCHED_MODE_HWREG_SIMM16 = 1 | (4 << 6) | ((5 - 1) << 11);

/**
 * @brief Set the wave's SCHED_MODE to `m`.
 *
 * Prefer the `expert` RAII guard below -- direct use of `set_mode` skips the
 * automatic restoration on scope exit.
 */
__device__ __forceinline__ void set_mode(mode m) {
    __builtin_amdgcn_s_setreg(SCHED_MODE_HWREG_SIMM16, static_cast<unsigned>(m));
}

/**
 * @brief RAII guard that switches SCHED_MODE for a scope.
 *
 * @code
 *   {
 *     kittens::sched::expert _sched;          // limited expert in scope
 *     for (int k = 0; k < K; ++k) { ... }
 *   }                                          // mode restored to normal
 * @endcode
 *
 * The default ctor enables `mode::limited`, which is the recommended setting
 * for hand-scheduled producer/consumer kernels. Use `expert{mode::full}` when
 * you have verified there are no inter-class hazards.
 */
struct expert {
    __device__ __forceinline__ expert(mode m = mode::limited) { set_mode(m); }
    __device__ __forceinline__ ~expert()                       { set_mode(mode::normal); }

    expert(const expert&)            = delete;
    expert& operator=(const expert&) = delete;
};

/**
 * @brief Bump the wave priority field.
 *
 * Lowers to `s_setprio N`. N is in [0,3]; higher = more SIMD slots.
 * Template-parameterized because the builtin requires a constant.
 */
template<int PRIO>
__device__ __forceinline__ void set_priority() {
    static_assert(PRIO >= 0 && PRIO <= 3, "s_setprio takes a 2-bit constant");
    __builtin_amdgcn_s_setprio(static_cast<short>(PRIO));
}

/**
 * @brief Bump the priority of every wave in this WG on the same SIMD by N.
 *
 * Lowers to `s_setprio_inc_wg N`. Useful when one warp wants to nudge
 * the entire WG forward (e.g. a producer warp boosting all WG members
 * once the prologue is past).
 */
template<int DELTA>
__device__ __forceinline__ void boost_priority() {
    static_assert(DELTA >= 0 && DELTA <= 3, "s_setprio_inc_wg takes a 2-bit constant");
    __builtin_amdgcn_s_setprio_inc_wg(static_cast<short>(DELTA));
}

/**
 * @brief Sleep the wave for N cycles.
 *
 * Lowers to `s_sleep N`. N is a small immediate (0..15 on gfx12).
 */
template<int N>
__device__ __forceinline__ void sleep() {
    __builtin_amdgcn_s_sleep(N);
}

/**
 * @brief Compiler-only scheduling fence.
 *
 * Tells the LLVM scheduler not to reorder instructions across this point
 * but emits no hardware op. Useful when constraining the compiler's WMMA
 * burst grouping without paying a runtime barrier.
 */
__device__ __forceinline__ void compiler_fence() {
    __builtin_amdgcn_sched_barrier(0);
}

} // namespace sched
} // namespace kittens

#endif // KITTENS_UDNA1
