#pragma once
#include <cstdint>
#include <hip/hip_runtime.h>

// Per-module "current stream" for the epilogue kernels.
//
// WHY THIS EXISTS -- and why the stream is NOT a Globals field.
//
// `Globals` is passed BY VALUE into the kernel, so anything in it rides into the kernarg
// segment. A host-side stream handle has no business there (that was the review comment that
// got `hipStream_t stream` removed from all 11 structs). But the stream still has to come from
// somewhere, and it cannot be a `launch()` argument either: `bind_function` constructs Globals
// from the bound members and calls `dispatch(g)` with exactly one argument.
//
// So the stream lives beside the kernel, not inside it. Python sets it once per module:
//
//     tk_rmsnorm_scale.set_stream(torch.cuda.current_stream().cuda_stream)
//
// and every subsequent `dispatch` launches there until it is changed.
//
// THIS IS WHAT MAKES CUDA-GRAPH CAPTURE POSSIBLE. Launching on the legacy default stream (0)
// is not capturable: `torch.cuda.graph` records a specific stream, so kernels issued on the
// null stream run OUTSIDE the capture and the graph comes back EMPTY -- silently, with no
// error (observed: capture succeeds, replay writes nothing). Pointing these kernels at torch's
// capture stream is the prerequisite for graphing HipKittens at all.
namespace hkstream {

inline hipStream_t& cur() { static hipStream_t s = nullptr; return s; }   // nullptr == default stream
inline void set(hipStream_t s) { cur() = s; }

// Registers `set_stream(int)` / `get_stream()` on a pybind module. One line per binding.
// The handle is passed as a uintptr so torch's `.cuda_stream` (an int) crosses directly.
template <typename Module>
inline void bind(Module& m) {
    m.def("set_stream", [](std::uintptr_t s) { set(reinterpret_cast<hipStream_t>(s)); },
          "Launch subsequent dispatches on this HIP stream (0 = default). "
          "Required before torch.cuda.graph capture.");
    m.def("get_stream", []() { return reinterpret_cast<std::uintptr_t>(cur()); });
}

} // namespace hkstream
