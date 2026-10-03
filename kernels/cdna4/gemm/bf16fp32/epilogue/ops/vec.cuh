#pragma once
#include "epilogue_args.cuh"

// Multiply every accumulator value by the scalar alpha, read on the GPU from a 1-element fp32
// tensor (g.alpha). Because it is read at run time, alpha can change between launches (or inside
// a captured graph) without recompiling.
template<typename Globals, typename Accum>
__device__ inline void apply_scale(const Globals& g, Accum& C) {
    const float a = g.alpha[{0, 0, 0, 0}];
    mul(C[0][0], C[0][0], a); mul(C[0][1], C[0][1], a);
    mul(C[1][0], C[1][0], a); mul(C[1][1], C[1][1], a);
}
