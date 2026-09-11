// silu_precision_probe.cpp - does CODA_OPT_FAST_SILU actually cost accuracy?
//
// Answers the question directly instead of arguing from ULP datasheets: evaluate BOTH silu
// implementations on the SAME inputs, on the real gfx950 hardware, and compare each against a
// double-precision reference computed on the same device.
//
//   BASELINE : t = -x ; t = __expf(t) ; t = t + 1 ; x / t        <- IEEE correctly-rounded divide
//   FAST     : x * __builtin_amdgcn_rcpf(1 + __builtin_amdgcn_exp2f(x * -log2(e)))
//
// Reports, per input range:
//   * max ULP error of each implementation vs the double reference (ULP = Unit in the Last Place,
//     the gap between adjacent fp32 values -- the natural unit for "how wrong is this float"),
//   * how the two differ from EACH OTHER in fp32,
//   * and the number that actually decides the question: after rounding to bf16 -- which is what
//     the epilogue really stores -- how many of the two implementations' results DIFFER AT ALL.
//
// Build:  hipcc -O3 --offload-arch=gfx950 silu_precision_probe.cpp -o silu_probe
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <string>
#include <vector>
#include <algorithm>
#include <random>

__global__ void probe(const float* __restrict__ x, float* __restrict__ base,
                      float* __restrict__ fast, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = x[i];
    // --- exactly what silu_op does today (ops/activations.cuh, baseline branch) ---
    float t = v * -1.0f;
    t = __expf(t);
    t = t + 1.0f;
    base[i] = v / t;
    // --- exactly what the CODA_OPT_FAST_SILU branch does ---
    fast[i] = v * __builtin_amdgcn_rcpf(1.0f + __builtin_amdgcn_exp2f(v * -1.4426950408889634f));
}

// round-to-nearest-even fp32 -> bf16, returned as the 16 raw bits (what the epilogue stores)
static inline uint16_t to_bf16(float f) {
    uint32_t u; __builtin_memcpy(&u, &f, 4);
    if (((u >> 23) & 0xff) == 0xff && (u & 0x7fffff)) return uint16_t((u >> 16) | 0x0040);  // NaN
    uint32_t r = ((u >> 16) & 1) + 0x7fff;
    return uint16_t((u + r) >> 16);
}

// |got - ref| expressed in ULPs of the fp32 grid around ref
static double ulp_err(float got, double ref) {
    if (std::isnan(ref) || std::isnan((double)got)) return 0.0;
    if (std::isinf(ref) || ref == 0.0) return (got == (float)ref) ? 0.0 : 1e9;
    double u = std::abs((double)std::nextafterf((float)ref, HUGE_VALF) - (double)(float)ref);
    return u > 0 ? std::abs((double)got - ref) / u : 0.0;
}

struct Range { const char* name; float lo, hi; };

int main() {
    // Ranges chosen to cover what a bf16 GEMM accumulator actually produces at these shapes
    // (we measured |D| up to ~90 at 8192^3), plus the analytic danger zones at both extremes.
    const Range ranges[] = {
        {"typical GEMM output   [-90, 90]",      -90.f,   90.f},
        {"near zero             [-1, 1]",         -1.f,    1.f},
        {"small negative        [-10, -1]",      -10.f,   -1.f},
        {"large positive        [90, 1000]",      90.f, 1000.f},
        {"exp underflow edge    [-200, -80]",   -200.f,  -80.f},   // exp(-x) overflows fp32 near x=-88.7
        {"exp overflow edge     [80, 200]",       80.f,  200.f},
    };
    const int N = 1 << 22;   // 4.2M samples per range

    float *dx, *db, *df;
    hipMalloc(&dx, N * 4); hipMalloc(&db, N * 4); hipMalloc(&df, N * 4);
    std::vector<float> hx(N), hb(N), hf(N);
    std::mt19937 rng(12345);

    printf("%-34s %12s %12s %14s %14s %12s\n", "range", "base ULP", "fast ULP",
           "max |b-f|", "max rel b-f", "bf16 differ");
    printf("%s\n", std::string(112, '-').c_str());

    for (const auto& r : ranges) {
        std::uniform_real_distribution<float> d(r.lo, r.hi);
        for (int i = 0; i < N; i++) hx[i] = d(rng);
        hipMemcpy(dx, hx.data(), N * 4, hipMemcpyHostToDevice);
        hipLaunchKernelGGL(probe, dim3((N + 255) / 256), dim3(256), 0, 0, dx, db, df, N);
        hipMemcpy(hb.data(), db, N * 4, hipMemcpyDeviceToHost);
        hipMemcpy(hf.data(), df, N * 4, hipMemcpyDeviceToHost);

        double mb = 0, mf = 0, mabs = 0, mrel = 0;
        long long bf16_diff = 0, bf16_2ulp = 0;
        for (int i = 0; i < N; i++) {
            double xr  = (double)hx[i];
            double ref = xr / (1.0 + std::exp(-xr));          // double-precision oracle
            mb = std::max(mb, ulp_err(hb[i], ref));
            mf = std::max(mf, ulp_err(hf[i], ref));
            double ad = std::abs((double)hb[i] - (double)hf[i]);
            mabs = std::max(mabs, ad);
            if (hb[i] != 0.f) mrel = std::max(mrel, ad / std::abs((double)hb[i]));
            uint16_t bb = to_bf16(hb[i]), ff = to_bf16(hf[i]);
            if (bb != ff) {
                bf16_diff++;
                int gap = (int)bb - (int)ff;
                if (gap > 1 || gap < -1) bf16_2ulp++;
            }
        }
        printf("%-34s %12.3f %12.3f %14.3e %14.3e %8lld (%.4f%%)%s\n", r.name, mb, mf, mabs, mrel,
               bf16_diff, 100.0 * bf16_diff / N, bf16_2ulp ? "  <-- >1 bf16 ULP!" : "");
    }

    // Exhaustive over every bf16-representable input: 65536 values, no sampling, no luck involved.
    {
        std::vector<float> xs;
        for (uint32_t b = 0; b < 65536; b++) {
            uint32_t u = b << 16; float f; __builtin_memcpy(&f, &u, 4);
            if (std::isfinite(f)) xs.push_back(f);
        }
        int n = (int)xs.size();
        hipMemcpy(dx, xs.data(), n * 4, hipMemcpyHostToDevice);
        hipLaunchKernelGGL(probe, dim3((n + 255) / 256), dim3(256), 0, 0, dx, db, df, n);
        hipMemcpy(hb.data(), db, n * 4, hipMemcpyDeviceToHost);
        hipMemcpy(hf.data(), df, n * 4, hipMemcpyDeviceToHost);
        long long diff = 0, worst = 0; double mb = 0, mf = 0;
        for (int i = 0; i < n; i++) {
            double xr = (double)xs[i], ref = xr / (1.0 + std::exp(-xr));
            if (std::isfinite(ref)) { mb = std::max(mb, ulp_err(hb[i], ref)); mf = std::max(mf, ulp_err(hf[i], ref)); }
            uint16_t bb = to_bf16(hb[i]), ff = to_bf16(hf[i]);
            if (bb != ff) { diff++; worst = std::max(worst, (long long)std::abs((int)bb - (int)ff)); }
        }
        printf("\nEXHAUSTIVE over all %d finite bf16-representable inputs:\n", n);
        printf("  max ULP vs double  : baseline %.3f   fast %.3f\n", mb, mf);
        printf("  bf16 results differ: %lld / %d  (%.4f%%), worst gap %lld bf16 ULP\n",
               diff, n, 100.0 * diff / n, worst);
    }
    hipFree(dx); hipFree(db); hipFree(df);
    return 0;
}
