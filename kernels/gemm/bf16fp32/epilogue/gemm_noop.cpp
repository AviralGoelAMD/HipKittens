#include "gemm_base.cuh"
#include "pyutils/pyutils.cuh"
void dispatch_micro(micro_globals g) {
    unsigned long mem = g.dynamic_shared_memory();
    hipFuncSetAttribute((void*)micro_tk<NoOpEpilogue>, hipFuncAttributeMaxDynamicSharedMemorySize, mem);
    micro_tk<NoOpEpilogue><<<g.grid(), g.block(), mem, g.stream>>>(g, g.M, g.N, g.K);
}
PYBIND11_MODULE(TK_MODULE_NAME, m) {
    m.doc() = "tk_kernel epilogue module";
    py::bind_function<dispatch_micro>(m, "dispatch_micro", &micro_globals::a, &micro_globals::b, &micro_globals::c);
}
