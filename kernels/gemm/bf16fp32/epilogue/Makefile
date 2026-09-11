# Compiler
GPU_TARGET ?= CDNA4
KERNEL     ?= noop
MODULE     ?= tk_kernel
SRC        := bindings/gemm_$(KERNEL).cpp
DEPFILE    := $(MODULE).d

# HIP variables
HIP_INCLUDE_DIR  := /opt/rocm/include/hip

HIPCXX ?= /opt/rocm/bin/hipcc

# Compiler flags based on GPU target
ifeq ($(GPU_TARGET),CDNA2)
HIPFLAGS+= -DKITTENS_CDNA2 --offload-arch=gfx90a
else ifeq ($(GPU_TARGET),CDNA3)
HIPFLAGS+= -DKITTENS_CDNA3 --offload-arch=gfx942
else ifeq ($(GPU_TARGET),CDNA4)
HIPFLAGS+= -DKITTENS_CDNA4 --offload-arch=gfx950
endif

# Common variables and flags
CXX_STD   := c++20
ICXXFLAGS := -std=$(CXX_STD)
ICPPFLAGS := -I. -Iops -Iepilogues -I${THUNDERKITTENS_ROOT}/include -I$(HIP_INCLUDE_DIR)
ILDFLAGS  :=
ILDLIBS   :=

CXXFLAGS := -w

ICXXFLAGS += $(CXXFLAGS) 
ICPPFLAGS += $(CPPFLAGS)
ILDFLAGS  += $(LDFLAGS)
ILDLIBS   += $(LDLIBS)

PY_LDFLAGS := $(shell python3-config --ldflags | sed 's/-lcrypt//g')
ICXXFLAGS += $(PY_LDFLAGS)
ICXXFLAGS+= -I${THUNDERKITTENS_ROOT}/prototype $(shell python3 -m pybind11 --includes) -shared -fPIC -Rpass-analysis=kernel-resource-usage -MMD -MP -MF $(DEPFILE)

HIPFLAGS   += -DTK_MODULE_NAME=$(MODULE)

# --- Verified epilogue optimizations, ON BY DEFAULT -------------------------------------------
# Each entry is a compile-time switch guarding an alternative body written next to the original
# in the same source file, so the two can be A/B'd from one tree and a winner ships by flipping
# a default rather than by resurrecting a patch.
#
#   CODA_OPT_FAST_SILU   ops/activations.cuh -- silu's IEEE divide replaced by
#                        x * rcp(1 + exp2(-x*log2e)). gfx950 MI355X: silu 1.13x,
#                        rmsnorm_swiglu 1.10x, swiglu 1.07x. Accuracy measured identical
#                        (util/silu_ab.py). Does NOT touch the base GEMM mainloop.
#
# To build the ORIGINAL bodies -- for an A/B, or to rule this out when bisecting a numerics
# question -- override with an empty value:
#     make CODA_OPT_DEFAULTS= ...
#     CODA_OPT_DEFAULTS= util/build.sh silu
CODA_OPT_DEFAULTS ?= -DCODA_OPT_FAST_SILU
HIPFLAGS   += $(CODA_OPT_DEFAULTS)

# EXTRA_FLAGS is for switches under evaluation that are not yet defaults.
HIPFLAGS   += $(EXTRA_FLAGS)

.PHONY: all clean
# Default target
all: $(MODULE)$(shell python3-config --extension-suffix)

$(MODULE)$(shell python3-config --extension-suffix): $(SRC)
	$(HIPCXX) $(SRC) $(HIPFLAGS) $(ICXXFLAGS) $(ICPPFLAGS) $(ILDFLAGS) $(ILDLIBS) -o $@

# Clean target
clean:
	rm -f $(MODULE)*.so $(MODULE)*.d

# header dependencies emitted by -MMD (line is silently skipped on the first build)
-include $(DEPFILE)
