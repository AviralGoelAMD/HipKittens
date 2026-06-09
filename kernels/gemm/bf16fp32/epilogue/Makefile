# Compiler
GPU_TARGET ?= CDNA4
KERNEL     ?= noop
MODULE     ?= tk_kernel
SRC        := bindings/gemm_$(KERNEL).cpp
DEPFILE    := $(MODULE).d

# HIP variables
ROCM_INSTALL_DIR := $(ROCM_PATH)
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
ICPPFLAGS := -I. -Iops -I${THUNDERKITTENS_ROOT}/include -I$(HIP_INCLUDE_DIR)
ILDFLAGS  :=
ILDLIBS   :=

CXXFLAGS ?= -Wall -Wextra
CXXFLAGS := -w

ICXXFLAGS += $(CXXFLAGS) 
ICPPFLAGS += $(CPPFLAGS)
ILDFLAGS  += $(LDFLAGS)
ILDLIBS   += $(LDLIBS)

PY_LDFLAGS := $(shell python3-config --ldflags | sed 's/-lcrypt//g')
ICXXFLAGS += $(PY_LDFLAGS)
ICXXFLAGS+= -I${THUNDERKITTENS_ROOT}/prototype $(shell python3 -m pybind11 --includes) -shared -fPIC -Rpass-analysis=kernel-resource-usage -MMD -MP -MF $(DEPFILE)

HIPFLAGS   += -DTK_MODULE_NAME=$(MODULE)

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
