# Build system for the gfx1250 GEMM optimization ladder.
#
# Invocation:
#   make KERNEL=gemm_naive               # build only `gemm_naive` as a standalone exe
#   make KERNEL=gemm_naive run M=256 N=256 K=256
#   make all                             # build every kernel in LADDER
#   make clean
#
# Requires `hipcc` from ROCm 7.2+ (clang 22+). On a host without that, run
# the make inside the `rocm/dev-ubuntu-24.04:7.2` docker image -- the
# `docker_make.sh` wrapper in this directory does that automatically.

LADDER := gemm_naive gemm_double_buf gemm_async gemm_padded gemm_split_bar gemm_segment gemm_expert gemm_tdm_arrive

KERNEL ?= gemm_naive
SRC    := $(KERNEL).cpp
BIN    := $(KERNEL).out

HIPCXX ?= hipcc
HIP_INCLUDE_DIR ?= /opt/rocm/include/hip

KITTENS_ROOT ?= ../../../..

# gfx1250 target. The hardware is wave-32 only: the WMMA builtins
# and split barriers all assume wavefrontsize32 implicitly.
HIPFLAGS := -DKITTENS_UDNA1 \
            --offload-arch=gfx1250

CXX_STD  := c++20
CPPFLAGS += -I$(KITTENS_ROOT)/include -I$(HIP_INCLUDE_DIR)
CXXFLAGS += -std=$(CXX_STD) -O3 -Wall -Wno-unused-variable -Wno-unused-local-typedef \
            -Wno-duplicate-decl-specifier -Wno-unused-value -Wno-pass-failed

M ?= 256
N ?= 256
K ?= 256
ITERS  ?= 1
VERIFY ?= 1

.PHONY: ladder run clean all

HK_HEADERS := $(shell find $(KITTENS_ROOT)/include -name '*.cuh' 2>/dev/null)

# Default target: build the currently selected KERNEL.
$(BIN): $(SRC) harness.h common.h $(HK_HEADERS)
	$(HIPCXX) $(HIPFLAGS) $(CXXFLAGS) $(CPPFLAGS) -DHARNESS_MAIN $(SRC) -o $(BIN)

all: ladder

# Loop variant -- `make ladder` rebuilds every rung.
ladder:
	@for k in $(LADDER); do $(MAKE) KERNEL=$$k || exit 1; done

run: $(BIN)
	./$(BIN) $(M) $(N) $(K) $(ITERS) $(VERIFY)

isa: $(SRC) harness.h common.h $(HK_HEADERS)
	$(HIPCXX) $(HIPFLAGS) $(CXXFLAGS) $(CPPFLAGS) -DHARNESS_MAIN --save-temps $(SRC) -o $(BIN)
	@echo "ISA: $(KERNEL)-hip-amdgcn-amd-amdhsa-gfx1250.s"

clean:
	rm -f *.out
