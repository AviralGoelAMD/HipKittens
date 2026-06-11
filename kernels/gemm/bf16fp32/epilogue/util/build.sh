#!/usr/bin/env bash
# build.sh - clean-build the base GEMM (tk_kernel) + named fused epilogue variants.
#
# Removes stale .so first (the Makefile depends only on the .cpp, NOT the .cuh headers,
# so it will NOT rebuild on a header edit -> stale .so silently masks fixes; see plan [C12c]),
# and copies the base tk_kernel .so next to the bench/test scripts (the unfused path needs it).
#
# Kernel names are the SHORT registry/module names (noop, scale, k5, ...); the matching
# bindings/gemm_<name>*.cpp is found by glob, and the module is named tk_<name>.
#
# Run INSIDE the kreb container (needs hipcc + kreb env), from anywhere:
#   util/build.sh                 # base only (tk_kernel)
#   util/build.sh scale,rmsnorm_scale   # base + tk_scale + tk_rmsnorm_scale
#   util/build.sh --no-base noop  # skip base, build tk_noop only
#   GPU_TARGET=CDNA3 util/build.sh rmsnorm_scale   # gfx942 instead of the CDNA4 default
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EPI="$(dirname "$SCRIPT_DIR")"                 # .../bf16fp32/epilogue
BF16="$(dirname "$EPI")"                        # .../bf16fp32  (base GEMM Makefile lives here)
HKROOT="$(cd "$EPI/../../../.." && pwd)"        # repo root (derived from this script's location)
export THUNDERKITTENS_ROOT="$HKROOT"            # FORCE it -- a stale inherited value (e.g. $HOME) breaks the include path
GPU_TARGET="${GPU_TARGET:-CDNA4}"

base=1
kernels=""
for a in "$@"; do
  case "$a" in
    --no-base)      base=0 ;;
    --gpu-target=*) GPU_TARGET="${a#*=}" ;;
    -*)             echo "build.sh: unknown flag '$a'" >&2; exit 2 ;;
    *)              kernels="$kernels ${a//,/ }" ;;
  esac
done

cd "$EPI"
echo "build.sh: THUNDERKITTENS_ROOT=$THUNDERKITTENS_ROOT  GPU_TARGET=$GPU_TARGET"

if [ "$base" = 1 ]; then
  echo "== base GEMM (tk_kernel) =="
  ( cd "$BF16" && rm -f tk_kernel*.so && make GPU_TARGET="$GPU_TARGET" )
  cp "$BF16"/tk_kernel.cpython*.so "$EPI"/
fi

for k in $kernels; do
  src=$(ls bindings/gemm_"${k}"*.cpp 2>/dev/null || true)
  n=$(printf '%s\n' $src | grep -c . || true)
  [ "$n" = 1 ] || { echo "build.sh: '$k' -> $n bindings match (need exactly 1): ${src:-<none>}" >&2; exit 3; }
  kfile=$(basename "$src" .cpp); kfile="${kfile#gemm_}"   # e.g. gemm_rmsnorm_scale.cpp -> rmsnorm_scale
  mod="tk_$k"
  echo "== $mod  (bindings/gemm_$kfile.cpp) =="
  rm -f "$mod"*.so
  make KERNEL="$kfile" MODULE="$mod" GPU_TARGET="$GPU_TARGET"
done

echo "build.sh: done -> $(ls tk_*.so 2>/dev/null | tr '\n' ' ')"
