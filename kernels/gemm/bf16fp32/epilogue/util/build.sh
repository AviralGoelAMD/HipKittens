#!/usr/bin/env bash
# build.sh - clean-build the base GEMM (tk_kernel) + named fused epilogue variants.
#
# The epilogue Makefile does header-aware incremental rebuilds (-MMD/.d tracks .cuh deps), so only
# what changed recompiles. The base GEMM Makefile (bf16fp32/) does NOT track headers, so the base
# tk_kernel .so is force-removed to guarantee a fresh build, then copied next to the bench/test
# scripts (the unfused path needs it). (Switching GPU_TARGET isn't dep-tracked -> `make clean` first.)
#
# Kernel names are the SHORT registry/module names (noop, scale, rmsnorm_scale, ...); the matching
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

# The -DCODA_OPT_* optimization switches (see the Makefile) are COMPILER FLAGS, and a flag change
# is NOT tracked by the -MMD header dependencies -- those only track #include edits. So flipping
# CODA_OPT_DEFAULTS or EXTRA_FLAGS leaves every source mtime untouched, `make` reports the module
# up to date, and you silently keep the PREVIOUS variant's binary. An A/B done that way compares a
# kernel against itself. The `rm -f` in the loop below is what prevents that; it is the same trap
# this script's own header already calls out for GPU_TARGET.
export EXTRA_FLAGS="${EXTRA_FLAGS:-}"
[ -n "$EXTRA_FLAGS" ] && echo "build.sh: EXTRA_FLAGS=$EXTRA_FLAGS" || true   # `|| true`: set -e would abort on the empty case

for k in $kernels; do
  src=$(ls bindings/gemm_"${k}"*.cpp 2>/dev/null || true)
  n=$(printf '%s\n' $src | grep -c . || true)
  [ "$n" = 1 ] || { echo "build.sh: '$k' -> $n bindings match (need exactly 1): ${src:-<none>}" >&2; exit 3; }
  kfile=$(basename "$src" .cpp); kfile="${kfile#gemm_}"   # e.g. gemm_rmsnorm_scale.cpp -> rmsnorm_scale
  mod="tk_$k"
  rm -f "$mod"*.so "$mod".d
  echo "== $mod  (bindings/gemm_$kfile.cpp) =="
  # CODA_OPT_DEFAULTS is forwarded only when the caller SET it (including to empty), so
  # `CODA_OPT_DEFAULTS= util/build.sh silu` builds the original bodies for an A/B while an
  # ordinary invocation gets the Makefile's verified-fast default.
  make KERNEL="$kfile" MODULE="$mod" GPU_TARGET="$GPU_TARGET" EXTRA_FLAGS="$EXTRA_FLAGS" \
       ${CODA_OPT_DEFAULTS+CODA_OPT_DEFAULTS="$CODA_OPT_DEFAULTS"}
done

echo "build.sh: done -> $(ls tk_*.so 2>/dev/null | tr '\n' ' ')"
