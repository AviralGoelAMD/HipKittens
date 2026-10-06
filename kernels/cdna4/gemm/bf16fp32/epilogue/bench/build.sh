#!/usr/bin/env bash
# Build everything the benchmark suite needs, in parallel:
#   - the epilogue modules tk_<name> (in the epilogue directory),
#   - upstream's base GEMM tk_kernel (kernels/cdna4/gemm/bf16fp32, for ab_noop_vs_base.py),
#   - gqa_causal once per sequence length, as bench/attn_build/tk_attn_H<H>_KV<H_KV>_D<Dh>_N<M>*.so (for
#     bench_layer.py). Its shape is fixed at compile time. kernel.cpp names its module tk_kernel; the
#     -Dtk_kernel=<name> flag renames it per build, because Python returns the first-loaded module for
#     every later load of the same module name (measured: the N=4096 load ran the N=2048 kernel).
# Logs: bench/build_logs/<target>.log.
# Usage: bench/build.sh [--config pr1] [--M 2048,4096,8192] [-j JOBS]
set -euo pipefail

BENCH=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
EPI=$(dirname "$BENCH")
BASE=$(dirname "$EPI")
ROOT=$(cd "$BENCH/../../../../../.." && pwd)
ATTN="$ROOT/kernels/cdna4/attn/gqa_causal"
LOGS="$BENCH/build_logs"
export THUNDERKITTENS_ROOT="$ROOT"      # kreb-activate may export a different root

CONFIG=pr1
MS=2048,4096,8192
JOBS=$(nproc)
while [ "$#" -gt 0 ]; do
  case "$1" in
    --config) CONFIG=$2; shift 2 ;;
    --M) MS=$2; shift 2 ;;
    -j) JOBS=$2; shift 2 ;;
    *) echo "usage: $0 [--config NAME] [--M M1,M2,...] [-j JOBS]" >&2; exit 2 ;;
  esac
done

# Attention head counts from CONFIGS in common.py, read without importing it (no torch needed to build).
read -r H H_KV DH < <(python3 - "$BENCH/common.py" "$CONFIG" <<'PY'
import ast, sys
tree = ast.parse(open(sys.argv[1]).read())
node = next(n.value for n in tree.body if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "CONFIGS")
cfg = eval(compile(ast.Expression(node), "CONFIGS", "eval"), {"__builtins__": {}, "dict": dict})[sys.argv[2]]
print(cfg["H"], cfg["H_KV"], cfg["Dh"])
PY
) || true
[ -n "${DH:-}" ] || { echo "unknown config '$CONFIG' (see CONFIGS in bench/common.py)" >&2; exit 2; }

MODULES="noop silu scale residual_add rmsnorm_scale swiglu rmsnorm_swiglu rope rmsnorm_rope partialrms residual_rms rms_reduce"

build_one() {   # build_one <kind> <arg>; kind: epi | base | attn
  local kind=$1 arg=$2 log out
  case "$kind" in
    epi)  log="$LOGS/tk_$arg.log"
          make -C "$EPI" KERNEL="$arg" > "$log" 2>&1 ;;
    base) log="$LOGS/base.log"
          make -C "$BASE" > "$log" 2>&1 ;;
    attn) name="tk_attn_H${H}_KV${H_KV}_D${DH}_N${arg}"
          log="$LOGS/$name.log"
          mkdir -p "$BENCH/attn_build" && rm -f "$BENCH/attn_build/$name".*so
          make -C "$ATTN" GPU_TARGET=CDNA4 ATTN_B=1 ATTN_N="$arg" ATTN_H="$H" ATTN_H_KV="$H_KV" \
               ATTN_D="$DH" EXTRA_CPPFLAGS="-Dtk_kernel=$name" TARGET="$BENCH/attn_build/$name" > "$log" 2>&1 ;;
  esac && echo "built $kind $arg" || { echo "FAILED $kind $arg (log: $log)"; tail -20 "$log"; return 1; }
}
export -f build_one
export EPI BASE ATTN BENCH LOGS H H_KV DH

mkdir -p "$LOGS"
start=$(date +%s)
{
  for m in $MODULES; do echo "epi $m"; done
  echo "base -"
  for m in ${MS//,/ }; do echo "attn $m"; done
} | xargs -P "$JOBS" -L 1 bash -c 'build_one "$@"' _
echo "build done in $(( $(date +%s) - start )) s (config $CONFIG: H=$H H_KV=$H_KV Dh=$DH, M=$MS, -j $JOBS)"
