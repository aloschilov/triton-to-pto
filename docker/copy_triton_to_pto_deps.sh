#!/usr/bin/env bash
# Collect only *.so needed by triton-to-pto-mlir (transitive closure under /llvm-workspace).
# Expects: LLVM_BUILD_DIR, T2P_DEPS_DIR, TRITON_TO_PTO_SOURCE_DIR

set -e

# So triton-to-pto-mlir and its deps (PTOIR, Triton) can be resolved by ldd
BUILD_LIB_DIRS=$(find "$TRITON_TO_PTO_SOURCE_DIR/mlir_tool/build" -name "*.so" -exec dirname {} \; 2>/dev/null | sort -u | tr '\n' ':' | sed 's/:$//')
export LD_LIBRARY_PATH="${LLVM_BUILD_DIR}/lib:${BUILD_LIB_DIRS}:${LD_LIBRARY_PATH}"

T2P_BIN="${TRITON_TO_PTO_SOURCE_DIR}/mlir_tool/build/tools/triton-to-pto-mlir/triton-to-pto-mlir"

copy_so() {
  local f="$1"
  [[ -f "$f" ]] || return 0
  local name
  name=$(basename "$f")
  [[ -f "${T2P_DEPS_DIR}/${name}" ]] && return 0
  cp -n "$f" "${T2P_DEPS_DIR}/" 2>/dev/null || true
  while read -r res; do
    copy_so "$res"
  done < <(ldd "$f" 2>/dev/null | awk '/=> \/llvm-workspace\// {print $3}')
}

mkdir -p "$T2P_DEPS_DIR"
if [[ -f "$T2P_BIN" ]]; then
  while read -r res; do
    copy_so "$res"
  done < <(ldd "$T2P_BIN" 2>/dev/null | awk '/=> \/llvm-workspace\// {print $3}')
fi
