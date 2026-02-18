#!/usr/bin/env bash
# Build LLVM 19 + MLIR from llvm-project and install to a local prefix.
# PTOAS requires LLVM 19 (see PTOAS/CMakeLists.txt: "LLVM 19 必需").
#
# Prerequisites:
#   - llvm-project at LLVM_PROJECT_ROOT (default: ../../llvm-project from mlir_tool)
#   - CMake, Ninja, C++17 compiler
#
# Usage:
#   cd mlir_tool
#   ./scripts/build_llvm19.sh
#   # Then build triton-to-pto with:
#   #   cmake -S . -B build -DMLIR_DIR=<LLVM_BUILD>/lib/cmake/mlir \
#   #         -DLLVM_DIR=<LLVM_BUILD>/lib/cmake/llvm \
#   #         -DPTOAS_SOURCE_DIR=/path/to/PTOAS
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MLIR_TOOL_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${MLIR_TOOL_ROOT}/../.." && pwd)"

# llvm-project root: default sibling of triton-to-pto
LLVM_PROJECT_ROOT="${LLVM_PROJECT_ROOT:-${REPO_ROOT}/llvm-project}"
# Where to build LLVM (inside llvm-project)
LLVM_BUILD_DIR="${LLVM_BUILD_DIR:-${LLVM_PROJECT_ROOT}/build-19}"
# Where to install (optional; if unset, use build dir for install)
LLVM_INSTALL_PREFIX="${LLVM_INSTALL_PREFIX:-${LLVM_PROJECT_ROOT}/install-19}"

if [[ ! -d "${LLVM_PROJECT_ROOT}/llvm" ]]; then
  echo "ERROR: LLVM project not found at ${LLVM_PROJECT_ROOT}. Set LLVM_PROJECT_ROOT." >&2
  exit 1
fi

# Source: llvm subdir (monorepo; CMake will find ../mlir via LLVM_ENABLE_PROJECTS=mlir).
# Ensure llvm-project is on release/19.x: git fetch origin release/19.x && git checkout -f origin/release/19.x
LLVM_SOURCE="${LLVM_PROJECT_ROOT}/llvm"

echo "Configuring LLVM+MLIR in ${LLVM_BUILD_DIR} (install prefix: ${LLVM_INSTALL_PREFIX})"
cmake -S "${LLVM_SOURCE}" -B "${LLVM_BUILD_DIR}" -G Ninja \
  -DLLVM_ENABLE_PROJECTS=mlir \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="${LLVM_INSTALL_PREFIX}" \
  -DLLVM_TARGETS_TO_BUILD=host \
  -DLLVM_INSTALL_UTILS=ON \
  -DLLVM_ENABLE_ASSERTIONS=OFF

echo "Building (this may take a long time)..."
cmake --build "${LLVM_BUILD_DIR}" -j "$(sysctl -n hw.ncpu 2>/dev/null || nproc 2>/dev/null || echo 4)"

echo "Installing to ${LLVM_INSTALL_PREFIX}..."
cmake --build "${LLVM_BUILD_DIR}" --target install

echo "Done. Use for triton-to-pto:"
echo "  MLIR_DIR=${LLVM_INSTALL_PREFIX}/lib/cmake/mlir \\"
echo "  LLVM_DIR=${LLVM_INSTALL_PREFIX}/lib/cmake/llvm \\"
echo "  PTOAS_SOURCE_DIR=/path/to/PTOAS \\"
echo "  cmake -S . -B build && cmake --build build"
