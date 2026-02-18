#!/usr/bin/env bash
set -e
cd "$TRITON_TO_PTO_SOURCE_DIR/mlir_tool"
rm -rf build
PYBIND11_DIR=$(python -m pybind11 --cmakedir)
cmake -G Ninja -S . -B build \
  -DLLVM_DIR="$LLVM_BUILD_DIR/lib/cmake/llvm" \
  -DMLIR_DIR="$LLVM_BUILD_DIR/lib/cmake/mlir" \
  -DPTOAS_SOURCE_DIR="$PTO_SOURCE_DIR" \
  -DTRITON_SOURCE_DIR="$TRITON_SOURCE_DIR" \
  -Dpybind11_DIR="$PYBIND11_DIR" \
  -DPython3_ROOT_DIR="${PY_PATH}" \
  -DPython3_EXECUTABLE="${PY_PATH}/bin/python" \
  -DMLIR_PYTHON_PACKAGE_DIR="$LLVM_BUILD_DIR/tools/mlir/python_packages/mlir_core"
