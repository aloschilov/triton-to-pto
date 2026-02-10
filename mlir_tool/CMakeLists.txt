cmake_minimum_required(VERSION 3.16)

project(TritonToPTO_MLIR LANGUAGES CXX C)

# This CMake project is a *skeleton* for a standalone MLIR-based
# Triton→PTO translator. It assumes LLVM + MLIR are available and
# configured with a CMake package config.
#
# Typical usage (from this directory):
#   cmake -S . -B build -DMLIR_DIR=/path/to/llvm/lib/cmake/mlir
#   cmake --build build
#
# The actual LLVM/MLIR installation paths and Triton dialect integration
# will depend on your environment.

set(CMAKE_CXX_STANDARD 17 CACHE STRING "C++ standard to conform to")
set(CMAKE_CXX_STANDARD_REQUIRED YES)
set(CMAKE_CXX_EXTENSIONS NO)

find_package(MLIR REQUIRED CONFIG)

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
include(AddLLVM)
include(AddMLIR)

message(STATUS "Found MLIR: ${MLIR_PACKAGE_VERSION}")

include_directories(
  ${MLIR_INCLUDE_DIRS}
)
list(APPEND LLVM_TABLEGEN_FLAGS "-I${MLIR_INCLUDE_DIRS}")
add_definitions(${MLIR_CMAKE_CXX_FLAGS})

add_subdirectory(lib)
add_subdirectory(tools)

# Optional lit/FileCheck-based test suite for the MLIR tool.
# This assumes that LLVM/MLIR's testing utilities and FileCheck are
# available in the environment.
find_package(Python3 COMPONENTS Interpreter)

if (Python3_Interpreter_FOUND)
  add_lit_testsuite(check-mlir-tool "MLIR Tool Tests"
    ${CMAKE_CURRENT_SOURCE_DIR}/test
    DEPENDS triton-to-pto-mlir
  )
endif()

