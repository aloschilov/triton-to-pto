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
find_package(LLVM REQUIRED CONFIG)

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
include(AddLLVM)
include(AddMLIR)

# ---- Triton MLIR subproject -------------------------------------------------

# Location of the Triton source tree that provides the Triton MLIR dialects.
# Default: sibling "triton" next to triton-to-pto (../../triton from mlir_tool).
set(TRITON_SOURCE_DIR
  "${CMAKE_CURRENT_SOURCE_DIR}/../../triton"
  CACHE PATH "Path to the Triton source tree used for MLIR dialects")

# Configure Triton so that only the C++/MLIR pieces we need are built.
set(TRITON_BUILD_PYTHON_MODULE OFF CACHE BOOL
  "Build Python Triton bindings (disabled for embedded MLIR tool)")
set(TRITON_BUILD_PROTON OFF CACHE BOOL
  "Build the Triton Proton profiler (disabled for embedded MLIR tool)")
set(TRITON_BUILD_UT OFF CACHE BOOL
  "Build C++ Triton unit tests (disabled for embedded MLIR tool)")
set(TRITON_CODEGEN_BACKENDS "nvidia" CACHE STRING
  "Enable Triton codegen backends (enable 'nvidia' so Triton GPU/NVIDIA dialects are fully generated)")

# Use TRITON_SOURCE_DIR as-is. For LLVM 19, point it to a Triton checkout
# compatible with LLVM 19 (e.g. a release branch or tag that supports it).
if (EXISTS "${TRITON_SOURCE_DIR}/CMakeLists.txt")
  add_subdirectory(${TRITON_SOURCE_DIR} ${CMAKE_BINARY_DIR}/triton)
else()
  message(FATAL_ERROR
    "TRITON_SOURCE_DIR ('${TRITON_SOURCE_DIR}') does not contain a Triton CMake project. "
    "Set TRITON_SOURCE_DIR to a valid Triton checkout (for LLVM 19, use an LLVM-19-compatible version).")
endif()

message(STATUS "Found MLIR: ${MLIR_PACKAGE_VERSION}")

include_directories(
  ${LLVM_INCLUDE_DIRS}
  ${MLIR_INCLUDE_DIRS}
  ${TRITON_SOURCE_DIR}/include
  ${CMAKE_BINARY_DIR}/triton/include
)
foreach(_inc ${MLIR_INCLUDE_DIRS})
  list(APPEND LLVM_TABLEGEN_FLAGS "-I${_inc}")
endforeach()
add_definitions(${MLIR_CMAKE_CXX_FLAGS})

# ---- PTOAS: PTO dialect single source of truth ----
# Option A: PTOAS as subproject (set PTOAS_SOURCE_DIR to PTOAS repo root).
# Option B: Installed PTOAS (find_package(PTOAS)).
set(PTOAS_SOURCE_DIR "" CACHE PATH "Path to PTOAS source tree (for in-tree PTO dialect)")
if(PTOAS_SOURCE_DIR AND EXISTS "${PTOAS_SOURCE_DIR}/CMakeLists.txt")
  add_subdirectory(${PTOAS_SOURCE_DIR} ${CMAKE_BINARY_DIR}/ptoas)
  set(PTOIR_TARGET PTOIR)
  message(STATUS "Using PTOAS dialect from subproject: ${PTOAS_SOURCE_DIR}")
elseif(PTOAS_SOURCE_DIR)
  message(FATAL_ERROR "PTOAS_SOURCE_DIR is set but does not contain CMakeLists.txt: ${PTOAS_SOURCE_DIR}")
else()
  find_package(PTOAS QUIET CONFIG)
  if(PTOAS_FOUND)
    set(PTOIR_TARGET PTOAS::PTOIR)
    message(STATUS "Using PTOAS dialect from find_package(PTOAS)")
  else()
    message(FATAL_ERROR
      "PTO dialect is provided by PTOAS. Set PTOAS_SOURCE_DIR to the PTOAS repo root, "
      "or install PTOAS and set PTOAS_DIR to its lib/cmake/PTOAS.")
  endif()
endif()

add_subdirectory(lib)
add_subdirectory(tools)

# Optional lit/FileCheck-based test suite for the MLIR tool.
# Paths to triton-to-pto-mlir and FileCheck are passed via env so tests do not rely on PATH.
find_package(Python3 COMPONENTS Interpreter)

if (Python3_Interpreter_FOUND)
  # Lit executable: prefer LLVM_EXTERNAL_LIT (e.g. -DLLVM_EXTERNAL_LIT=/usr/local/bin/lit)
  if (LLVM_EXTERNAL_LIT)
    set(LIT_EXECUTABLE "${LLVM_EXTERNAL_LIT}")
  else()
    find_program(LIT_EXECUTABLE NAMES lit llvm-lit)
  endif()

  # FileCheck: infer from MLIR install prefix (same as LLVM when using an LLVM install)
  get_filename_component(_mlir_cmake_parent "${MLIR_DIR}" DIRECTORY)
  get_filename_component(_llvm_lib_parent "${_mlir_cmake_parent}" DIRECTORY)
  get_filename_component(_llvm_install_prefix "${_llvm_lib_parent}" DIRECTORY)
  find_program(FILECHECK_EXECUTABLE FileCheck
    PATHS "${_llvm_install_prefix}/bin"
    NO_DEFAULT_PATH)
  if (NOT FILECHECK_EXECUTABLE)
    find_program(FILECHECK_EXECUTABLE FileCheck)
  endif()

  if (LIT_EXECUTABLE AND FILECHECK_EXECUTABLE)
    add_custom_target(check-mlir-tool
      COMMAND ${CMAKE_COMMAND} -E env
        "TRITON_TO_PTO_MLIR=$<TARGET_FILE:triton-to-pto-mlir>"
        "LIT_FILECHECK=${FILECHECK_EXECUTABLE}"
        "${LIT_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/test"
      DEPENDS triton-to-pto-mlir
      WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
      COMMENT "Running MLIR Tool Tests"
      USES_TERMINAL
    )
  else()
    if (NOT LIT_EXECUTABLE)
      message(STATUS "check-mlir-tool disabled: lit not found (set LLVM_EXTERNAL_LIT or add lit to PATH)")
    endif()
    if (NOT FILECHECK_EXECUTABLE)
      message(STATUS "check-mlir-tool disabled: FileCheck not found (add LLVM bin to PATH)")
    endif()
  endif()
endif()

