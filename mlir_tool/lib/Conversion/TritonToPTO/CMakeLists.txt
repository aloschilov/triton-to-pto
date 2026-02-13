add_mlir_library(TritonToPTOPass
  TritonToPTO.cpp

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRTransforms
  MLIRArithDialect
  MLIRFuncDialect
  MLIRSCFDialect
  MLIRSCFTransforms
  PTOMLIRDialect
  TritonIR
)

target_include_directories(TritonToPTOPass
  PUBLIC
  ${CMAKE_SOURCE_DIR}/include
  ${CMAKE_BINARY_DIR}/lib/Dialect/PTO
  ${TRITON_SOURCE_DIR}/include
  ${CMAKE_BINARY_DIR}/triton/include
)

