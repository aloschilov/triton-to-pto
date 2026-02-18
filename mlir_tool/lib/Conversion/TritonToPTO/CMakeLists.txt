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
  ${PTOIR_TARGET}
  TritonIR
)

target_include_directories(TritonToPTOPass
  PUBLIC
  ${TRITON_SOURCE_DIR}/include
  ${CMAKE_BINARY_DIR}/triton/include
)

