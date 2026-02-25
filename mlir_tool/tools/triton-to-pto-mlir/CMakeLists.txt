add_executable(triton-to-pto-mlir
  triton-to-pto-mlir.cpp
)

target_link_libraries(triton-to-pto-mlir
  PRIVATE
  MLIRIR
  MLIRPass
  MLIRParser
  MLIRSupport
  TritonIR
  TritonGPUIR
  TritonAnalysis
  MLIRTensorUtils
  MLIRTensorTransforms
  MLIRTransformUtils
  MLIRDialectUtils
  MLIRArithDialect
  MLIRMathDialect
  MLIRSCFDialect
  MLIRControlFlowDialect
  MLIRFuncDialect
  MLIRTensorDialect
  MLIRTransforms
  ${PTOIR_TARGET}
  TritonToPTOPass
)

