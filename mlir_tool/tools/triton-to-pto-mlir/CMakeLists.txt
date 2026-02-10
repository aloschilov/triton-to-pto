add_executable(triton-to-pto-mlir
  triton-to-pto-mlir.cpp
)

target_link_libraries(triton-to-pto-mlir
  PRIVATE
  MLIRIR
  MLIRParser
  MLIRSupport
  PTOMLIRDialect
  TritonToPTOPass
)

