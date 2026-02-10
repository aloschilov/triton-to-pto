add_mlir_library(TritonToPTOPass
  TritonToPTO.cpp

  LINK_LIBS PUBLIC
  MLIRIR
  PTOMLIRDialect
)

