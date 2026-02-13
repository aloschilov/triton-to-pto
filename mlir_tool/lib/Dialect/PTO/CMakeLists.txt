set(LLVM_TARGET_DEFINITIONS
  ${CMAKE_SOURCE_DIR}/include/pto-mlir/Dialect/PTO/PTOOps.td
)

mlir_tablegen(PTOOps.h.inc -gen-op-decls)
mlir_tablegen(PTOOps.cpp.inc -gen-op-defs)
mlir_tablegen(PTODialect.h.inc -gen-dialect-decls -dialect=pto)
mlir_tablegen(PTODialect.cpp.inc -gen-dialect-defs -dialect=pto)
mlir_tablegen(PTOTypeDefs.h.inc -gen-typedef-decls -typedefs-dialect=pto)
mlir_tablegen(PTOTypeDefs.cpp.inc -gen-typedef-defs -typedefs-dialect=pto)

add_public_tablegen_target(PTOMLIRTableGen)

add_mlir_dialect_library(PTOMLIRDialect
  PTODialect.cpp

  DEPENDS
  PTOMLIRTableGen

  LINK_LIBS PUBLIC
  MLIRIR
)

target_include_directories(PTOMLIRDialect
  PUBLIC
  ${CMAKE_SOURCE_DIR}/include
  ${CMAKE_BINARY_DIR}/lib/Dialect/PTO
)

