//===- PTODialect.h - PTO MLIR Dialect ---------------------------*- C++ -*-===//
//
// Minimal dialect declaration used by the Triton→PTO MLIR tool.
//
//===----------------------------------------------------------------------===//

#ifndef PTO_MLIR_DIALECT_PTO_PTODIALECT_H
#define PTO_MLIR_DIALECT_PTO_PTODIALECT_H

#include "mlir/IR/Dialect.h"

namespace pto {

class PTODialect : public mlir::Dialect {
public:
  explicit PTODialect(mlir::MLIRContext *ctx);

  static llvm::StringRef getDialectNamespace() { return "pto"; }

  void initialize();
};

} // namespace pto

#endif // PTO_MLIR_DIALECT_PTO_PTODIALECT_H

//===- PTODialect.h - PTO MLIR Dialect ---------------------------*- C++ -*-===//
//
// Minimal dialect declaration used by the Triton→PTO MLIR tool.
//
//===----------------------------------------------------------------------===//

#ifndef PTO_MLIR_DIALECT_PTO_PTODIALECT_H
#define PTO_MLIR_DIALECT_PTO_PTODIALECT_H

#include "mlir/IR/Dialect.h"

namespace pto {

class PTODialect : public mlir::Dialect {
public:
  explicit PTODialect(mlir::MLIRContext *ctx);

  static llvm::StringRef getDialectNamespace() { return "pto"; }

  // Custom parsing/printing for types could be added here to support
  // !pto.tile<...>, !pto.memref<...>, !pto.event> exactly as in PTO-AS.
};

} // namespace pto

// Generated declarations from PTOOps.td
#include "pto-mlir/Dialect/PTO/PTODialect.h.inc"

#endif // PTO_MLIR_DIALECT_PTO_PTODIALECT_H

