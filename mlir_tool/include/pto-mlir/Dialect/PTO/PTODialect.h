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

  mlir::Type parseType(mlir::DialectAsmParser &parser) const override;
  void printType(mlir::Type type,
                 mlir::DialectAsmPrinter &printer) const override;
};

} // namespace pto

// TableGen type declarations (useDefaultTypePrinterParser handles parse/print)
#define GET_TYPEDEF_CLASSES
#include "PTOTypeDefs.h.inc"

// Op declarations (full definitions pulled in via GET_OP_CLASSES in .cpp)
#include "PTOOps.h.inc"

#endif // PTO_MLIR_DIALECT_PTO_PTODIALECT_H
