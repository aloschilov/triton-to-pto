//===- PTODialect.cpp - PTO MLIR Dialect registration --------------------===//
//
// This is a minimal PTO dialect implementation suitable for a Triton→PTO
// lowering skeleton. The bulk of the op/type definitions live in the
// TableGen-generated headers from PTOOps.td.
//
//===----------------------------------------------------------------------===//

#include "pto-mlir/Dialect/PTO/PTODialect.h"
#include "mlir/IR/DialectImplementation.h"

using namespace mlir;
using namespace pto;

PTODialect::PTODialect(mlir::MLIRContext *ctx)
    : Dialect(getDialectNamespace(), ctx, mlir::TypeID::get<PTODialect>()) {}

void PTODialect::initialize() {
  // No registered ops yet; the current Triton→PTO pass only renames ops
  // by name (tt.* -> pto.*) without relying on a typed PTO dialect.
}

