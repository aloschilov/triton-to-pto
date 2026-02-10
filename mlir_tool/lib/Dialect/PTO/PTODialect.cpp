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
    : Dialect(getDialectNamespace(), ctx, mlir::TypeID::get<PTODialect>()) {
  allowUnknownOperations(true);
}

void PTODialect::initialize() {
  // Ops can be registered here via addOperations<#define GET_OP_LIST ...
  // For now the convert-triton-to-pto pass creates pto.* ops by name;
  // allowUnknownOperations permits them so the pipeline and print succeed.
}

