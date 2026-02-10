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

// Generated dialect and op declarations/definitions.
#include "pto-mlir/Dialect/PTO/PTODialect.cpp.inc"

void PTODialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "pto-mlir/Dialect/PTO/PTOOps.cpp.inc"
      >();
}

