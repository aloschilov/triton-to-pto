//===- PTODialect.td - PTO dialect definition --------------*- tablegen -*-===//
//
// Defines the MLIR PTO dialect (aligns with PTOAS).
//
//===----------------------------------------------------------------------===//

#ifndef PTO_MLIR_DIALECT_PTO_PTODIALECT_TD
#define PTO_MLIR_DIALECT_PTO_PTODIALECT_TD

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// PTO Dialect
//===----------------------------------------------------------------------===//

def PTO_Dialect : Dialect {
  let summary = "PTO dialect for tile operations";
  let description = [{
    The PTO dialect describes tile load/store and compute operations
    in PTO-AS style (ins/outs DPS, partition views, tile buffers).
  }];

  let name = "pto";
  let cppNamespace = "::pto";

  let useDefaultTypePrinterParser = 1;
  let useDefaultAttributePrinterParser = 1;
}

#endif // PTO_MLIR_DIALECT_PTO_PTODIALECT_TD
