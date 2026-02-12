//===- PTOOps.td - PTO MLIR Dialect ----------------------------*- tablegen -*-===//
//
// Minimal PTO dialect definition for Triton→PTO lowering.
// This is a *skeleton*: it defines just enough types/ops for the
// planned Triton→PTO pass and PTO-AS-like printing.
//
//===----------------------------------------------------------------------===//

include "mlir/IR/OpBase.td"
include "mlir/IR/DialectBase.td"

def PTODialect : Dialect {
  let name = "pto";
  let cppNamespace = "::pto";
}

//===----------------------------------------------------------------------===//
// Core Ops (subset)
//===----------------------------------------------------------------------===//

def PTOTLoadOp : Op<PTODialect, "tload"> {
  let summary = "PTO tile load from memref";
  let description = [{
    Synchronous tile load operation, mirroring TLOAD from the PTO ISA.
  }];

  let arguments = (ins
    AnyType:$memref,
    Index:$row,
    Index:$col
  );
  let results = (outs
    AnyType:$tile
  );

  let assemblyFormat = "$memref `[` $row `,` $col `]` attr-dict `:` type($memref) `->` type($tile)";
}

def PTOTStoreOp : Op<PTODialect, "tstore"> {
  let summary = "PTO tile store to memref";
  let description = [{
    Synchronous tile store operation, mirroring TSTORE from the PTO ISA.
  }];

  let arguments = (ins
    AnyType:$tile,
    AnyType:$memref,
    Index:$row,
    Index:$col
  );
  let results = (outs);

  let assemblyFormat = "$tile `,` $memref `[` $row `,` $col `]` attr-dict `:` type($tile) `,` type($memref)";
}

def PTOTMatmulOp : Op<PTODialect, "tmatmul"> {
  let summary = "PTO tile matmul";
  let description = [{
    Matrix multiply (GEMM) producing an accumulator/output tile.
  }];

  let arguments = (ins
    AnyType:$a,
    AnyType:$b
  );
  let results = (outs
    AnyType:$acc
  );
}

def PTOTAddOp : Op<PTODialect, "tadd"> {
  let summary = "PTO tile add";
  let description = [{
    Elementwise add of two tiles.
  }];

  let arguments = (ins
    AnyType:$lhs,
    AnyType:$rhs
  );
  let results = (outs
    AnyType:$dst
  );

  // Parse both operand types so TableGen can build the op; result type matches.
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($dst)";
}

