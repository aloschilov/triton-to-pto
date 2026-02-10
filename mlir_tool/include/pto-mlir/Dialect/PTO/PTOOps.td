//===- PTOOps.td - PTO MLIR Dialect ----------------------------*- tablegen -*-===//
//
// Minimal PTO dialect definition for Triton→PTO lowering.
// This is a *skeleton*: it defines just enough types/ops for the
// planned Triton→PTO pass and PTO-AS-like printing.
//
//===----------------------------------------------------------------------===//

include "mlir/IR/OpBase.td"
include "mlir/IR/Dialect.h.td"

def PTODialect : Dialect {
  let name = "pto";
  let cppNamespace = "::pto";
}

//===----------------------------------------------------------------------===//
// Types (opaque in this skeleton; concrete typing lives in PTO ISA)
//===----------------------------------------------------------------------===//

// Tile type: !pto.tile<rowsxcolsxdtype>
def PTOTileType : DialectType<PTODialect, "tile">;

// Memref wrapper: !pto.memref<space, shape, dtype>
def PTOMemRefType : DialectType<PTODialect, "memref">;

// Event type: !pto.event
def PTOEventType : DialectType<PTODialect, "event">;

//===----------------------------------------------------------------------===//
// Core Ops (subset)
//===----------------------------------------------------------------------===//

def PTOTLoadOp : Op<PTODialect, "tload", [Pure]> {
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

  let assemblyFormat = [{
    $tile `=` $opName $memref `[` $row `,` $col `]` `:` type($memref) `,` type($row) `,` type($col) `->` type($tile)
  }];
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

  let assemblyFormat = [{
    $opName $tile `,` $memref `[` $row `,` $col `]` `:` type($tile) `,` type($memref) `,` type($row) `,` type($col)
  }];
}

def PTOTMatmulOp : Op<PTODialect, "tmatmul", [Pure]> {
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

  let assemblyFormat = [{
    $acc `=` $opName $a `,` $b `:` `(` type($a) `,` type($b) `)` `->` type($acc)
  }];
}

def PTOTAddOp : Op<PTODialect, "tadd", [Pure]> {
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

  let assemblyFormat = [{
    $dst `=` $opName $lhs `,` $rhs `:` type($lhs) `,` type($rhs) `->` type($dst)
  }];
}

