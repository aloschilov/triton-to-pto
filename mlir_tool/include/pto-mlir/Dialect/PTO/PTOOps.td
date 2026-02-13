//===- PTOOps.td - PTO MLIR Dialect ----------------------------*- tablegen -*-===//
//
// Minimal PTO dialect definition for Triton→PTO lowering.
// This is a *skeleton*: it defines just enough types/ops for the
// planned Triton→PTO pass and PTO-AS-like printing.
//
//===----------------------------------------------------------------------===//

include "mlir/IR/OpBase.td"
include "mlir/IR/DialectBase.td"
include "mlir/IR/AttrTypeBase.td"

//===----------------------------------------------------------------------===//
// PTO Dialect (aligns with PTOAS)
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

//===----------------------------------------------------------------------===//
// PTO Types (TableGen TypeDefs, aligned with PTOAS)
//===----------------------------------------------------------------------===//

def PTO_PtrType : TypeDef<PTO_Dialect, "Ptr"> {
  let mnemonic = "ptr";
  let parameters = (ins "mlir::Type":$elementType);
  let assemblyFormat = "`<` $elementType `>`";
}

def PTO_TensorViewType : TypeDef<PTO_Dialect, "TensorView"> {
  let mnemonic = "tensor_view";
  let parameters = (ins
    ArrayRefParameter<"int64_t">:$shape,
    "mlir::Type":$elementType
  );
  let hasCustomAssemblyFormat = 1;

  let builders = [
    TypeBuilder<(ins "int64_t":$rank, "Type":$elementType), [{
      SmallVector<int64_t, 4> shp(rank, mlir::ShapedType::kDynamic);
      return Base::get($_ctxt, shp, elementType);
    }]>
  ];

  let extraClassDeclaration = [{
    int64_t getRank() const { return getShape().size(); }
    int64_t getDimSize(unsigned idx) const { return getShape()[idx]; }
    int64_t getNumElements() const {
      int64_t num = 1;
      for (int64_t dim : getShape()) {
        if (dim == mlir::ShapedType::kDynamic) return mlir::ShapedType::kDynamic;
        num *= dim;
      }
      return num;
    }
  }];
}

def PTO_PartitionTensorViewType : TypeDef<PTO_Dialect, "PartitionTensorView"> {
  let mnemonic = "partition_tensor_view";
  let summary = "Logical partition of a tensor view";
  let parameters = (ins
    ArrayRefParameter<"int64_t">:$shape,
    "mlir::Type":$elementType
  );
  let hasCustomAssemblyFormat = 1;

  let extraClassDeclaration = [{
    int64_t getRank() const { return getShape().size(); }
    int64_t getDimSize(unsigned idx) const { return getShape()[idx]; }
    int64_t getNumElements() const {
      int64_t num = 1;
      for (int64_t dim : getShape()) {
        if (dim == mlir::ShapedType::kDynamic) return mlir::ShapedType::kDynamic;
        num *= dim;
      }
      return num;
    }
  }];
}

def PTO_TileType : TypeDef<PTO_Dialect, "Tile"> {
  let mnemonic = "tile";
  let parameters = (ins
    ArrayRefParameter<"int64_t">:$shape,
    "mlir::Type":$elementType
  );
  let hasCustomAssemblyFormat = 1;

  let extraClassDeclaration = [{
    int64_t getRank() const { return getShape().size(); }
    int64_t getDimSize(unsigned idx) const { return getShape()[idx]; }
    int64_t getNumElements() const {
      int64_t num = 1;
      for (int64_t dim : getShape()) num *= dim;
      return num;
    }
  }];
}

def PTO_TileBufType : TypeDef<PTO_Dialect, "TileBuf"> {
  let mnemonic = "tile_buf";
  let summary = "Tile buffer (physical buffer for tile data)";
  let parameters = (ins
    ArrayRefParameter<"int64_t">:$shape,
    "mlir::Type":$elementType
  );
  let hasCustomAssemblyFormat = 1;

  let extraClassDeclaration = [{
    int64_t getRank() const { return getShape().size(); }
    int64_t getDimSize(unsigned idx) const { return getShape()[idx]; }
    int64_t getNumElements() const {
      int64_t num = 1;
      for (int64_t dim : getShape()) num *= dim;
      return num;
    }
  }];
}

//===----------------------------------------------------------------------===//
// Infrastructure ops (PTOAS-style)
//===----------------------------------------------------------------------===//

def PTOMakeTensorViewOp : Op<PTO_Dialect, "make_tensor_view"> {
  let summary = "Create a tensor view over a pointer (2D)";
  let description = [{
    Wraps a pointer into a logical tensor view with given shape and strides.
  }];

  let arguments = (ins
    PTO_PtrType:$ptr,
    Index:$rows,
    Index:$cols,
    Index:$stride_row,
    Index:$stride_col
  );
  let results = (outs
    PTO_TensorViewType:$result
  );

  let assemblyFormat = "$ptr `,` $rows `,` $cols `,` $stride_row `,` $stride_col attr-dict `:` type($ptr) `->` type($result)";
}

def PTOPartitionViewOp : Op<PTO_Dialect, "partition_view"> {
  let summary = "Create a partition (slice) of a tensor view (2D)";
  let description = [{
    Creates a logical partition from a tensor view with given offsets and sizes.
  }];

  let arguments = (ins
    PTO_TensorViewType:$source,
    Index:$offset_row,
    Index:$offset_col,
    Index:$size_row,
    Index:$size_col
  );
  let results = (outs
    PTO_PartitionTensorViewType:$result
  );

  let assemblyFormat = "$source `,` $offset_row `,` $offset_col `,` $size_row `,` $size_col attr-dict `:` qualified(type($source)) `->` type($result)";
}

def PTOAllocTileOp : Op<PTO_Dialect, "alloc_tile"> {
  let summary = "Allocate a tile buffer";
  let description = [{
    Allocates a tile buffer with the given shape and element type.
  }];

  let arguments = (ins);
  let results = (outs
    PTO_TileBufType:$result
  );

  let assemblyFormat = "attr-dict `:` type($result)";
}

//===----------------------------------------------------------------------===//
// Core Ops (DPS format: ins/outs)
//===----------------------------------------------------------------------===//

def PTOTLoadOp : Op<PTO_Dialect, "tload"> {
  let summary = "PTO tile load (DPS)";
  let description = [{
    Tile load: copies from partition view into tile buffer.
  }];

  let arguments = (ins
    PTO_PartitionTensorViewType:$src,
    PTO_TileBufType:$dst
  );
  let results = (outs);

  let assemblyFormat = "`ins` `(` $src `:` type($src) `)` `outs` `(` $dst `:` type($dst) `)` attr-dict";
}

def PTOTStoreOp : Op<PTO_Dialect, "tstore"> {
  let summary = "PTO tile store (DPS)";
  let description = [{
    Tile store: copies from tile buffer into partition view.
  }];

  let arguments = (ins
    PTO_TileBufType:$src,
    PTO_PartitionTensorViewType:$dst
  );
  let results = (outs);

  let assemblyFormat = "`ins` `(` $src `:` type($src) `)` `outs` `(` $dst `:` type($dst) `)` attr-dict";
}

def PTOTMatmulOp : Op<PTO_Dialect, "tmatmul"> {
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

def PTOTAddOp : Op<PTO_Dialect, "tadd"> {
  let summary = "PTO tile add (DPS)";
  let description = [{
    Elementwise add: dst = lhs + rhs (tile buffers).
  }];

  let arguments = (ins
    PTO_TileBufType:$lhs,
    PTO_TileBufType:$rhs,
    PTO_TileBufType:$dst
  );
  let results = (outs);

  let assemblyFormat = "`ins` `(` $lhs `:` type($lhs) `,` $rhs `:` type($rhs) `)` `outs` `(` $dst `:` type($dst) `)` attr-dict";
}

def PTOTReduceSumOp : Op<PTO_Dialect, "treduce_sum"> {
  let summary = "PTO tile reduction (sum)";
  let description = [{
    Reduce all elements of a tile to a scalar sum.
  }];

  let arguments = (ins
    AnyType:$tile
  );
  let results = (outs
    AnyType:$result
  );

  let assemblyFormat = "$tile attr-dict `:` type($tile) `->` type($result)";
}

def PTOSStoreOp : Op<PTO_Dialect, "sstore"> {
  let summary = "PTO scalar store to memref element";
  let description = [{
    Store a single scalar value into a memref element at [row, col].
  }];

  let arguments = (ins
    AnyType:$value,
    AnyType:$memref,
    Index:$row,
    Index:$col
  );
  let results = (outs);

  let assemblyFormat = "$value `,` $memref `[` $row `,` $col `]` attr-dict `:` type($value) `,` type($memref)";
}

