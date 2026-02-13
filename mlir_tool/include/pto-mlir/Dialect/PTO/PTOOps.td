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
include "mlir/Interfaces/SideEffectInterfaces.td"

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
// Infrastructure ops (PTOAS-aligned)
//===----------------------------------------------------------------------===//

def PTOGetBlockIdxOp : Op<PTO_Dialect, "get_block_idx", [Pure]> {
  let summary = "Get the current block index (core ID).";
  let description = [{
    Returns the linear index of the current compute unit (CUBE Core) within
    the task. The return value is in the range [0, BlockNum - 1].
  }];
  let arguments = (ins);
  let results = (outs I64:$result);
  let assemblyFormat = "attr-dict";
}

def PTOMakeTensorViewOp : Op<PTO_Dialect, "make_tensor_view", [AttrSizedOperandSegments]> {
  let summary = "Wrap a pointer as a tensor_view descriptor (PTOAS-aligned)";
  let description = [{
    Creates a logical tensor view over a pointer with given shape and strides.
    Uses variadic shape/strides to support arbitrary rank.
  }];

  let arguments = (ins
    PTO_PtrType:$ptr,
    Variadic<Index>:$shape,
    Variadic<Index>:$strides
  );
  let results = (outs PTO_TensorViewType:$result);

  let assemblyFormat = [{
    $ptr `,` `shape` `=` `[` $shape `]` `strides` `=` `[` $strides `]`
    attr-dict `:` type($ptr) `->` qualified(type($result))
  }];
}

def PTOPartitionViewOp : Op<PTO_Dialect, "partition_view", [AttrSizedOperandSegments]> {
  let summary = "Partition a tensor view into a logical slice (PTOAS-aligned)";
  let description = [{
    Captures a specific calculation region from a large view.
    Carries offsets (where to read) and sizes (how much to read).
  }];

  let arguments = (ins
    PTO_TensorViewType:$source,
    Variadic<Index>:$offsets,
    Variadic<Index>:$sizes
  );
  let results = (outs PTO_PartitionTensorViewType:$result);

  let assemblyFormat = [{
    $source `,` `offsets` `=` `[` $offsets `]` `,` `sizes` `=` `[` $sizes `]`
    attr-dict `:` qualified(type($source)) `->` qualified(type($result))
  }];
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
// Core Ops (DPS format: ins/outs, PTOAS-aligned)
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

def PTOTExpandsOp : Op<PTO_Dialect, "texpands"> {
  let summary = "Broadcast scalar into tile (DPS)";
  let description = [{
    Fills every element of the destination tile buffer with the given scalar.
  }];
  let arguments = (ins AnyType:$scalar, PTO_TileBufType:$dst);
  let results = (outs);
  let assemblyFormat = [{
    `ins` `(` $scalar `:` type($scalar) `)`
    `outs` `(` $dst `:` qualified(type($dst)) `)` attr-dict
  }];
}

def PTOTRowSumOp : Op<PTO_Dialect, "trowsum"> {
  let summary = "Reduce each row by summing across columns (DPS)";
  let description = [{
    For each row i: dst[i,0] = sum_j src[i,j]. Requires a tmp tile (same shape
    as src) for intermediate storage.
  }];
  let arguments = (ins PTO_TileBufType:$src, PTO_TileBufType:$tmp, PTO_TileBufType:$dst);
  let results = (outs);
  let assemblyFormat = [{
    `ins` `(` $src `,` $tmp `:` qualified(type($src)) `,` qualified(type($tmp)) `)`
    `outs` `(` $dst `:` qualified(type($dst)) `)` attr-dict
  }];
}

def PTOTColSumOp : Op<PTO_Dialect, "tcolsum"> {
  let summary = "Reduce each column by summing across rows (DPS)";
  let description = [{
    For each col j: dst[0,j] = sum_i src[i,j]. Requires a tmp tile for
    intermediate storage.
  }];
  let arguments = (ins PTO_TileBufType:$src, PTO_TileBufType:$tmp, PTO_TileBufType:$dst);
  let results = (outs);
  let assemblyFormat = [{
    `ins` `(` $src `,` $tmp `:` qualified(type($src)) `,` qualified(type($tmp)) `)`
    `outs` `(` $dst `:` qualified(type($dst)) `)` attr-dict
  }];
}

def PTOTGetValOp : Op<PTO_Dialect, "tgetval"> {
  let summary = "Read scalar from tile at offset";
  let description = [{
    Reads a single element from tile-like src at the given linear offset.
  }];
  let arguments = (ins PTO_TileBufType:$src, Index:$offset);
  let results = (outs AnyType:$dst);
  let assemblyFormat = [{
    `ins` `(` $src `,` $offset `:` qualified(type($src)) `,` type($offset) `)`
    `outs` `:` type($dst) attr-dict
  }];
}

def PTOTSetValOp : Op<PTO_Dialect, "tsetval"> {
  let summary = "Write scalar into tile at offset (DPS)";
  let description = [{
    Writes a scalar value into a single element of dst at the given linear
    offset.
  }];
  let arguments = (ins PTO_TileBufType:$dst, Index:$offset, AnyType:$val);
  let results = (outs);
  let assemblyFormat = [{
    `ins` `(` $offset `,` $val `:` type($offset) `,` type($val) `)`
    `outs` `(` $dst `:` qualified(type($dst)) `)` attr-dict
  }];
}

