//===- PTOTypeDefs.td - PTO type definitions ---------------*- tablegen -*-===//
//
// PTO dialect types (aligned with PTOAS). Simplified TileBufType without
// TileBufConfigAttr for initial migration.
//
//===----------------------------------------------------------------------===//

include "mlir/IR/AttrTypeBase.td"
include "PTODialect.td"

// ---- !pto.ptr<elem> ----
def PTO_PtrType : TypeDef<PTO_Dialect, "Ptr"> {
  let mnemonic = "ptr";
  let parameters = (ins "mlir::Type":$elementType);
  let assemblyFormat = "`<` $elementType `>`";
}

// ---- !pto.tensor_view<d0xd1x...xelem> ----
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

// ---- !pto.partition_tensor_view<d0xd1x...xelem> ----
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

// ---- !pto.tile<d0xd1x...xelem> (logical tile, no buffer) ----
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

// ---- !pto.tile_buf<d0xd1x...xelem> (simplified: shape + elementType only) ----
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
