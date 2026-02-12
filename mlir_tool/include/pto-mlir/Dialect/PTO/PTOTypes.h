//===- PTOTypes.h - PTO Dialect Types ---------------------------*- C++ -*-===//
//
// Custom types for the PTO dialect: tile and memref (PTO-AS aligned).
//
//===----------------------------------------------------------------------===//

#ifndef PTO_MLIR_DIALECT_PTO_PTOTYPES_H
#define PTO_MLIR_DIALECT_PTO_PTOTYPES_H

#include "mlir/IR/Types.h"

namespace pto {
namespace detail {
struct PTOTileTypeStorage;
struct PTOMemRefTypeStorage;
} // namespace detail

/// PTO tile type: !pto.tile<RowsxColsxElementType> (e.g. 16x64xf32).
class PTOTileType
    : public mlir::Type::TypeBase<PTOTileType, mlir::Type,
                                  detail::PTOTileTypeStorage> {
public:
  using Base::Base;

  static PTOTileType get(mlir::MLIRContext *ctx, llvm::ArrayRef<int64_t> shape,
                         mlir::Type elementType);

  llvm::ArrayRef<int64_t> getShape() const;
  mlir::Type getElementType() const;
};

/// PTO memref type: !pto.memref<D0xD1x...xElementType> (e.g. 1x1x1x16x64xf32).
class PTOMemRefType
    : public mlir::Type::TypeBase<PTOMemRefType, mlir::Type,
                                 detail::PTOMemRefTypeStorage> {
public:
  using Base::Base;

  static PTOMemRefType get(mlir::MLIRContext *ctx,
                           llvm::ArrayRef<int64_t> shape,
                           mlir::Type elementType);

  llvm::ArrayRef<int64_t> getShape() const;
  mlir::Type getElementType() const;
};

} // namespace pto

#endif // PTO_MLIR_DIALECT_PTO_PTOTYPES_H
