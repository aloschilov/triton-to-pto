//===- PTODialect.cpp - PTO MLIR Dialect registration --------------------===//
//
// PTO dialect with TableGen types (!pto.ptr, !pto.tensor_view, etc.) and ops.
//
//===----------------------------------------------------------------------===//

#include "pto-mlir/Dialect/PTO/PTODialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/Types.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Casting.h"

#include "mlir/Bytecode/BytecodeOpInterface.h"

#define GET_OP_CLASSES
#include "PTOOps.h.inc"

#include "mlir/IR/StorageUniquerSupport.h"
#include "mlir/IR/TypeSupport.h"

using namespace mlir;
using namespace pto;

//===----------------------------------------------------------------------===//
// PTODialect
//===----------------------------------------------------------------------===//

PTODialect::PTODialect(MLIRContext *ctx)
    : Dialect(getDialectNamespace(), ctx, TypeID::get<PTODialect>()) {
  allowUnknownOperations(true);
  initialize();
}

void PTODialect::initialize() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "PTOTypeDefs.cpp.inc"
  >();
  addOperations<
#define GET_OP_LIST
#include "PTOOps.cpp.inc"
      >();
}

// Include the generated op implementations (build, parse, print, verify).
#define GET_OP_CLASSES
#include "PTOOps.cpp.inc"

//===----------------------------------------------------------------------===//
// Custom parse/print for types with hasCustomAssemblyFormat = 1
// Format: <dim0>x<dim1>x...xelementType (e.g. 16x64xf32)
//===----------------------------------------------------------------------===//

namespace {
static void printShapeAndElementType(AsmPrinter &printer,
                                     ArrayRef<int64_t> shape,
                                     Type elementType) {
  printer << '<';
  llvm::interleave(
      shape,
      [&](int64_t d) {
        if (ShapedType::isDynamic(d))
          printer << '?';
        else
          printer << d;
      },
      [&] { printer << 'x'; });
  printer << 'x';
  printer.printType(elementType);
  printer << '>';
}

static ParseResult parseShapeAndElementType(AsmParser &parser,
                                            SmallVectorImpl<int64_t> &shape,
                                            Type &elementType) {
  if (parser.parseLess())
    return failure();
  if (parser.parseDimensionList(shape, /*allowDynamic=*/true,
                                /*withTrailingX=*/true))
    return failure();
  if (parser.parseType(elementType))
    return failure();
  if (parser.parseGreater())
    return failure();
  return success();
}
} // namespace

namespace pto {
// TensorViewType
Type TensorViewType::parse(AsmParser &parser) {
  SmallVector<int64_t, 4> shape;
  Type elementType;
  if (failed(parseShapeAndElementType(parser, shape, elementType)))
    return Type();
  return TensorViewType::get(parser.getContext(), shape, elementType);
}
void TensorViewType::print(AsmPrinter &printer) const {
  printShapeAndElementType(printer, getShape(), getElementType());
}

// PartitionTensorViewType
Type PartitionTensorViewType::parse(AsmParser &parser) {
  SmallVector<int64_t, 4> shape;
  Type elementType;
  if (failed(parseShapeAndElementType(parser, shape, elementType)))
    return Type();
  return PartitionTensorViewType::get(parser.getContext(), shape, elementType);
}
void PartitionTensorViewType::print(AsmPrinter &printer) const {
  printShapeAndElementType(printer, getShape(), getElementType());
}

// TileType
Type TileType::parse(AsmParser &parser) {
  SmallVector<int64_t, 4> shape;
  Type elementType;
  if (failed(parseShapeAndElementType(parser, shape, elementType)))
    return Type();
  return TileType::get(parser.getContext(), shape, elementType);
}
void TileType::print(AsmPrinter &printer) const {
  printShapeAndElementType(printer, getShape(), getElementType());
}

// TileBufType
Type TileBufType::parse(AsmParser &parser) {
  SmallVector<int64_t, 4> shape;
  Type elementType;
  if (failed(parseShapeAndElementType(parser, shape, elementType)))
    return Type();
  return TileBufType::get(parser.getContext(), shape, elementType);
}
void TileBufType::print(AsmPrinter &printer) const {
  printShapeAndElementType(printer, getShape(), getElementType());
}
} // namespace pto

// Generated type storage, PtrType parse/print, and dialect parseType/printType.
#define GET_TYPEDEF_CLASSES
#include "PTOTypeDefs.cpp.inc"
