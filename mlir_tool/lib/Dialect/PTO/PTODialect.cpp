//===- PTODialect.cpp - PTO MLIR Dialect registration --------------------===//
//
// PTO dialect with custom types (!pto.tile, !pto.memref) and ops.
//
//===----------------------------------------------------------------------===//

#include "pto-mlir/Dialect/PTO/PTODialect.h"
#include "pto-mlir/Dialect/PTO/PTOTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#define GET_OP_CLASSES
#include "PTOOps.h.inc"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/StorageUniquerSupport.h"
#include "mlir/IR/TypeSupport.h"
#include "mlir/IR/Types.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Casting.h"

using namespace mlir;
using namespace pto;

//===----------------------------------------------------------------------===//
// PTOTileTypeStorage
//===----------------------------------------------------------------------===//

namespace pto {
namespace detail {

struct PTOTileTypeStorage : public TypeStorage {
  using KeyTy = std::tuple<ArrayRef<int64_t>, Type>;

  static PTOTileTypeStorage *construct(TypeStorageAllocator &allocator,
                                       const KeyTy &key) {
    ArrayRef<int64_t> shape = std::get<0>(key);
    Type elementType = std::get<1>(key);
    shape = allocator.copyInto(shape);
    return new (allocator.allocate<PTOTileTypeStorage>())
        PTOTileTypeStorage(shape, elementType);
  }

  static llvm::hash_code hashKey(const KeyTy &key) {
    return llvm::hash_combine(std::get<0>(key), std::get<1>(key));
  }

  bool operator==(const KeyTy &key) const {
    return std::get<0>(key) == shape && std::get<1>(key) == elementType;
  }

  PTOTileTypeStorage(ArrayRef<int64_t> shape, Type elementType)
      : shape(shape), elementType(elementType) {}

  ArrayRef<int64_t> shape;
  Type elementType;
};

} // namespace detail
} // namespace pto

PTOTileType PTOTileType::get(MLIRContext *ctx, ArrayRef<int64_t> shape,
                             Type elementType) {
  return Base::get(ctx, shape, elementType);
}

ArrayRef<int64_t> PTOTileType::getShape() const {
  return getImpl()->shape;
}

Type PTOTileType::getElementType() const {
  return getImpl()->elementType;
}

//===----------------------------------------------------------------------===//
// PTOMemRefTypeStorage
//===----------------------------------------------------------------------===//

namespace pto {
namespace detail {

struct PTOMemRefTypeStorage : public TypeStorage {
  using KeyTy = std::tuple<ArrayRef<int64_t>, Type>;

  static PTOMemRefTypeStorage *construct(TypeStorageAllocator &allocator,
                                         const KeyTy &key) {
    ArrayRef<int64_t> shape = std::get<0>(key);
    Type elementType = std::get<1>(key);
    shape = allocator.copyInto(shape);
    return new (allocator.allocate<PTOMemRefTypeStorage>())
        PTOMemRefTypeStorage(shape, elementType);
  }

  static llvm::hash_code hashKey(const KeyTy &key) {
    return llvm::hash_combine(std::get<0>(key), std::get<1>(key));
  }

  bool operator==(const KeyTy &key) const {
    return std::get<0>(key) == shape && std::get<1>(key) == elementType;
  }

  PTOMemRefTypeStorage(ArrayRef<int64_t> shape, Type elementType)
      : shape(shape), elementType(elementType) {}

  ArrayRef<int64_t> shape;
  Type elementType;
};

} // namespace detail
} // namespace pto

PTOMemRefType PTOMemRefType::get(MLIRContext *ctx, ArrayRef<int64_t> shape,
                                 Type elementType) {
  return Base::get(ctx, shape, elementType);
}

ArrayRef<int64_t> PTOMemRefType::getShape() const {
  return getImpl()->shape;
}

Type PTOMemRefType::getElementType() const {
  return getImpl()->elementType;
}

//===----------------------------------------------------------------------===//
// PTODialect
//===----------------------------------------------------------------------===//

PTODialect::PTODialect(MLIRContext *ctx)
    : Dialect(getDialectNamespace(), ctx, TypeID::get<PTODialect>()) {
  allowUnknownOperations(true);
  initialize();
}

void PTODialect::initialize() {
  // Register custom types with the uniquer (required for TypeStorage-based types).
  addType(
      PTOTileType::getTypeID(),
      AbstractType::get(
          *this, mlir::detail::InterfaceMap(),
          [](TypeID) { return false; },
          [](Type type, function_ref<void(Attribute)> walkAttrs,
             function_ref<void(Type)> walkTypes) {
            if (auto t = llvm::dyn_cast<PTOTileType>(type))
              walkTypes(t.getElementType());
          },
          [](Type type, ArrayRef<Attribute> attrs,
             ArrayRef<Type> types) -> Type {
            if (llvm::isa<PTOTileType>(type) && types.size() == 1)
              return PTOTileType::get(
                  type.getContext(),
                  llvm::cast<PTOTileType>(type).getShape(), types[0]);
            return type;
          },
          PTOTileType::getTypeID(), "tile"));
  mlir::detail::TypeUniquer::registerType<PTOTileType>(getContext());
  addType(
      PTOMemRefType::getTypeID(),
      AbstractType::get(
          *this, mlir::detail::InterfaceMap(),
          [](TypeID) { return false; },
          [](Type type, function_ref<void(Attribute)> walkAttrs,
             function_ref<void(Type)> walkTypes) {
            if (auto t = llvm::dyn_cast<PTOMemRefType>(type))
              walkTypes(t.getElementType());
          },
          [](Type type, ArrayRef<Attribute> attrs,
             ArrayRef<Type> types) -> Type {
            if (llvm::isa<PTOMemRefType>(type) && types.size() == 1)
              return PTOMemRefType::get(
                  type.getContext(),
                  llvm::cast<PTOMemRefType>(type).getShape(), types[0]);
            return type;
          },
          PTOMemRefType::getTypeID(), "memref"));
  mlir::detail::TypeUniquer::registerType<PTOMemRefType>(getContext());
  addOperations<
#define GET_OP_LIST
#include "PTOOps.cpp.inc"
      >();
}

// Include the generated op implementations (build, parse, print, verify).
#define GET_OP_CLASSES
#include "PTOOps.cpp.inc"

static void printShapeAndElementType(DialectAsmPrinter &printer,
                                     ArrayRef<int64_t> shape,
                                     Type elementType) {
  printer << '<';
  llvm::interleave(
      shape,
      [&](int64_t d) { printer << d; },
      [&] { printer << 'x'; });
  printer << 'x';
  printer.printType(elementType);
  printer << '>';
}

void PTODialect::printType(Type type, DialectAsmPrinter &printer) const {
  if (auto tileType = llvm::dyn_cast<PTOTileType>(type)) {
    printer << "tile";
    printShapeAndElementType(printer, tileType.getShape(),
                             tileType.getElementType());
    return;
  }
  if (auto memrefType = llvm::dyn_cast<PTOMemRefType>(type)) {
    printer << "memref";
    printShapeAndElementType(printer, memrefType.getShape(),
                             memrefType.getElementType());
    return;
  }
  llvm_unreachable("unexpected PTO type");
}

Type PTODialect::parseType(DialectAsmParser &parser) const {
  StringRef typeKeyword;
  if (failed(parser.parseKeyword(&typeKeyword)))
    return Type();

  if (typeKeyword == "tile") {
    if (failed(parser.parseLess()))
      return Type();
    SmallVector<int64_t, 4> shape;
    // parseDimensionList(shape, allowDynamic, withTrailingX) parses e.g. 16x64x
    if (failed(parser.parseDimensionList(shape, /*allowDynamic=*/true,
                                         /*withTrailingX=*/true)))
      return Type();
    Type elementType;
    if (failed(parser.parseType(elementType)))
      return Type();
    if (failed(parser.parseGreater()))
      return Type();
    return PTOTileType::get(getContext(), shape, elementType);
  }

  if (typeKeyword == "memref") {
    if (failed(parser.parseLess()))
      return Type();
    SmallVector<int64_t, 6> shape;
    if (failed(parser.parseDimensionList(shape, /*allowDynamic=*/true,
                                         /*withTrailingX=*/true)))
      return Type();
    Type elementType;
    if (failed(parser.parseType(elementType)))
      return Type();
    if (failed(parser.parseGreater()))
      return Type();
    return PTOMemRefType::get(getContext(), shape, elementType);
  }

  parser.emitError(parser.getNameLoc(), "unknown PTO type: ") << typeKeyword;
  return Type();
}
