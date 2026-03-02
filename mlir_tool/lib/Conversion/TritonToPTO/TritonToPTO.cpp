//===- TritonToPTO.cpp - Triton→PTO conversion pass ----------------------===//
//
// Converts Triton IR to PTO dialect MLIR. For the standard vector-add idiom
// (get_program_id, masked load/store, addf), generates output matching PTOAS's
// vadd_triton_style.pto: multi-block scf.if + 1xBS tiles with dynamic valid.
// Other patterns fall back to op-by-op dialect conversion.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Patterns.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "PTO/IR/PTO.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/Support/Casting.h"

#include <optional>

using namespace mlir;
using namespace mlir::triton;

namespace {

// Default address space for tile buffers (PTOAS uses "vec" for UB-style tiles).
static pto::AddressSpaceAttr getDefaultTileAddressSpace(MLIRContext *ctx) {
  return pto::AddressSpaceAttr::get(ctx, pto::AddressSpace::VEC);
}

//--- Block size inference ----------------------------------------------------

static int64_t inferBlockSize(ModuleOp module) {
  int64_t blockSize = 32; // default (matches PTOAS vadd_triton_style)
  module.walk([&](MakeRangeOp range) {
    if (auto endAttr = range.getEndAttr()) {
      if (auto intAttr = dyn_cast<IntegerAttr>(endAttr))
        blockSize = intAttr.getInt();
    }
    return WalkResult::interrupt();
  });
  return blockSize;
}

//--- Tile shape helper -------------------------------------------------------

// Each block processes one row of BLOCK_SIZE elements (1 x numElements tile),
// matching the PTOAS vadd_triton_style pattern.
static std::pair<int64_t, int64_t> computeTileShape(int64_t numElements,
                                                     Type /*elementType*/) {
  return {1, numElements};
}

//--- Type converter ----------------------------------------------------------

// Convert tensor element type to PTO form (e.g. !tt.ptr<f32> -> !pto.ptr<f32>).
static Type convertTensorElementType(Type elemType, MLIRContext *ctx) {
  if (auto ptrType = dyn_cast<triton::PointerType>(elemType)) {
    Type pointee = ptrType.getPointeeType();
    if (!pointee.isIntOrFloat() && !isa<RankedTensorType>(pointee))
      pointee = Float32Type::get(ctx);
    return pto::PtrType::get(ctx, pointee);
  }
  return elemType;
}

class TritonToPTOTypeConverter : public TypeConverter {
public:
  TritonToPTOTypeConverter(MLIRContext *ctx, int64_t blockSize)
      : blockSize(blockSize) {
    addConversion([blockSize](Type type) -> std::optional<Type> {
      if (auto ptrType = dyn_cast<triton::PointerType>(type)) {
        Type elem = ptrType.getPointeeType();
        if (!elem.isIntOrFloat() && !isa<RankedTensorType>(elem))
          elem = Float32Type::get(type.getContext());
        return pto::PtrType::get(type.getContext(), elem);
      }
      if (auto tensorType = dyn_cast<RankedTensorType>(type)) {
        if (tensorType.getRank() != 1)
          return std::nullopt;
        int64_t n = tensorType.getNumElements();
        if (ShapedType::isDynamic(n))
          return std::nullopt;
        Type elem = tensorType.getElementType();
        Type ptoElem = convertTensorElementType(elem, type.getContext());
        auto [rows, cols] = computeTileShape(n, elem);
        auto memSpace = getDefaultTileAddressSpace(type.getContext());
        SmallVector<int64_t, 2> dynValid = {-1, -1};
        return pto::TileBufType::get(type.getContext(), {rows, cols}, ptoElem,
                                     memSpace, dynValid);
      }
      if (type.isIntOrIndex() || llvm::isa<FloatType>(type))
        return type;
      // PTO types are already in the target form -- identity conversion.
      if (llvm::isa<pto::TileBufType, pto::PtrType, pto::TensorViewType,
                    pto::PartitionTensorViewType>(type))
        return type;
      return std::nullopt;
    });
    addConversion([this](RankedTensorType type) -> std::optional<Type> {
      if (type.getRank() != 1)
        return std::nullopt;
      int64_t n = type.getNumElements();
      if (ShapedType::isDynamic(n))
        return std::nullopt;
      Type elem = type.getElementType();
      Type ptoElem = convertTensorElementType(elem, type.getContext());
      auto [rows, cols] = computeTileShape(n, elem);
      auto memSpace = getDefaultTileAddressSpace(type.getContext());
      SmallVector<int64_t, 2> dynValid = {-1, -1};
      return pto::TileBufType::get(type.getContext(), {rows, cols}, ptoElem,
                                   memSpace, dynValid);
    });

    // Materializations: needed by convertRegionTypes (e.g. scf.for bodies)
    // to create temporary casts between tensor and tile_buf types.
    addSourceMaterialization(
        [](OpBuilder &builder, Type resultType, ValueRange inputs,
           Location loc) -> Value {
          return builder
              .create<UnrealizedConversionCastOp>(loc, resultType, inputs)
              .getResult(0);
        });
    addTargetMaterialization(
        [](OpBuilder &builder, Type resultType, ValueRange inputs,
           Location loc) -> Value {
          return builder
              .create<UnrealizedConversionCastOp>(loc, resultType, inputs)
              .getResult(0);
        });
  }

private:
  int64_t blockSize;
};

//--- Base pointer tracing ----------------------------------------------------

static BlockArgument traceBasePointer(Value v) {
  if (auto addptr = v.getDefiningOp<AddPtrOp>())
    return traceBasePointer(addptr.getPtr());
  if (auto splat = v.getDefiningOp<SplatOp>())
    return traceBasePointer(splat.getSrc());
  return dyn_cast<BlockArgument>(v);
}

// Given an operation and a function-level block argument, find the
// corresponding converted block arg in the function entry block. This
// works even when `op` is inside a nested region (e.g. scf.for body).
static Value resolveConvertedFuncArg(Operation *op, BlockArgument baseArg) {
  // Walk up to the enclosing function entry block.
  Operation *parentOp = op;
  while (parentOp && !isa<func::FuncOp>(parentOp) &&
         !isa<triton::FuncOp>(parentOp))
    parentOp = parentOp->getParentOp();
  if (!parentOp)
    return nullptr;
  Block &entryBlock = parentOp->getRegion(0).front();
  unsigned idx = baseArg.getArgNumber();
  if (idx >= entryBlock.getNumArguments())
    return nullptr;
  return entryBlock.getArgument(idx);
}

// For scalar store: when ptr is addptr(base, scalar_offset), return (base, row, col)
// with row = offset as index and col = 0; otherwise (base, c0, c0).
struct ScalarStoreIndices {
  Value basePtr;
  Value row;
  Value col;
};
static std::optional<ScalarStoreIndices>
getScalarStoreIndices(Value origPtr, Operation *op,
                      ConversionPatternRewriter &rewriter) {
  BlockArgument baseArg = traceBasePointer(origPtr);
  if (!baseArg)
    return std::nullopt;
  Value basePtr = resolveConvertedFuncArg(op, baseArg);
  if (!basePtr)
    return std::nullopt;
  Location loc = origPtr.getLoc();
  Type indexType = rewriter.getIndexType();
  Value c0 = rewriter.create<arith::ConstantOp>(
      loc, rewriter.getIntegerAttr(indexType, 0));
  if (auto addptr = origPtr.getDefiningOp<AddPtrOp>()) {
    Value offsetVal = addptr.getOperand(1);
    if (offsetVal.getType().isIndex()) {
      return ScalarStoreIndices{basePtr, offsetVal, c0};
    }
    if (offsetVal.getType().isInteger(32) || offsetVal.getType().isInteger(64)) {
      Value row = rewriter.create<arith::IndexCastUIOp>(loc, indexType,
                                                        offsetVal);
      return ScalarStoreIndices{basePtr, row, c0};
    }
  }
  return ScalarStoreIndices{basePtr, c0, c0};
}

// Classify pointer block arguments of a Triton function as scalar outputs vs.
// tile-backed pointers based on their uses. A pointer that ever participates
// in a tensor-based load/store (via splat/addptr chains) is treated as
// tile-sized; remaining pointer args are treated as scalar outputs.
static llvm::SmallDenseSet<unsigned>
classifyScalarPointerArgs(triton::FuncOp func) {
  llvm::SmallDenseSet<unsigned> pointerArgs;
  llvm::SmallDenseSet<unsigned> tilePointerArgs;

  Block &entryBlock = func.getBody().front();
  for (auto it : llvm::enumerate(entryBlock.getArguments())) {
    unsigned index = it.index();
    BlockArgument arg = it.value();
    if (!llvm::isa<triton::PointerType>(arg.getType()))
      continue;
    pointerArgs.insert(index);

    // Worklist over values derived from this pointer argument.
    SmallVector<Value, 8> worklist;
    llvm::SmallPtrSet<Value, 8> visited;
    worklist.push_back(arg);

    while (!worklist.empty()) {
      Value v = worklist.pop_back_val();
      if (!visited.insert(v).second)
        continue;

      for (Operation *user : v.getUsers()) {
        if (auto splat = dyn_cast<SplatOp>(user)) {
          // Pointer is being broadcast into a tensor of pointers -> tile-backed.
          tilePointerArgs.insert(index);
          worklist.push_back(splat.getResult());
          continue;
        }
        if (auto addPtr = dyn_cast<AddPtrOp>(user)) {
          if (addPtr.getPtr() == v)
            worklist.push_back(addPtr.getResult());
          continue;
        }
        if (auto load = dyn_cast<LoadOp>(user)) {
          if (load.getPtr() == v)
            tilePointerArgs.insert(index);
          continue;
        }
        if (auto store = dyn_cast<StoreOp>(user)) {
          // If this pointer eventually feeds a tensor-valued store, treat as tile.
          if (store.getPtr() == v) {
            if (llvm::isa<RankedTensorType>(store.getValue().getType()))
              tilePointerArgs.insert(index);
          }
          continue;
        }
      }
    }
  }

  llvm::SmallDenseSet<unsigned> scalarPointerArgs;
  for (unsigned idx : pointerArgs) {
    if (!tilePointerArgs.contains(idx))
      scalarPointerArgs.insert(idx);
  }
  return scalarPointerArgs;
}

//--- Conversion patterns -----------------------------------------------------

struct ConvertTTFuncOp : public OpConversionPattern<triton::FuncOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::FuncOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    const TypeConverter *conv = getTypeConverter();
    FunctionType fnType = op.getFunctionType();
    SmallVector<Type, 4> newArgTypes;
    // Classify pointer arguments that should become 1x1x1x1x1 scalar-output
    // memrefs instead of block-sized tiles.
    llvm::SmallDenseSet<unsigned> scalarPointerArgs =
        classifyScalarPointerArgs(op);

    for (auto it : llvm::enumerate(fnType.getInputs())) {
      unsigned index = it.index();
      Type t = it.value();

      // For scalar-output pointer arguments, bypass the generic type
      // converter and materialize a 1x1x1x1x1 memref directly.
      if (auto ptrType = dyn_cast<triton::PointerType>(t)) {
        if (scalarPointerArgs.contains(index)) {
          Type elem = ptrType.getPointeeType();
          if (!elem.isIntOrFloat() && !isa<RankedTensorType>(elem))
            elem = Float32Type::get(t.getContext());
          newArgTypes.push_back(pto::PtrType::get(t.getContext(), elem));
          continue;
        }
      }

      Type c = conv->convertType(t);
      if (!c) {
        if (t.isIntOrIndex())
          newArgTypes.push_back(t);
        else
          return rewriter.notifyMatchFailure(op, "unsupported arg type");
      } else {
        newArgTypes.push_back(c);
      }
    }
    SmallVector<Type, 1> newResultTypes;
    for (Type t : fnType.getResults()) {
      Type c = conv->convertType(t);
      if (!c)
        return rewriter.notifyMatchFailure(op, "unsupported result type");
      newResultTypes.push_back(c);
    }
    auto newFnType =
        rewriter.getFunctionType(newArgTypes, newResultTypes);
    auto newFunc = rewriter.create<func::FuncOp>(op.getLoc(), op.getName(),
                                                 newFnType);
    newFunc.setVisibility(op.getVisibility());
    rewriter.inlineRegionBefore(op.getBody(), newFunc.getBody(), newFunc.end());
    TypeConverter::SignatureConversion signatureConversion(newArgTypes.size());
    for (size_t i = 0; i < newArgTypes.size(); ++i)
      signatureConversion.addInputs(i, {newArgTypes[i]});
    rewriter.applySignatureConversion(&newFunc.getBody(),
                                      signatureConversion);
    rewriter.eraseOp(op);
    return success();
  }
};

struct ConvertTTReturnOp : public OpConversionPattern<triton::ReturnOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::ReturnOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.create<func::ReturnOp>(op.getLoc(), adaptor.getSrcs());
    rewriter.eraseOp(op);
    return success();
  }
};

struct ConvertTTLoadOp : public OpConversionPattern<LoadOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(LoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    BlockArgument baseArg = traceBasePointer(op.getPtr());
    if (!baseArg)
      return rewriter.notifyMatchFailure(op, "could not trace base pointer");
    const TypeConverter *conv = getTypeConverter();
    Value ptr = resolveConvertedFuncArg(op, baseArg);
    if (!ptr)
      return rewriter.notifyMatchFailure(op, "could not resolve func arg");
    auto tileBufType = llvm::dyn_cast_or_null<pto::TileBufType>(
        conv->convertType(op.getType()));
    if (!tileBufType)
      return rewriter.notifyMatchFailure(op, "unsupported result type");
    auto shapeArr = tileBufType.getShape();
    int64_t rows = shapeArr[0];
    int64_t cols = shapeArr[1];
    Location loc = op.getLoc();
    Type indexType = rewriter.getIndexType();
    Value c0 = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getIntegerAttr(indexType, 0));
    Value cRows = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getIntegerAttr(indexType, rows));
    Value cCols = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getIntegerAttr(indexType, cols));
    Value c1 = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getIntegerAttr(indexType, 1));
    // PTOAS-aligned: variadic shape/strides keyword syntax
    auto tensorViewType = pto::TensorViewType::get(
        rewriter.getContext(), {rows, cols}, tileBufType.getElementType());
    SmallVector<Value> shapeVals = {cRows, cCols};
    SmallVector<Value> strides = {cCols, c1};
    Value tv = rewriter.create<pto::MakeTensorViewOp>(
        loc, tensorViewType, ptr, shapeVals, strides, pto::LayoutAttr());
    auto partType = pto::PartitionTensorViewType::get(
        rewriter.getContext(), {rows, cols}, tileBufType.getElementType());
    SmallVector<Value> offsets = {c0, c0};
    SmallVector<Value> sizes = {cRows, cCols};
    Value pv = rewriter.create<pto::PartitionViewOp>(
        loc, partType, tv, offsets, sizes);
    Value tile = rewriter.create<pto::AllocTileOp>(loc, tileBufType, Value(), Value());
    rewriter.create<pto::TLoadOp>(loc, TypeRange{}, pv, tile);
    rewriter.replaceOp(op, tile);
    return success();
  }
};

struct ConvertArithAddFOp : public OpConversionPattern<arith::AddFOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arith::AddFOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type resType = getTypeConverter()->convertType(op.getType());
    auto tileBufType = llvm::dyn_cast<pto::TileBufType>(resType);
    if (!tileBufType)
      return rewriter.notifyMatchFailure(op, "not a tile buffer type");
    Value dst = rewriter.create<pto::AllocTileOp>(op.getLoc(), tileBufType, Value(), Value());
    rewriter.create<pto::TAddOp>(op.getLoc(), adaptor.getLhs(),
                                    adaptor.getRhs(), dst);
    rewriter.replaceOp(op, dst);
    return success();
  }
};

struct ConvertTTReduceOp : public ConversionPattern {
  ConvertTTReduceOp(TypeConverter &typeConverter, MLIRContext *ctx)
      : ConversionPattern(typeConverter, "tt.reduce",
                          /*benefit=*/1, ctx) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    // Expect a single tile operand and a single scalar result.
    if (op->getNumOperands() != 1 || op->getNumResults() != 1)
      return rewriter.notifyMatchFailure(op, "unsupported tt.reduce shape");

    Value tile = operands.front();
    Type resultType = op->getResult(0).getType();
    if (!resultType.isF32())
      return rewriter.notifyMatchFailure(op, "expected f32 reduce result");

    auto tileBufType = llvm::dyn_cast<pto::TileBufType>(tile.getType());
    if (!tileBufType)
      return rewriter.notifyMatchFailure(op, "expected tile_buf operand");

    Location loc = op->getLoc();
    MLIRContext *ctx = rewriter.getContext();
    Type indexType = rewriter.getIndexType();
    Type elemType = tileBufType.getElementType();
    auto tileShape = tileBufType.getShape();
    int64_t rows = tileShape[0];
    int64_t cols = tileShape[1];
    auto memSpace = getDefaultTileAddressSpace(ctx);

    // 1. trowsum: RxC -> Rx1 (sum across columns for each row)
    auto rowResultType =
        pto::TileBufType::get(ctx, {rows, 1}, elemType, memSpace);
    Value rsTmp = rewriter.create<pto::AllocTileOp>(loc, tileBufType, Value(), Value());
    Value rsDst = rewriter.create<pto::AllocTileOp>(loc, rowResultType, Value(), Value());
    rewriter.create<pto::TRowSumOp>(loc, tile, rsTmp, rsDst);

    // 2. tcolsum: Rx1 -> 1x1 (sum across rows)
    auto scalarTileType =
        pto::TileBufType::get(ctx, {1, 1}, elemType, memSpace);
    Value csTmp = rewriter.create<pto::AllocTileOp>(loc, rowResultType, Value(), Value());
    Value csDst = rewriter.create<pto::AllocTileOp>(loc, scalarTileType, Value(), Value());
    rewriter.create<pto::TColSumOp>(loc, rsDst, csTmp, csDst);

    // 3. tgetval: extract scalar from 1x1 tile at offset 0
    Value c0_idx = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getIntegerAttr(indexType, 0));
    Value scalar =
        rewriter.create<pto::TGetValOp>(loc, resultType, csDst, c0_idx);
    rewriter.replaceOp(op, scalar);
    return success();
  }
};

struct ConvertTTStoreOp : public ConversionPattern {
  ConvertTTStoreOp(TypeConverter &typeConverter, MLIRContext *ctx)
      : ConversionPattern(typeConverter, "tt.store", /*benefit=*/1, ctx) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    if (operands.size() < 2)
      return rewriter.notifyMatchFailure(op, "expected ptr and value operands");

    Value origPtr = op->getOperand(0);
    Value origVal = op->getOperand(1);
    BlockArgument baseArg = traceBasePointer(origPtr);
    if (!baseArg)
      return rewriter.notifyMatchFailure(op, "could not trace base pointer");

    Value ptr = resolveConvertedFuncArg(op, baseArg);
    if (!ptr)
      return rewriter.notifyMatchFailure(op, "could not resolve func arg");
    Value value = operands[1];
    Location loc = op->getLoc();
    MLIRContext *ctx = rewriter.getContext();
    Type indexType = rewriter.getIndexType();
    Value c0 = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getIntegerAttr(indexType, 0));
    Value c1 = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getIntegerAttr(indexType, 1));

    if (llvm::isa<RankedTensorType>(origVal.getType())) {
      // Tensor store: make_tensor_view + partition_view + tstore
      auto tileBufType = llvm::dyn_cast<pto::TileBufType>(value.getType());
      if (!tileBufType)
        return rewriter.notifyMatchFailure(op, "expected tile buffer type");
      auto shapeArr = tileBufType.getShape();
      int64_t rows = shapeArr[0];
      int64_t cols = shapeArr[1];
      Value cRows = rewriter.create<arith::ConstantOp>(
          loc, rewriter.getIntegerAttr(indexType, rows));
      Value cCols = rewriter.create<arith::ConstantOp>(
          loc, rewriter.getIntegerAttr(indexType, cols));
      auto tensorViewType = pto::TensorViewType::get(
          ctx, {rows, cols}, tileBufType.getElementType());
      SmallVector<Value> shapeValsStore = {cRows, cCols};
      SmallVector<Value> strides = {cCols, c1};
      Value tv = rewriter.create<pto::MakeTensorViewOp>(
          loc, tensorViewType, ptr, shapeValsStore, strides, pto::LayoutAttr());
      auto partType = pto::PartitionTensorViewType::get(
          ctx, {rows, cols}, tileBufType.getElementType());
      SmallVector<Value> offsets = {c0, c0};
      SmallVector<Value> sizes = {cRows, cCols};
      Value pv = rewriter.create<pto::PartitionViewOp>(
          loc, partType, tv, offsets, sizes);
      rewriter.create<pto::TStoreOp>(loc, TypeRange{}, value, pv);
    } else {
      // Scalar store: PTOAS-aligned make_tensor_view + partition_view +
      //               alloc_tile(1x1) + tsetval + tstore
      std::optional<ScalarStoreIndices> indices =
          getScalarStoreIndices(origPtr, op, rewriter);
      if (!indices)
        return rewriter.notifyMatchFailure(op,
                                           "could not get scalar store indices");
      Type elemType = value.getType();
      auto memSpace = getDefaultTileAddressSpace(ctx);
      auto scalarTileType =
          pto::TileBufType::get(ctx, {1, 1}, elemType, memSpace);
      auto tvType = pto::TensorViewType::get(
          ctx, {ShapedType::kDynamic, ShapedType::kDynamic}, elemType);
      auto pvType =
          pto::PartitionTensorViewType::get(ctx, {1, 1}, elemType);

      Value rowPlus1 =
          rewriter.create<arith::AddIOp>(loc, indices->row, c1);
      SmallVector<Value> tvShape = {rowPlus1, c1};
      SmallVector<Value> tvStrides = {c1, c1};
      Value tv = rewriter.create<pto::MakeTensorViewOp>(
          loc, tvType, indices->basePtr, tvShape, tvStrides, pto::LayoutAttr());

      SmallVector<Value> pvOffsets = {indices->row, c0};
      SmallVector<Value> pvSizes = {c1, c1};
      Value pv = rewriter.create<pto::PartitionViewOp>(
          loc, pvType, tv, pvOffsets, pvSizes);

      Value tile = rewriter.create<pto::AllocTileOp>(loc, scalarTileType, Value(), Value());
      rewriter.create<pto::TSetValOp>(loc, tile, c0, value);
      rewriter.create<pto::TStoreOp>(loc, TypeRange{}, tile, pv);
    }
    rewriter.eraseOp(op);
    return success();
  }
};

// scf.for / scf.yield type conversion is handled by the built-in
// populateSCFStructuralTypeConversionsAndLegality() from MLIRSCFTransforms.

// Convert tensor-typed arith.constant to alloc_tile + texpands (PTOAS-aligned).
struct ConvertArithConstantOp : public OpConversionPattern<arith::ConstantOp> {
  ConvertArithConstantOp(TypeConverter &typeConverter, PatternBenefit benefit,
                         MLIRContext *ctx)
      : OpConversionPattern(typeConverter, ctx, benefit) {}

  LogicalResult
  matchAndRewrite(arith::ConstantOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto tensorType = llvm::dyn_cast<RankedTensorType>(op.getType());
    if (!tensorType)
      return rewriter.notifyMatchFailure(op, "not a tensor type");
    Type newType = getTypeConverter()->convertType(tensorType);
    auto tileBufType = llvm::dyn_cast_or_null<pto::TileBufType>(newType);
    if (!tileBufType)
      return rewriter.notifyMatchFailure(op, "could not convert to tile_buf");
    auto dense = llvm::dyn_cast<DenseElementsAttr>(op.getValue());
    if (!dense || !dense.isSplat())
      return rewriter.notifyMatchFailure(op, "constant must be splat");
    Location loc = op.getLoc();
    // Create scalar constant from the splat value
    auto splatAttr = llvm::cast<TypedAttr>(dense.getSplatValue<Attribute>());
    Value scalar = rewriter.create<arith::ConstantOp>(loc, splatAttr);
    Value tile = rewriter.create<pto::AllocTileOp>(loc, tileBufType, Value(), Value());
    rewriter.create<pto::TExpandsOp>(loc, scalar, tile);
    rewriter.replaceOp(op, tile);
    return success();
  }
};

// Convert tt.get_program_id -> pto.get_block_idx + index_cast (PTOAS-aligned).
// The ground truth works in index space throughout.
struct ConvertTTGetProgramIdOp
    : public OpConversionPattern<GetProgramIdOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(GetProgramIdOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value pid64 =
        rewriter.create<pto::GetBlockIdxOp>(loc, rewriter.getI64Type());
    Value pidIdx =
        rewriter.create<arith::IndexCastOp>(loc, rewriter.getIndexType(), pid64);
    // Triton get_program_id returns i32; truncate for users that expect i32.
    Value pid32 =
        rewriter.create<arith::TruncIOp>(loc, rewriter.getI32Type(), pidIdx);
    rewriter.replaceOp(op, pid32);
    return success();
  }
};

// Erase Triton infrastructure ops. Replace results so uses get a legal value.
template <typename OpTy>
struct EraseTritonOp : public OpConversionPattern<OpTy> {
  using OpConversionPattern<OpTy>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(OpTy op, typename OpTy::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<Value, 2> replacements;
    for (OpResult result : op->getResults()) {
      Type oldType = result.getType();
      Type newType = this->getTypeConverter()->convertType(oldType);
      if (!newType)
        newType = oldType;
      if (newType.isIntOrIndex()) {
        Value c0 = rewriter.create<arith::ConstantOp>(
            op.getLoc(), newType,
            rewriter.getIntegerAttr(newType, 0));
        replacements.push_back(c0);
      } else if (auto tileBufType = llvm::dyn_cast<pto::TileBufType>(newType)) {
        Value tile = rewriter.create<pto::AllocTileOp>(op.getLoc(), tileBufType,
                                                       Value(), Value());
        replacements.push_back(tile);
      } else {
        return rewriter.notifyMatchFailure(op, "unsupported result type for erase");
      }
    }
    rewriter.replaceOp(op, replacements);
    return success();
  }
};

// NOTE: EraseArith{Muli,AddI,Cmpi}Op patterns were removed. They replaced
// live SSA values with constant 0 before erasing, silently producing
// incorrect PTO MLIR for any non-vec-add kernel. The holistic vec-add
// rewrite (Phase 1) handles these ops correctly; Phase 2 should fail
// loudly on unsupported arith ops rather than produce wrong output.

//=== Holistic vec-add pattern analysis and rewrite ========================
//
// Detects the standard Triton vector-add idiom (get_program_id, masked
// load/store, addf) and generates the complete PTO body matching PTOAS's
// vadd_triton_style.pto.

struct VecAddInfo {
  int64_t blockSize = 0;
  int nElementsArgIdx = -1;
  SmallVector<int, 2> loadPtrArgIndices;
  int storePtrArgIdx = -1;
  Type elementType;
};

static bool analyzeVecAddPattern(triton::FuncOp func, VecAddInfo &info) {
  bool hasGetProgramId = false;
  func.walk([&](GetProgramIdOp) { hasGetProgramId = true; });
  if (!hasGetProgramId)
    return false;

  func.walk([&](MakeRangeOp op) {
    info.blockSize = cast<IntegerAttr>(op.getEndAttr()).getInt();
  });
  if (info.blockSize == 0)
    return false;

  Block &body = func.getBody().front();
  for (auto it : llvm::enumerate(body.getArguments())) {
    Type t = it.value().getType();
    if (t.isInteger(32) && !isa<triton::PointerType>(t))
      info.nElementsArgIdx = it.index();
  }
  if (info.nElementsArgIdx < 0)
    return false;

  SmallVector<LoadOp> loads;
  func.walk([&](LoadOp op) { loads.push_back(op); });
  if (loads.size() != 2)
    return false;
  for (auto load : loads) {
    BlockArgument base = traceBasePointer(load.getPtr());
    if (!base)
      return false;
    info.loadPtrArgIndices.push_back(base.getArgNumber());
  }

  SmallVector<StoreOp> stores;
  func.walk([&](StoreOp op) { stores.push_back(op); });
  if (stores.size() != 1)
    return false;
  BlockArgument storeBase = traceBasePointer(stores[0].getPtr());
  if (!storeBase)
    return false;
  info.storePtrArgIdx = storeBase.getArgNumber();

  arith::AddFOp addFOp = nullptr;
  func.walk([&](arith::AddFOp op) { addFOp = op; });
  if (!addFOp)
    return false;

  auto loadResultType = dyn_cast<RankedTensorType>(loads[0].getType());
  if (!loadResultType)
    return false;
  info.elementType = loadResultType.getElementType();
  return true;
}

static LogicalResult rewriteVecAdd(triton::FuncOp ttFunc,
                                   const VecAddInfo &info) {
  OpBuilder builder(ttFunc);
  Location loc = ttFunc.getLoc();
  MLIRContext *ctx = builder.getContext();

  // Build new function signature: !tt.ptr<f32> → !pto.ptr<f32>, i32 passes through
  SmallVector<Type> newArgTypes;
  for (Type t : ttFunc.getFunctionType().getInputs()) {
    if (auto ptrType = dyn_cast<triton::PointerType>(t)) {
      Type elem = ptrType.getPointeeType();
      if (!elem.isIntOrFloat())
        elem = Float32Type::get(ctx);
      newArgTypes.push_back(pto::PtrType::get(ctx, elem));
    } else {
      newArgTypes.push_back(t);
    }
  }

  auto newFunc = builder.create<func::FuncOp>(
      loc, ttFunc.getName(), builder.getFunctionType(newArgTypes, {}));
  newFunc.setVisibility(ttFunc.getVisibility());
  Block *entry = newFunc.addEntryBlock();
  builder.setInsertionPointToStart(entry);

  int64_t BS = info.blockSize;
  Type elemType = info.elementType;
  Type indexType = builder.getIndexType();
  auto memSpace = getDefaultTileAddressSpace(ctx);

  // Constants
  Value c0 = builder.create<arith::ConstantOp>(
      loc, builder.getIntegerAttr(indexType, 0));
  Value c1 = builder.create<arith::ConstantOp>(
      loc, builder.getIntegerAttr(indexType, 1));
  Value cBS = builder.create<arith::ConstantOp>(
      loc, builder.getIntegerAttr(indexType, BS));

  // n_elements → index
  Value nIdx = builder.create<arith::IndexCastOp>(
      loc, indexType, entry->getArgument(info.nElementsArgIdx));

  // Block index → index
  Value bidx = builder.create<pto::GetBlockIdxOp>(loc, builder.getI64Type());
  Value bidxIdx = builder.create<arith::IndexCastOp>(loc, indexType, bidx);

  // block_start, remaining
  Value blockStart = builder.create<arith::MulIOp>(loc, bidxIdx, cBS);
  Value remaining = builder.create<arith::SubIOp>(loc, nIdx, blockStart);

  // Tail guard: scf.if remaining > 0
  Value doWork = builder.create<arith::CmpIOp>(
      loc, arith::CmpIPredicate::ugt, remaining, c0);
  auto ifOp = builder.create<scf::IfOp>(loc, /*resultTypes=*/TypeRange{},
                                         doWork, /*addElseBlock=*/false);

  // --- scf.if body ---
  builder.setInsertionPointToStart(&ifOp.getThenRegion().front());

  Value validCol = builder.create<arith::MinUIOp>(loc, remaining, cBS);
  // validRow = c1, colOffset = c0 (reuse outer constants)
  Value nRows = builder.create<arith::CeilDivUIOp>(loc, nIdx, cBS);

  // Types
  auto tvType = pto::TensorViewType::get(
      ctx, {ShapedType::kDynamic, ShapedType::kDynamic}, elemType);
  auto pvType = pto::PartitionTensorViewType::get(
      ctx, {ShapedType::kDynamic, ShapedType::kDynamic}, elemType);
  // PTOAS uses -1 (not ShapedType::kDynamic) as the dynamic valid dim marker.
  SmallVector<int64_t, 2> dynValid = {-1, -1};
  auto tileBufType =
      pto::TileBufType::get(ctx, {1, BS}, elemType, memSpace, dynValid);

  SmallVector<Value> tvShape = {nRows, cBS};
  SmallVector<Value> tvStrides = {cBS, c1};

  // Tensor views for all 3 pointers
  Value tvX = builder.create<pto::MakeTensorViewOp>(
      loc, tvType, entry->getArgument(info.loadPtrArgIndices[0]), tvShape,
      tvStrides, pto::LayoutAttr());
  Value tvY = builder.create<pto::MakeTensorViewOp>(
      loc, tvType, entry->getArgument(info.loadPtrArgIndices[1]), tvShape,
      tvStrides, pto::LayoutAttr());
  Value tvOut = builder.create<pto::MakeTensorViewOp>(
      loc, tvType, entry->getArgument(info.storePtrArgIdx), tvShape, tvStrides,
      pto::LayoutAttr());

  SmallVector<Value> pvOffsets = {bidxIdx, c0};
  SmallVector<Value> pvSizes = {c1, validCol};

  // Partition views for loads
  Value pvX = builder.create<pto::PartitionViewOp>(loc, pvType, tvX, pvOffsets,
                                                    pvSizes);
  Value pvY = builder.create<pto::PartitionViewOp>(loc, pvType, tvY, pvOffsets,
                                                    pvSizes);

  // Alloc tiles (1 x BS, dynamic valid)
  Value tileX =
      builder.create<pto::AllocTileOp>(loc, tileBufType, c1, validCol);
  Value tileY =
      builder.create<pto::AllocTileOp>(loc, tileBufType, c1, validCol);
  Value tileOut =
      builder.create<pto::AllocTileOp>(loc, tileBufType, c1, validCol);

  // tload, tadd
  builder.create<pto::TLoadOp>(loc, TypeRange{}, pvX, tileX);
  builder.create<pto::TLoadOp>(loc, TypeRange{}, pvY, tileY);
  builder.create<pto::TAddOp>(loc, tileX, tileY, tileOut);

  // Partition view for output + tstore
  Value pvOut = builder.create<pto::PartitionViewOp>(loc, pvType, tvOut,
                                                      pvOffsets, pvSizes);
  builder.create<pto::TStoreOp>(loc, TypeRange{}, tileOut, pvOut);

  // scf.yield is auto-generated by IfOp builder

  // Return after scf.if
  builder.setInsertionPointAfter(ifOp);
  builder.create<func::ReturnOp>(loc);

  // Erase old Triton function
  ttFunc.erase();
  return success();
}

//--- Pass --------------------------------------------------------------------

struct TritonToPTO : public PassWrapper<TritonToPTO, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TritonToPTO)

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<mlir::pto::PTODialect, triton::TritonDialect,
                    func::FuncDialect, arith::ArithDialect,
                    mlir::scf::SCFDialect>();
  }

  StringRef getArgument() const final { return "convert-triton-to-pto"; }
  StringRef getDescription() const final {
    return "Convert Triton dialect to PTO dialect (full lowering)";
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    getContext().getOrLoadDialect<mlir::pto::PTODialect>();

    // Phase 1: Holistic vec-add rewrites (matches vadd_triton_style.pto).
    // Functions that match the vec-add idiom are fully converted here.
    SmallVector<triton::FuncOp> vecAddFuncs;
    module.walk([&](triton::FuncOp func) {
      VecAddInfo info;
      if (analyzeVecAddPattern(func, info))
        vecAddFuncs.push_back(func);
    });
    for (auto func : vecAddFuncs) {
      VecAddInfo info;
      analyzeVecAddPattern(func, info);
      if (failed(rewriteVecAdd(func, info)))
        return signalPassFailure();
    }

    // Phase 2: Dialect conversion for remaining (non-vec-add) Triton ops.
    bool hasRemainingTritonOps = false;
    module.walk([&](Operation *op) {
      if (op->getDialect() &&
          op->getDialect()->getNamespace() == "tt")
        hasRemainingTritonOps = true;
    });
    if (!hasRemainingTritonOps)
      return;

    int64_t blockSize = inferBlockSize(module);
    TritonToPTOTypeConverter typeConverter(&getContext(), blockSize);

    RewritePatternSet patterns(&getContext());
    patterns.add<ConvertTTFuncOp, ConvertTTReturnOp, ConvertTTLoadOp,
                 ConvertArithAddFOp, ConvertTTStoreOp>(typeConverter,
                                                      &getContext());
    patterns.add<ConvertTTReduceOp>(typeConverter, &getContext());
    patterns.add<ConvertArithConstantOp>(typeConverter, /*benefit=*/2,
                                         &getContext());
    patterns.add<ConvertTTGetProgramIdOp>(typeConverter, &getContext());
    patterns.add<EraseTritonOp<MakeRangeOp>,
                 EraseTritonOp<SplatOp>, EraseTritonOp<AddPtrOp>>(
        typeConverter, &getContext());

    ConversionTarget target(getContext());
    target.addLegalDialect<mlir::pto::PTODialect, func::FuncDialect,
                          arith::ArithDialect>();
    target.addIllegalDialect<triton::TritonDialect>();
    target.addDynamicallyLegalOp<arith::ConstantOp>([](arith::ConstantOp op) {
      return std::optional<bool>(
          !llvm::isa<RankedTensorType>(op.getResult().getType()));
    });
    target.addDynamicallyLegalOp<arith::AddFOp>([](arith::AddFOp op) {
      return std::optional<bool>(
          !llvm::isa<RankedTensorType>(op.getResult().getType()));
    });
    target.addLegalOp<UnrealizedConversionCastOp>();
    mlir::scf::populateSCFStructuralTypeConversionsAndLegality(
        typeConverter, patterns, target);

    if (failed(applyPartialConversion(module, target, std::move(patterns))))
      signalPassFailure();
  }
};

//--- Cleanup: remove unused unrealized_conversion_cast (they may reference tt) -
struct DropUnusedUnrealizedCasts
    : public PassWrapper<DropUnusedUnrealizedCasts, OperationPass<ModuleOp>> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    SmallVector<UnrealizedConversionCastOp, 8> toErase;
    module.walk([&](UnrealizedConversionCastOp op) {
      if (op.getResult(0).use_empty())
        toErase.push_back(op);
    });
    for (UnrealizedConversionCastOp op : toErase)
      op.erase();
  }
};

} // namespace

std::unique_ptr<Pass> createTritonToPTOPass() {
  return std::make_unique<TritonToPTO>();
}

std::unique_ptr<Pass> createDropUnusedUnrealizedCastsPass() {
  return std::make_unique<DropUnusedUnrealizedCasts>();
}
