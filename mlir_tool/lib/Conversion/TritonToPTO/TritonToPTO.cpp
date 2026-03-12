//===- TritonToPTO.cpp - Triton→PTO conversion pass ----------------------===//
//
// Converts Triton IR to PTO dialect MLIR. Recognized kernel idioms (vec-add,
// element-wise unary, reduction, fused softmax) use a grid-stride scf.for
// loop with pto.get_block_num as stride, so each core processes multiple
// blocks. Other patterns fall back to op-by-op dialect conversion.
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

// Convert tt.get_num_programs -> pto.get_block_num + index_cast (Phase 2 fallback).
struct ConvertTTGetNumProgramsOp
    : public OpConversionPattern<GetNumProgramsOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(GetNumProgramsOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value num64 =
        rewriter.create<pto::GetBlockNumOp>(loc, rewriter.getI64Type());
    Value numIdx =
        rewriter.create<arith::IndexCastOp>(loc, rewriter.getIndexType(), num64);
    Value num32 =
        rewriter.create<arith::TruncIOp>(loc, rewriter.getI32Type(), numIdx);
    rewriter.replaceOp(op, num32);
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

  // Grid-stride loop: each core iterates blockIdx, blockIdx+nCores, ...
  Value bidx = builder.create<pto::GetBlockIdxOp>(loc, builder.getI64Type());
  Value nCores = builder.create<pto::GetBlockNumOp>(loc, builder.getI64Type());
  Value bidxIdx = builder.create<arith::IndexCastOp>(loc, indexType, bidx);
  Value strideIdx = builder.create<arith::IndexCastOp>(loc, indexType, nCores);
  Value nBlocks = builder.create<arith::CeilDivUIOp>(loc, nIdx, cBS);

  auto forOp = builder.create<scf::ForOp>(loc, bidxIdx, nBlocks, strideIdx);
  builder.setInsertionPointToStart(forOp.getBody());
  Value blockId = forOp.getInductionVar();

  Value blockStart = builder.create<arith::MulIOp>(loc, blockId, cBS);
  Value remaining = builder.create<arith::SubIOp>(loc, nIdx, blockStart);
  Value validCol = builder.create<arith::MinUIOp>(loc, remaining, cBS);

  // Types
  auto tvType = pto::TensorViewType::get(
      ctx, {ShapedType::kDynamic, ShapedType::kDynamic}, elemType);
  auto pvType = pto::PartitionTensorViewType::get(
      ctx, {ShapedType::kDynamic, ShapedType::kDynamic}, elemType);
  SmallVector<int64_t, 2> dynValid = {-1, -1};
  auto tileBufType =
      pto::TileBufType::get(ctx, {1, BS}, elemType, memSpace, dynValid);

  SmallVector<Value> tvShape = {nBlocks, cBS};
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

  SmallVector<Value> pvOffsets = {blockId, c0};
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

  // scf.yield is auto-generated by ForOp builder

  // Return after scf.for
  builder.setInsertionPointAfter(forOp);
  builder.create<func::ReturnOp>(loc);

  // Erase old Triton function
  ttFunc.erase();
  return success();
}

//=== Holistic element-wise unary pattern ====================================
//
// Future: detect single-operand element-wise kernels (abs, relu, gelu, exp,
// etc.) with the structure: get_program_id, masked load, unary op, masked
// store. The rewrite follows the same template as vec-add but with a single
// tload, a pto unary tile op, and a tstore.

struct ElementWiseUnaryInfo {
  int64_t blockSize = 0;
  int nElementsArgIdx = -1;
  int loadPtrArgIdx = -1;
  int storePtrArgIdx = -1;
  Type elementType;
  StringRef unaryOpName; // e.g. "abs", "exp", "relu"
};

static bool analyzeElementWiseUnary(triton::FuncOp func,
                                    ElementWiseUnaryInfo &info) {
  bool hasGetProgramId = false;
  func.walk([&](GetProgramIdOp) { hasGetProgramId = true; });
  if (!hasGetProgramId)
    return false;

  func.walk([&](MakeRangeOp op) {
    info.blockSize = cast<IntegerAttr>(op.getEndAttr()).getInt();
  });
  if (info.blockSize == 0)
    return false;

  SmallVector<LoadOp> loads;
  func.walk([&](LoadOp op) { loads.push_back(op); });
  if (loads.size() != 1)
    return false;

  SmallVector<StoreOp> stores;
  func.walk([&](StoreOp op) { stores.push_back(op); });
  if (stores.size() != 1)
    return false;

  // Must not have addf (that's vec-add, not unary)
  bool hasAddF = false;
  func.walk([&](arith::AddFOp) { hasAddF = true; });
  if (hasAddF)
    return false;

  // Check for known unary ops (math.absf, math.exp, arith.maxf with 0, etc.)
  bool hasUnary = false;
  func.walk([&](Operation *op) {
    if (op->getDialect() &&
        op->getDialect()->getNamespace() == "math") {
      hasUnary = true;
      info.unaryOpName = op->getName().getStringRef();
    }
  });
  if (!hasUnary)
    return false;

  BlockArgument loadBase = traceBasePointer(loads[0].getPtr());
  BlockArgument storeBase = traceBasePointer(stores[0].getPtr());
  if (!loadBase || !storeBase)
    return false;
  info.loadPtrArgIdx = loadBase.getArgNumber();
  info.storePtrArgIdx = storeBase.getArgNumber();

  auto loadType = dyn_cast<RankedTensorType>(loads[0].getType());
  if (!loadType)
    return false;
  info.elementType = loadType.getElementType();

  Block &body = func.getBody().front();
  for (auto it : llvm::enumerate(body.getArguments())) {
    Type t = it.value().getType();
    if (t.isInteger(32) && !isa<triton::PointerType>(t))
      info.nElementsArgIdx = it.index();
  }

  return info.nElementsArgIdx >= 0;
}

static LogicalResult rewriteElementWiseUnary(triton::FuncOp ttFunc,
                                             const ElementWiseUnaryInfo &info) {
  OpBuilder builder(ttFunc);
  Location loc = ttFunc.getLoc();
  MLIRContext *ctx = builder.getContext();

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

  Value c0 = builder.create<arith::ConstantOp>(
      loc, builder.getIntegerAttr(indexType, 0));
  Value c1 = builder.create<arith::ConstantOp>(
      loc, builder.getIntegerAttr(indexType, 1));
  Value cBS = builder.create<arith::ConstantOp>(
      loc, builder.getIntegerAttr(indexType, BS));

  Value nIdx = builder.create<arith::IndexCastOp>(
      loc, indexType, entry->getArgument(info.nElementsArgIdx));

  // Grid-stride loop: each core iterates blockIdx, blockIdx+nCores, ...
  Value bidx = builder.create<pto::GetBlockIdxOp>(loc, builder.getI64Type());
  Value nCores = builder.create<pto::GetBlockNumOp>(loc, builder.getI64Type());
  Value bidxIdx = builder.create<arith::IndexCastOp>(loc, indexType, bidx);
  Value strideIdx = builder.create<arith::IndexCastOp>(loc, indexType, nCores);
  Value nBlocks = builder.create<arith::CeilDivUIOp>(loc, nIdx, cBS);

  auto forOp = builder.create<scf::ForOp>(loc, bidxIdx, nBlocks, strideIdx);
  builder.setInsertionPointToStart(forOp.getBody());
  Value blockId = forOp.getInductionVar();

  Value blockStart = builder.create<arith::MulIOp>(loc, blockId, cBS);
  Value remaining = builder.create<arith::SubIOp>(loc, nIdx, blockStart);
  Value validCol = builder.create<arith::MinUIOp>(loc, remaining, cBS);

  auto tvType = pto::TensorViewType::get(
      ctx, {ShapedType::kDynamic, ShapedType::kDynamic}, elemType);
  auto pvType = pto::PartitionTensorViewType::get(
      ctx, {ShapedType::kDynamic, ShapedType::kDynamic}, elemType);
  SmallVector<int64_t, 2> dynValid = {-1, -1};
  auto tileBufType =
      pto::TileBufType::get(ctx, {1, BS}, elemType, memSpace, dynValid);

  SmallVector<Value> tvShape = {nBlocks, cBS};
  SmallVector<Value> tvStrides = {cBS, c1};

  Value tvIn = builder.create<pto::MakeTensorViewOp>(
      loc, tvType, entry->getArgument(info.loadPtrArgIdx), tvShape, tvStrides,
      pto::LayoutAttr());
  Value tvOut = builder.create<pto::MakeTensorViewOp>(
      loc, tvType, entry->getArgument(info.storePtrArgIdx), tvShape, tvStrides,
      pto::LayoutAttr());

  SmallVector<Value> pvOffsets = {blockId, c0};
  SmallVector<Value> pvSizes = {c1, validCol};

  Value pvIn = builder.create<pto::PartitionViewOp>(loc, pvType, tvIn,
                                                     pvOffsets, pvSizes);

  Value tileSrc =
      builder.create<pto::AllocTileOp>(loc, tileBufType, c1, validCol);
  Value tileDst =
      builder.create<pto::AllocTileOp>(loc, tileBufType, c1, validCol);

  builder.create<pto::TLoadOp>(loc, TypeRange{}, pvIn, tileSrc);

  // Dispatch unary op: map math dialect op name to PTO tile op.
  StringRef opName = info.unaryOpName;
  if (opName == "math.absf")
    builder.create<pto::TAbsOp>(loc, tileSrc, tileDst);
  else if (opName == "math.exp" || opName == "math.exp2")
    builder.create<pto::TExpOp>(loc, tileSrc, tileDst);
  else if (opName == "math.log" || opName == "math.log2")
    builder.create<pto::TLogOp>(loc, tileSrc, tileDst);
  else if (opName == "math.sqrt")
    builder.create<pto::TSqrtOp>(loc, tileSrc, tileDst);
  else if (opName == "math.rsqrt")
    builder.create<pto::TRsqrtOp>(loc, tileSrc, tileDst);
  else if (opName == "arith.negf")
    builder.create<pto::TNegOp>(loc, tileSrc, tileDst);
  else {
    ttFunc.emitError("unsupported unary op: ") << opName;
    newFunc.erase();
    return failure();
  }

  Value pvOut = builder.create<pto::PartitionViewOp>(loc, pvType, tvOut,
                                                      pvOffsets, pvSizes);
  builder.create<pto::TStoreOp>(loc, TypeRange{}, tileDst, pvOut);

  builder.setInsertionPointAfter(forOp);
  builder.create<func::ReturnOp>(loc);

  ttFunc.erase();
  return success();
}

//=== Holistic reduction pattern =============================================
//
// Detects simple Triton reduction kernels: get_program_id, masked load,
// tt.reduce (add/max/min combiner), scalar store. The rewrite loads a
// 1xBS tile, applies trowsum/trowmax/trowmin, extracts the scalar via
// tgetval, and stores via a 1x1 partition.

enum class ReduceKind { Sum, Max, Min };

struct ReductionInfo {
  int64_t blockSize = 0;
  int nElementsArgIdx = -1;
  int loadPtrArgIdx = -1;
  int storePtrArgIdx = -1;
  Type elementType;
  ReduceKind kind = ReduceKind::Sum;
};

static ReduceKind detectReduceKind(Operation *reduceOp) {
  if (reduceOp->getNumRegions() == 0)
    return ReduceKind::Sum;
  Region &body = reduceOp->getRegion(0);
  if (body.empty())
    return ReduceKind::Sum;
  for (Operation &op : body.front()) {
    StringRef name = op.getName().getStringRef();
    if (name == "arith.maxf" || name == "arith.maximumf")
      return ReduceKind::Max;
    if (name == "arith.minf" || name == "arith.minimumf")
      return ReduceKind::Min;
  }
  return ReduceKind::Sum;
}

static bool analyzeReduction(triton::FuncOp func, ReductionInfo &info) {
  Operation *reduceOp = nullptr;
  func.walk([&](Operation *op) {
    if (op->getName().getStringRef() == "tt.reduce")
      reduceOp = op;
  });
  if (!reduceOp)
    return false;

  func.walk([&](MakeRangeOp op) {
    info.blockSize = cast<IntegerAttr>(op.getEndAttr()).getInt();
  });
  if (info.blockSize == 0)
    return false;

  SmallVector<LoadOp> loads;
  func.walk([&](LoadOp op) { loads.push_back(op); });
  if (loads.size() != 1)
    return false;

  SmallVector<StoreOp> stores;
  func.walk([&](StoreOp op) { stores.push_back(op); });
  if (stores.size() != 1)
    return false;

  BlockArgument loadBase = traceBasePointer(loads[0].getPtr());
  BlockArgument storeBase = traceBasePointer(stores[0].getPtr());
  if (!loadBase || !storeBase)
    return false;
  info.loadPtrArgIdx = loadBase.getArgNumber();
  info.storePtrArgIdx = storeBase.getArgNumber();

  auto loadType = dyn_cast<RankedTensorType>(loads[0].getType());
  if (!loadType)
    return false;
  info.elementType = loadType.getElementType();

  Block &body = func.getBody().front();
  for (auto it : llvm::enumerate(body.getArguments())) {
    Type t = it.value().getType();
    if (t.isInteger(32) && !isa<triton::PointerType>(t))
      info.nElementsArgIdx = it.index();
  }
  if (info.nElementsArgIdx < 0)
    return false;

  info.kind = detectReduceKind(reduceOp);
  return true;
}

static LogicalResult rewriteReduction(triton::FuncOp ttFunc,
                                      const ReductionInfo &info) {
  OpBuilder builder(ttFunc);
  Location loc = ttFunc.getLoc();
  MLIRContext *ctx = builder.getContext();

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

  Value c0 = builder.create<arith::ConstantOp>(
      loc, builder.getIntegerAttr(indexType, 0));
  Value c1 = builder.create<arith::ConstantOp>(
      loc, builder.getIntegerAttr(indexType, 1));
  Value cBS = builder.create<arith::ConstantOp>(
      loc, builder.getIntegerAttr(indexType, BS));

  Value nIdx = builder.create<arith::IndexCastOp>(
      loc, indexType, entry->getArgument(info.nElementsArgIdx));

  // Grid-stride loop: each core iterates blockIdx, blockIdx+nCores, ...
  Value bidx = builder.create<pto::GetBlockIdxOp>(loc, builder.getI64Type());
  Value nCores = builder.create<pto::GetBlockNumOp>(loc, builder.getI64Type());
  Value bidxIdx = builder.create<arith::IndexCastOp>(loc, indexType, bidx);
  Value strideIdx = builder.create<arith::IndexCastOp>(loc, indexType, nCores);
  Value nBlocks = builder.create<arith::CeilDivUIOp>(loc, nIdx, cBS);

  auto forOp = builder.create<scf::ForOp>(loc, bidxIdx, nBlocks, strideIdx);
  builder.setInsertionPointToStart(forOp.getBody());
  Value blockId = forOp.getInductionVar();

  Value blockStart = builder.create<arith::MulIOp>(loc, blockId, cBS);
  Value remaining = builder.create<arith::SubIOp>(loc, nIdx, blockStart);
  Value validCol = builder.create<arith::MinUIOp>(loc, remaining, cBS);

  auto tvType = pto::TensorViewType::get(
      ctx, {ShapedType::kDynamic, ShapedType::kDynamic}, elemType);
  auto pvType = pto::PartitionTensorViewType::get(
      ctx, {ShapedType::kDynamic, ShapedType::kDynamic}, elemType);
  SmallVector<int64_t, 2> dynValid = {-1, -1};
  auto tileBufType =
      pto::TileBufType::get(ctx, {1, BS}, elemType, memSpace, dynValid);

  SmallVector<Value> tvShape = {nBlocks, cBS};
  SmallVector<Value> tvStrides = {cBS, c1};

  // Load input tile
  Value tvIn = builder.create<pto::MakeTensorViewOp>(
      loc, tvType, entry->getArgument(info.loadPtrArgIdx), tvShape, tvStrides,
      pto::LayoutAttr());
  SmallVector<Value> pvOffsets = {blockId, c0};
  SmallVector<Value> pvSizes = {c1, validCol};
  Value pvIn = builder.create<pto::PartitionViewOp>(loc, pvType, tvIn,
                                                     pvOffsets, pvSizes);
  Value tileSrc =
      builder.create<pto::AllocTileOp>(loc, tileBufType, c1, validCol);
  builder.create<pto::TLoadOp>(loc, TypeRange{}, pvIn, tileSrc);

  // Reduce: trowsum/trowmax/trowmin -> 1xBS -> 1x1
  auto tmpTileBufType =
      pto::TileBufType::get(ctx, {1, BS}, elemType, memSpace);
  auto scalarTileBufType =
      pto::TileBufType::get(ctx, {1, 1}, elemType, memSpace);
  Value rsTmp =
      builder.create<pto::AllocTileOp>(loc, tmpTileBufType, Value(), Value());
  Value rsDst =
      builder.create<pto::AllocTileOp>(loc, scalarTileBufType, Value(), Value());

  switch (info.kind) {
  case ReduceKind::Sum:
    builder.create<pto::TRowSumOp>(loc, tileSrc, rsTmp, rsDst);
    break;
  case ReduceKind::Max:
    builder.create<pto::TRowMaxOp>(loc, tileSrc, rsTmp, rsDst);
    break;
  case ReduceKind::Min:
    builder.create<pto::TRowMinOp>(loc, tileSrc, rsTmp, rsDst);
    break;
  }

  // Extract scalar from 1x1 tile
  Value scalar = builder.create<pto::TGetValOp>(loc, elemType, rsDst, c0);

  // Scalar store: write to output[blockId] via 1x1 partition
  auto outTvType = pto::TensorViewType::get(
      ctx, {ShapedType::kDynamic, ShapedType::kDynamic}, elemType);
  auto outPvType = pto::PartitionTensorViewType::get(ctx, {1, 1}, elemType);
  Value tvOut = builder.create<pto::MakeTensorViewOp>(
      loc, outTvType, entry->getArgument(info.storePtrArgIdx),
      SmallVector<Value>{nBlocks, c1}, SmallVector<Value>{c1, c1},
      pto::LayoutAttr());
  Value pvOut = builder.create<pto::PartitionViewOp>(
      loc, outPvType, tvOut, SmallVector<Value>{blockId, c0},
      SmallVector<Value>{c1, c1});

  auto scalarTileType =
      pto::TileBufType::get(ctx, {1, 1}, elemType, memSpace);
  Value scalarTile =
      builder.create<pto::AllocTileOp>(loc, scalarTileType, Value(), Value());
  builder.create<pto::TSetValOp>(loc, scalarTile, c0, scalar);
  builder.create<pto::TStoreOp>(loc, TypeRange{}, scalarTile, pvOut);

  builder.setInsertionPointAfter(forOp);
  builder.create<func::ReturnOp>(loc);

  ttFunc.erase();
  return success();
}

//=== Holistic softmax pattern =================================================
//
// Detects the fused softmax kernel from the Triton tutorial:
// grid-stride row loop (get_program_id/get_num_programs), per-row:
// load, reduce(maxf), subf, exp, reduce(addf), divf, store.
// Generates PTO: tload, trowmax, tsubs, texp, trowsum, tdivs, tstore.

struct SoftmaxInfo {
  int64_t blockSize = 0;
  int inputPtrArgIdx = -1;
  int outputPtrArgIdx = -1;
  int inputRowStrideArgIdx = -1;
  int outputRowStrideArgIdx = -1;
  int nRowsArgIdx = -1;
  int nColsArgIdx = -1;
  Type elementType;
};

static bool analyzeSoftmax(triton::FuncOp func, SoftmaxInfo &info) {
  // Key differentiator: tt.get_num_programs is unique to the softmax pattern.
  bool hasGetNumPrograms = false;
  func.walk([&](Operation *op) {
    if (op->getName().getStringRef() == "tt.get_num_programs")
      hasGetNumPrograms = true;
  });
  if (!hasGetNumPrograms)
    return false;

  // Exactly 2 tt.reduce ops (maxf for row-max, addf for row-sum).
  SmallVector<Operation *> reduces;
  func.walk([&](Operation *op) {
    if (op->getName().getStringRef() == "tt.reduce")
      reduces.push_back(op);
  });
  if (reduces.size() != 2)
    return false;

  bool hasExp = false;
  func.walk([&](Operation *op) {
    if (op->getName().getStringRef() == "math.exp")
      hasExp = true;
  });
  if (!hasExp)
    return false;

  func.walk([&](MakeRangeOp op) {
    info.blockSize = cast<IntegerAttr>(op.getEndAttr()).getInt();
  });
  if (info.blockSize == 0)
    return false;

  SmallVector<LoadOp> loads;
  func.walk([&](LoadOp op) { loads.push_back(op); });
  if (loads.size() != 1)
    return false;

  SmallVector<StoreOp> stores;
  func.walk([&](StoreOp op) { stores.push_back(op); });
  if (stores.size() != 1)
    return false;

  auto loadType = dyn_cast<RankedTensorType>(loads[0].getType());
  if (!loadType)
    return false;
  info.elementType = loadType.getElementType();

  BlockArgument loadBase = traceBasePointer(loads[0].getPtr());
  BlockArgument storeBase = traceBasePointer(stores[0].getPtr());
  if (!loadBase || !storeBase)
    return false;
  info.inputPtrArgIdx = loadBase.getArgNumber();
  info.outputPtrArgIdx = storeBase.getArgNumber();

  Block &entryBlock = func.getBody().front();

  // n_rows: trace the scf.for upper bound through index_cast to a func arg.
  scf::ForOp forOp = nullptr;
  func.walk([&](scf::ForOp op) { forOp = op; });
  if (!forOp)
    return false;

  if (auto ic = forOp.getUpperBound().getDefiningOp<arith::IndexCastOp>()) {
    if (auto arg = dyn_cast<BlockArgument>(ic.getIn()))
      if (arg.getOwner() == &entryBlock)
        info.nRowsArgIdx = arg.getArgNumber();
  }
  if (info.nRowsArgIdx < 0)
    return false;

  // n_cols: the operand of tt.splat that feeds the arith.cmpi slt mask.
  func.walk([&](arith::CmpIOp cmpOp) {
    if (cmpOp.getPredicate() != arith::CmpIPredicate::slt)
      return;
    if (auto splat = cmpOp.getRhs().getDefiningOp<SplatOp>()) {
      if (auto arg = dyn_cast<BlockArgument>(splat.getSrc()))
        if (arg.getOwner() == &entryBlock)
          info.nColsArgIdx = arg.getArgNumber();
    }
  });
  if (info.nColsArgIdx < 0)
    return false;

  // Strides: find scalar tt.addptr(func_arg_ptr, muli(..., stride_arg)).
  func.walk([&](AddPtrOp addptr) {
    if (isa<RankedTensorType>(addptr.getPtr().getType()))
      return;
    auto baseArg = dyn_cast<BlockArgument>(addptr.getPtr());
    if (!baseArg || baseArg.getOwner() != &entryBlock)
      return;
    int basePtrIdx = baseArg.getArgNumber();
    auto muli = addptr.getOffset().getDefiningOp<arith::MulIOp>();
    if (!muli)
      return;
    for (Value operand : {muli.getLhs(), muli.getRhs()}) {
      auto strideArg = dyn_cast<BlockArgument>(operand);
      if (!strideArg || strideArg.getOwner() != &entryBlock)
        continue;
      if (basePtrIdx == info.inputPtrArgIdx)
        info.inputRowStrideArgIdx = strideArg.getArgNumber();
      else if (basePtrIdx == info.outputPtrArgIdx)
        info.outputRowStrideArgIdx = strideArg.getArgNumber();
    }
  });

  return info.inputRowStrideArgIdx >= 0 && info.outputRowStrideArgIdx >= 0;
}

static LogicalResult rewriteSoftmax(triton::FuncOp ttFunc,
                                    const SoftmaxInfo &info) {
  OpBuilder builder(ttFunc);
  Location loc = ttFunc.getLoc();
  MLIRContext *ctx = builder.getContext();

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

  Value c0 = builder.create<arith::ConstantOp>(
      loc, builder.getIntegerAttr(indexType, 0));
  Value c1 = builder.create<arith::ConstantOp>(
      loc, builder.getIntegerAttr(indexType, 1));
  Value cBS = builder.create<arith::ConstantOp>(
      loc, builder.getIntegerAttr(indexType, BS));

  Value nRowsIdx = builder.create<arith::IndexCastOp>(
      loc, indexType, entry->getArgument(info.nRowsArgIdx));
  Value nColsIdx = builder.create<arith::IndexCastOp>(
      loc, indexType, entry->getArgument(info.nColsArgIdx));
  Value inStrideIdx = builder.create<arith::IndexCastOp>(
      loc, indexType, entry->getArgument(info.inputRowStrideArgIdx));
  Value outStrideIdx = builder.create<arith::IndexCastOp>(
      loc, indexType, entry->getArgument(info.outputRowStrideArgIdx));

  // Grid-stride loop over rows
  Value bidx = builder.create<pto::GetBlockIdxOp>(loc, builder.getI64Type());
  Value nCores = builder.create<pto::GetBlockNumOp>(loc, builder.getI64Type());
  Value bidxIdx = builder.create<arith::IndexCastOp>(loc, indexType, bidx);
  Value strideIdx = builder.create<arith::IndexCastOp>(loc, indexType, nCores);

  auto forOp = builder.create<scf::ForOp>(loc, bidxIdx, nRowsIdx, strideIdx);
  builder.setInsertionPointToStart(forOp.getBody());
  Value rowIdx = forOp.getInductionVar();

  Value validCol = builder.create<arith::MinUIOp>(loc, nColsIdx, cBS);

  // Tile types
  auto tvType = pto::TensorViewType::get(
      ctx, {ShapedType::kDynamic, ShapedType::kDynamic}, elemType);
  auto pvType = pto::PartitionTensorViewType::get(
      ctx, {ShapedType::kDynamic, ShapedType::kDynamic}, elemType);
  SmallVector<int64_t, 2> dynValid = {-1, -1};
  auto tileBufType =
      pto::TileBufType::get(ctx, {1, BS}, elemType, memSpace, dynValid);
  auto tmpTileBufType =
      pto::TileBufType::get(ctx, {1, BS}, elemType, memSpace);
  auto scalarTileBufType =
      pto::TileBufType::get(ctx, {1, 1}, elemType, memSpace);

  // Input tensor view: shape=[nRows, inStride], strides=[inStride, 1]
  Value tvIn = builder.create<pto::MakeTensorViewOp>(
      loc, tvType, entry->getArgument(info.inputPtrArgIdx),
      SmallVector<Value>{nRowsIdx, inStrideIdx},
      SmallVector<Value>{inStrideIdx, c1}, pto::LayoutAttr());

  SmallVector<Value> pvOffsets = {rowIdx, c0};
  SmallVector<Value> pvSizes = {c1, validCol};
  Value pvIn = builder.create<pto::PartitionViewOp>(
      loc, pvType, tvIn, pvOffsets, pvSizes);

  // 1. Load row
  Value src = builder.create<pto::AllocTileOp>(loc, tileBufType, c1, validCol);
  builder.create<pto::TLoadOp>(loc, TypeRange{}, pvIn, src);

  // 2. Row max
  Value maxTmp =
      builder.create<pto::AllocTileOp>(loc, tmpTileBufType, Value(), Value());
  Value scalarMaxTile =
      builder.create<pto::AllocTileOp>(loc, scalarTileBufType, Value(), Value());
  builder.create<pto::TRowMaxOp>(loc, src, maxTmp, scalarMaxTile);
  Value maxVal =
      builder.create<pto::TGetValOp>(loc, elemType, scalarMaxTile, c0);

  // 3. Subtract max: dst = src - maxVal
  Value shifted =
      builder.create<pto::AllocTileOp>(loc, tileBufType, c1, validCol);
  builder.create<pto::TSubSOp>(loc, src, maxVal, shifted);

  // 4. Exp
  Value expTile =
      builder.create<pto::AllocTileOp>(loc, tileBufType, c1, validCol);
  builder.create<pto::TExpOp>(loc, shifted, expTile);

  // 5. Row sum
  Value sumTmp =
      builder.create<pto::AllocTileOp>(loc, tmpTileBufType, Value(), Value());
  Value scalarSumTile =
      builder.create<pto::AllocTileOp>(loc, scalarTileBufType, Value(), Value());
  builder.create<pto::TRowSumOp>(loc, expTile, sumTmp, scalarSumTile);
  Value sumVal =
      builder.create<pto::TGetValOp>(loc, elemType, scalarSumTile, c0);

  // 6. Divide by sum: dst = expTile / sumVal
  Value result =
      builder.create<pto::AllocTileOp>(loc, tileBufType, c1, validCol);
  builder.create<pto::TDivSOp>(loc, expTile, sumVal, result);

  // 7. Store result
  Value tvOut = builder.create<pto::MakeTensorViewOp>(
      loc, tvType, entry->getArgument(info.outputPtrArgIdx),
      SmallVector<Value>{nRowsIdx, outStrideIdx},
      SmallVector<Value>{outStrideIdx, c1}, pto::LayoutAttr());
  Value pvOut = builder.create<pto::PartitionViewOp>(
      loc, pvType, tvOut, pvOffsets, pvSizes);
  builder.create<pto::TStoreOp>(loc, TypeRange{}, result, pvOut);

  // Return after scf.for
  builder.setInsertionPointAfter(forOp);
  builder.create<func::ReturnOp>(loc);

  ttFunc.erase();
  return success();
}

//=== Holistic matmul pattern (detection only) ================================
//
// Detects Triton matmul kernels: tt.dot in a scf.for accumulation loop with
// two 2D tile loads and one tile store. Full rewrite requires PTOAS matmul
// ground-truth for validation, so detection is provided for early rejection
// with a clear diagnostic.

struct MatmulInfo {
  int64_t blockM = 0;
  int64_t blockN = 0;
  int64_t blockK = 0;
  Type elementType;
};

static bool analyzeMatmul(triton::FuncOp func, MatmulInfo &info) {
  bool hasDot = false;
  func.walk([&](Operation *op) {
    if (op->getName().getStringRef() == "tt.dot")
      hasDot = true;
  });
  return hasDot;
}

static LogicalResult rewriteMatmul(triton::FuncOp ttFunc,
                                   const MatmulInfo & /*info*/) {
  return ttFunc.emitError("matmul pattern detected but holistic rewrite not "
                          "yet implemented; awaiting PTOAS ground-truth");
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

    // Phase 1: Holistic pattern-matched rewrites.
    // Each recognized kernel idiom is fully converted to PTO here.
    SmallVector<triton::FuncOp> allFuncs;
    module.walk([&](triton::FuncOp func) { allFuncs.push_back(func); });
    for (auto func : allFuncs) {
      VecAddInfo vaInfo;
      if (analyzeVecAddPattern(func, vaInfo)) {
        if (failed(rewriteVecAdd(func, vaInfo)))
          return signalPassFailure();
        continue;
      }
      ElementWiseUnaryInfo ewInfo;
      if (analyzeElementWiseUnary(func, ewInfo)) {
        if (failed(rewriteElementWiseUnary(func, ewInfo)))
          return signalPassFailure();
        continue;
      }
      SoftmaxInfo smInfo;
      if (analyzeSoftmax(func, smInfo)) {
        if (failed(rewriteSoftmax(func, smInfo)))
          return signalPassFailure();
        continue;
      }
      ReductionInfo redInfo;
      if (analyzeReduction(func, redInfo)) {
        if (failed(rewriteReduction(func, redInfo)))
          return signalPassFailure();
        continue;
      }
      MatmulInfo mmInfo;
      if (analyzeMatmul(func, mmInfo)) {
        if (failed(rewriteMatmul(func, mmInfo)))
          return signalPassFailure();
        continue;
      }
      // No holistic pattern matched; fall through to Phase 2.
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
    patterns.add<ConvertTTGetProgramIdOp, ConvertTTGetNumProgramsOp>(
        typeConverter, &getContext());
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
