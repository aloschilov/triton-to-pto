//===- TritonToPTO.cpp - Triton→PTO conversion pass ----------------------===//
//
// Full lowering of Triton IR to PTO-AS-aligned MLIR: converts tt.func with
// pointer args to func.func with !pto.memref, and tt.load/arith.addf/tt.store
// to pto.tload/pto.tadd/pto.tstore. Infrastructure ops (get_program_id,
// make_range, splat, addptr, tensor arith) are absorbed/erased.
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
#include "pto-mlir/Dialect/PTO/PTODialect.h"
#define GET_OP_CLASSES
#include "PTOOps.h.inc"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/Support/Casting.h"

#include <optional>

using namespace mlir;
using namespace mlir::triton;

namespace {

//--- Block size inference ----------------------------------------------------

static int64_t inferBlockSize(ModuleOp module) {
  int64_t blockSize = 1024; // default
  module.walk([&](Operation *op) {
    if (auto range = dyn_cast<MakeRangeOp>(op)) {
      if (auto endAttr = range.getEndAttr()) {
        if (auto intAttr = dyn_cast<IntegerAttr>(endAttr))
          blockSize = intAttr.getInt();
      }
      return WalkResult::interrupt();
    }
    if (auto cst = dyn_cast<arith::ConstantOp>(op)) {
      if (cst.getType().isInteger(32)) {
        if (auto intAttr = dyn_cast<IntegerAttr>(cst.getValue()))
          blockSize = intAttr.getInt();
      }
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return blockSize;
}

//--- Tile shape helper -------------------------------------------------------

static std::pair<int64_t, int64_t> computeTileShape(int64_t numElements,
                                                     Type elementType) {
  // PTO alignment: Cols * sizeof(element) % 32 == 0. For f32 (4 bytes), Cols % 8 == 0.
  int64_t elemSize = 4;
  if (elementType.isF32() || elementType.isInteger(32))
    elemSize = 4;
  else if (elementType.isF16() || elementType.isInteger(16))
    elemSize = 2;
  else if (elementType.isF64() || elementType.isInteger(64))
    elemSize = 8;
  int64_t align = 32 / elemSize; // 8 for f32
  int64_t cols = std::min(numElements, (int64_t)64);
  while (cols > 0 && (numElements % cols != 0 || cols % align != 0))
    --cols;
  if (cols == 0)
    cols = align;
  int64_t rows = numElements / cols;
  return {rows, cols};
}

//--- Type converter ----------------------------------------------------------

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
        auto [rows, cols] = computeTileShape(n, elem);
        return pto::TileBufType::get(type.getContext(), {rows, cols}, elem);
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
      auto [rows, cols] = computeTileShape(n, elem);
      return pto::TileBufType::get(type.getContext(), {rows, cols}, elem);
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
    rewriter.applySignatureConversion(&*newFunc.getBody().begin(),
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
    int64_t rows = tileBufType.getDimSize(0);
    int64_t cols = tileBufType.getDimSize(1);
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
    SmallVector<Value> shape = {cRows, cCols};
    SmallVector<Value> strides = {cCols, c1};
    Value tv = rewriter.create<pto::PTOMakeTensorViewOp>(
        loc, tensorViewType, ptr, shape, strides);
    auto partType = pto::PartitionTensorViewType::get(
        rewriter.getContext(), {rows, cols}, tileBufType.getElementType());
    SmallVector<Value> offsets = {c0, c0};
    SmallVector<Value> sizes = {cRows, cCols};
    Value pv = rewriter.create<pto::PTOPartitionViewOp>(
        loc, partType, tv, offsets, sizes);
    Value tile = rewriter.create<pto::PTOAllocTileOp>(loc, tileBufType);
    rewriter.create<pto::PTOTLoadOp>(loc, pv, tile);
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
    Value dst = rewriter.create<pto::PTOAllocTileOp>(op.getLoc(), tileBufType);
    rewriter.create<pto::PTOTAddOp>(op.getLoc(), adaptor.getLhs(),
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
    int64_t rows = tileBufType.getDimSize(0);
    int64_t cols = tileBufType.getDimSize(1);

    // 1. trowsum: RxC -> Rx1 (sum across columns for each row)
    auto rowResultType = pto::TileBufType::get(ctx, {rows, 1}, elemType);
    Value rsTmp = rewriter.create<pto::PTOAllocTileOp>(loc, tileBufType);
    Value rsDst = rewriter.create<pto::PTOAllocTileOp>(loc, rowResultType);
    rewriter.create<pto::PTOTRowSumOp>(loc, tile, rsTmp, rsDst);

    // 2. tcolsum: Rx1 -> 1x1 (sum across rows)
    auto scalarTileType = pto::TileBufType::get(ctx, {1, 1}, elemType);
    Value csTmp = rewriter.create<pto::PTOAllocTileOp>(loc, rowResultType);
    Value csDst = rewriter.create<pto::PTOAllocTileOp>(loc, scalarTileType);
    rewriter.create<pto::PTOTColSumOp>(loc, rsDst, csTmp, csDst);

    // 3. tgetval: extract scalar from 1x1 tile at offset 0
    Value c0_idx = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getIntegerAttr(indexType, 0));
    Value scalar =
        rewriter.create<pto::PTOTGetValOp>(loc, resultType, csDst, c0_idx);
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
      int64_t rows = tileBufType.getDimSize(0);
      int64_t cols = tileBufType.getDimSize(1);
      Value cRows = rewriter.create<arith::ConstantOp>(
          loc, rewriter.getIntegerAttr(indexType, rows));
      Value cCols = rewriter.create<arith::ConstantOp>(
          loc, rewriter.getIntegerAttr(indexType, cols));
      // PTOAS-aligned: variadic shape/strides keyword syntax
      auto tensorViewType = pto::TensorViewType::get(
          ctx, {rows, cols}, tileBufType.getElementType());
      SmallVector<Value> shape = {cRows, cCols};
      SmallVector<Value> strides = {cCols, c1};
      Value tv = rewriter.create<pto::PTOMakeTensorViewOp>(
          loc, tensorViewType, ptr, shape, strides);
      auto partType = pto::PartitionTensorViewType::get(
          ctx, {rows, cols}, tileBufType.getElementType());
      SmallVector<Value> offsets = {c0, c0};
      SmallVector<Value> sizes = {cRows, cCols};
      Value pv = rewriter.create<pto::PTOPartitionViewOp>(
          loc, partType, tv, offsets, sizes);
      rewriter.create<pto::PTOTStoreOp>(loc, value, pv);
    } else {
      // Scalar store: PTOAS-aligned make_tensor_view + partition_view +
      //               alloc_tile(1x1) + tsetval + tstore
      std::optional<ScalarStoreIndices> indices =
          getScalarStoreIndices(origPtr, op, rewriter);
      if (!indices)
        return rewriter.notifyMatchFailure(op,
                                           "could not get scalar store indices");
      Type elemType = value.getType();
      auto scalarTileType = pto::TileBufType::get(ctx, {1, 1}, elemType);
      auto tvType = pto::TensorViewType::get(
          ctx, {ShapedType::kDynamic, ShapedType::kDynamic}, elemType);
      auto pvType =
          pto::PartitionTensorViewType::get(ctx, {1, 1}, elemType);

      // tensor_view over base pointer; shape = [row+1, 1] to be large enough
      Value rowPlus1 =
          rewriter.create<arith::AddIOp>(loc, indices->row, c1);
      SmallVector<Value> tvShape = {rowPlus1, c1};
      SmallVector<Value> tvStrides = {c1, c1};
      Value tv = rewriter.create<pto::PTOMakeTensorViewOp>(
          loc, tvType, indices->basePtr, tvShape, tvStrides);

      // partition_view at [row, 0] with size [1, 1]
      SmallVector<Value> pvOffsets = {indices->row, c0};
      SmallVector<Value> pvSizes = {c1, c1};
      Value pv = rewriter.create<pto::PTOPartitionViewOp>(
          loc, pvType, tv, pvOffsets, pvSizes);

      // alloc 1x1 tile, set scalar value at offset 0, then tstore
      Value tile = rewriter.create<pto::PTOAllocTileOp>(loc, scalarTileType);
      rewriter.create<pto::PTOTSetValOp>(loc, tile, c0, value);
      rewriter.create<pto::PTOTStoreOp>(loc, tile, pv);
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
    // Allocate destination tile and broadcast scalar into it
    Value tile = rewriter.create<pto::PTOAllocTileOp>(loc, tileBufType);
    rewriter.create<pto::PTOTExpandsOp>(loc, scalar, tile);
    rewriter.replaceOp(op, tile);
    return success();
  }
};

// Convert tt.get_program_id -> pto.get_block_idx (PTOAS-aligned).
struct ConvertTTGetProgramIdOp
    : public OpConversionPattern<GetProgramIdOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(GetProgramIdOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    // pto.get_block_idx returns i64
    Value pid64 =
        rewriter.create<pto::PTOGetBlockIdxOp>(loc, rewriter.getI64Type());
    // Triton get_program_id returns i32 -- truncate
    Value pid32 =
        rewriter.create<arith::TruncIOp>(loc, rewriter.getI32Type(), pid64);
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
    // Replace each result with a legal value so remaining dead ops can be erased.
    for (OpResult result : op->getResults()) {
      Type oldType = result.getType();
      Type newType = this->getTypeConverter()->convertType(oldType);
      if (!newType)
        newType = oldType;
      if (newType.isIntOrIndex()) {
        Value c0 = rewriter.create<arith::ConstantOp>(
            op.getLoc(), newType,
            rewriter.getIntegerAttr(newType, 0));
        rewriter.replaceAllUsesWith(result, c0);
      }
      // For tensor/pointer types, do not replace; only GetProgramIdOp has i32 result.
    }
    rewriter.eraseOp(op);
    return success();
  }
};

struct EraseArithMuliOp : public OpConversionPattern<arith::MulIOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arith::MulIOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Result may be used by splat; replace with 0 so uses get a legal value.
    Type resultType = op.getResult().getType();
    if (resultType.isIntOrIndex()) {
      Value c0 = rewriter.create<arith::ConstantOp>(
          op.getLoc(), rewriter.getIntegerAttr(resultType, 0));
      rewriter.replaceAllUsesWith(op.getResult(), c0);
    }
    rewriter.eraseOp(op);
    return success();
  }
};

struct EraseArithAddIOp : public OpConversionPattern<arith::AddIOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arith::AddIOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.eraseOp(op);
    return success();
  }
};

struct EraseArithCmpiOp : public OpConversionPattern<arith::CmpIOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arith::CmpIOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.eraseOp(op);
    return success();
  }
};

//--- Pass --------------------------------------------------------------------

struct TritonToPTO : public PassWrapper<TritonToPTO, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TritonToPTO)

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<pto::PTODialect, triton::TritonDialect, func::FuncDialect,
                    arith::ArithDialect, mlir::scf::SCFDialect>();
  }

  StringRef getArgument() const final { return "convert-triton-to-pto"; }
  StringRef getDescription() const final {
    return "Convert Triton dialect to PTO dialect (full lowering)";
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    // Ensure PTO dialect is loaded so custom types are registered with the uniquer.
    getContext().getOrLoadDialect<pto::PTODialect>();
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
    patterns.add<EraseArithMuliOp, EraseArithAddIOp, EraseArithCmpiOp>(
        typeConverter, &getContext());

    ConversionTarget target(getContext());
    target.addLegalDialect<pto::PTODialect, func::FuncDialect,
                           arith::ArithDialect>();
    target.addIllegalDialect<triton::TritonDialect>();
    // arith.muli: legal for scalars (e.g. pid * block_size), illegal for tensors
    target.addDynamicallyLegalOp<arith::MulIOp>([](arith::MulIOp op) {
      return !llvm::isa<RankedTensorType>(op.getResult().getType());
    });
    target.addDynamicallyLegalOp<arith::ConstantOp>([](arith::ConstantOp op) {
      return std::optional<bool>(
          !llvm::isa<RankedTensorType>(op.getResult().getType()));
    });
    target.addDynamicallyLegalOp<arith::AddFOp>([](arith::AddFOp op) {
      return std::optional<bool>(
          !llvm::isa<RankedTensorType>(op.getResult().getType()));
    });
    target.addDynamicallyLegalOp<arith::AddIOp>([](arith::AddIOp op) {
      return std::optional<bool>(
          !llvm::isa<RankedTensorType>(op.getResult().getType()));
    });
    target.addDynamicallyLegalOp<arith::CmpIOp>([](arith::CmpIOp op) {
      return std::optional<bool>(
          !llvm::isa<RankedTensorType>(op.getResult().getType()));
    });
    target.addLegalOp<UnrealizedConversionCastOp>();
    // Use built-in SCF structural type conversions for scf.for/scf.yield.
    mlir::scf::populateSCFStructuralTypeConversionsAndLegality(
        typeConverter, patterns, target);

    if (failed(applyPartialConversion(module, target, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> createTritonToPTOPass() {
  return std::make_unique<TritonToPTO>();
}
