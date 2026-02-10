//===- TritonToPTO.cpp - Triton→PTO conversion pass ----------------------===//
//
// Skeleton MLIR pass that will eventually lower Triton dialect ops (tt.*, ttg.*)
// to PTO ops in the PTO dialect. At the moment this file only wires up a
// placeholder pass and demonstrates how to walk Triton ops.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace {

struct TritonToPTO
    : public PassWrapper<TritonToPTO, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TritonToPTO)

  StringRef getArgument() const final { return "convert-triton-to-pto"; }
  StringRef getDescription() const final {
    return "Convert Triton dialect ops (tt.*, ttg.*) to PTO dialect ops";
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    // Minimal, name-based lowering: map a handful of Triton ops to PTO
    // counterparts by renaming them to pto.* while preserving operands and
    // result types. This is intentionally simplistic and meant as a starting
    // point before introducing proper OpConversionPatterns.
    module.walk([&](Operation *op) {
      StringRef name = op->getName().getStringRef();

      StringRef newName;
      if (name == "tt.load")
        newName = "pto.tload";
      else if (name == "tt.store")
        newName = "pto.tstore";
      else if (name == "tt.dot")
        newName = "pto.tmatmul";
      else if (name == "tt.add")
        newName = "pto.tadd";
      else if (name == "arith.addf")
        newName = "pto.tadd";
      else
        return;

      OpBuilder builder(op);
      OperationState st(op->getLoc(), newName);
      st.addOperands(op->getOperands());
      st.addTypes(op->getResultTypes());
      for (auto &namedAttr : op->getAttrs())
        st.addAttribute(namedAttr.getName(), namedAttr.getValue());
      Operation *newOp = builder.create(st);
      op->replaceAllUsesWith(newOp);
      op->erase();
    });
  }
};

} // namespace

std::unique_ptr<Pass> createTritonToPTOPass() {
  return std::make_unique<TritonToPTO>();
}

