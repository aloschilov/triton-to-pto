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
    module.walk([](Operation *op) {
      auto name = op->getName().getStringRef();

      auto mapName = [](StringRef oldName) -> Optional<std::string> {
        if (oldName == "tt.load")
          return std::string("pto.tload");
        if (oldName == "tt.store")
          return std::string("pto.tstore");
        if (oldName == "tt.dot")
          return std::string("pto.tmatmul");
        if (oldName == "tt.add")
          return std::string("pto.tadd");
        // Leave all other ops unchanged in this skeleton.
        return llvm::None;
      };

      if (Optional<std::string> newName = mapName(name)) {
        OpBuilder builder(op);
        OperationState st(op->getLoc(), *newName);
        st.addOperands(op->getOperands());
        st.addTypes(op->getResultTypes());
        for (auto &namedAttr : op->getAttrs())
          st.addAttribute(namedAttr.getName(), namedAttr.getValue());
        Operation *newOp = builder.create(st);
        op->replaceAllUsesWith(newOp);
        op->erase();
      }
    });
  }
};

} // namespace

std::unique_ptr<Pass> createTritonToPTOPass() {
  return std::make_unique<TritonToPTO>();
}

