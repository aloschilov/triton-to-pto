//===- triton-to-pto-mlir.cpp - Triton→PTO MLIR tool ---------------------===//
//
// Standalone CLI tool that loads a Triton MLIR module, runs a (stub)
// Triton→PTO conversion pass, and prints the resulting module. This is a
// skeleton suitable for wiring into a full PTO dialect/ISA lowering.
//
// Usage (once built):
//   triton-to-pto-mlir input.mlir -o output.mlir
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Support/ToolUtilities.h"
#include "mlir/Tools/mlir-translate/Translation.h"

#include "pto-mlir/Dialect/PTO/PTODialect.h"

using namespace mlir;

// Forward-declared from TritonToPTO.cpp
std::unique_ptr<Pass> createTritonToPTOPass();

int main(int argc, char **argv) {
  DialectRegistry registry;
  // Register builtin + PTO dialects; Triton dialects can be registered here
  // when linked in (e.g. registry.insert<triton::TritonDialect, ...>();).
  registry.insert<pto::PTODialect>();

  MLIRContext context(registry);

  llvm::InitLLVM y(argc, argv);

  std::string inputFilename;
  std::string outputFilename;

  // Very small argument parser: [input] [-o output]
  if (argc > 1)
    inputFilename = argv[1];
  for (int i = 1; i < argc; ++i) {
    llvm::StringRef arg(argv[i]);
    if (arg == "-o" && i + 1 < argc) {
      outputFilename = argv[++i];
    }
  }

  std::string errorMessage;
  auto input = openInputFile(inputFilename, &errorMessage);
  if (!input) {
    llvm::errs() << \"Error opening input: \" << errorMessage << \"\\n\";
    return 1;
  }

  auto output = openOutputFile(outputFilename.empty() ? \"-\" : outputFilename,
                               &errorMessage);
  if (!output) {
    llvm::errs() << \"Error opening output: \" << errorMessage << \"\\n\";
    return 1;
  }

  // Parse module from input
  OwningOpRef<ModuleOp> module =
      parseSourceFile<ModuleOp>(input->getMemBufferRef(), &context);
  if (!module) {
    llvm::errs() << \"Failed to parse input MLIR module.\\n\";
    return 1;
  }

  PassManager pm(&context);
  pm.addPass(createTritonToPTOPass());

  if (failed(pm.run(*module))) {
    llvm::errs() << \"Triton→PTO pass pipeline failed.\\n\";
    return 1;
  }

  module->print(output->os());
  output->keep();
  return 0;
}

