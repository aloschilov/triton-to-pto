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
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Support/ToolUtilities.h"
#include "mlir/Tools/mlir-translate/Translation.h"

#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "PTO/IR/PTO.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

using namespace mlir;

// Forward-declared from TritonToPTO.cpp
std::unique_ptr<Pass> createTritonToPTOPass();

int main(int argc, char **argv) {
  DialectRegistry registry;
  registry.insert<mlir::pto::PTODialect,
                  mlir::triton::TritonDialect,
                  mlir::arith::ArithDialect,
                  mlir::func::FuncDialect,
                  mlir::scf::SCFDialect>();

  MLIRContext context(registry);
  context.allowUnregisteredDialects(true);
  context.getOrLoadDialect<mlir::triton::TritonDialect>();
  context.getOrLoadDialect<mlir::pto::PTODialect>();
  context.getOrLoadDialect<mlir::arith::ArithDialect>();
  context.getOrLoadDialect<mlir::func::FuncDialect>();
  context.getOrLoadDialect<mlir::scf::SCFDialect>();

  llvm::InitLLVM initLLVM(argc, argv);

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

  auto output = openOutputFile(outputFilename.empty() ? "-" : outputFilename,
                               &errorMessage);
  if (!output) {
    llvm::errs() << "Error opening output: " << errorMessage << "\n";
    return 1;
  }

  // Parse module from input
  if (inputFilename.empty()) {
    llvm::errs() << "No input file specified\n";
    return 1;
  }
  ParserConfig config(&context);
  OwningOpRef<ModuleOp> module =
      parseSourceFile<ModuleOp>(inputFilename, config);
  if (!module) {
    llvm::errs() << "Failed to parse input MLIR module.\n";
    return 1;
  }

  PassManager pm(&context);
  pm.addPass(createTritonToPTOPass());

  if (failed(pm.run(*module))) {
    llvm::errs() << "Triton->PTO pass pipeline failed.\n";
    return 1;
  }

  module->print(output->os());
  output->keep();
  return 0;
}

