# Triton→PTO (MLIR-based prototype)

This repository contains an **MLIR-based prototype** for lowering
**Triton MLIR** (`tt.*` / `ttg.*` dialects) into a **PTO-style**
representation, using a small custom MLIR dialect and a standalone
command-line tool.

> Note: an earlier Python-based translator existed in this repo but has
> been removed; the current direction is **C++ + LLVM/MLIR only** under
> `mlir_tool/`.

## Layout

- `mlir_tool/` – standalone CMake/MLIR project:
  - `CMakeLists.txt` – top-level MLIR project file.
  - `include/pto-mlir/Dialect/PTO/` – headers/TableGen definitions for the
    **PTO MLIR dialect**:
    - `PTOOps.td` – declares a minimal `pto` dialect and a small set of
      ops (`tload`, `tstore`, `tmatmul`, `tadd`, …).
    - `PTODialect.h` – C++ declaration of `pto::PTODialect`.
  - `lib/Dialect/PTO/` – dialect implementation:
    - `PTODialect.cpp` – registers `PTODialect` with MLIR (currently a
      very thin shell).
  - `lib/Conversion/TritonToPTO/` – **Triton→PTO conversion pass**:
    - `TritonToPTO.cpp` – a simple name-based pass that rewrites
      `tt.load`→`pto.tload`, `tt.store`→`pto.tstore`,
      `tt.dot`→`pto.tmatmul`, `tt.add`→`pto.tadd`.
  - `tools/triton-to-pto-mlir/` – CLI driver:
    - `triton-to-pto-mlir.cpp` – parses an MLIR module, runs the
      `convert-triton-to-pto` pass, and prints the result.
  - `test/` – basic MLIR tests:
    - `vec_add_triton.mlir` – lit/FileCheck test for a simple vector add
      kernel (Triton `tt.*` ops → `pto.*` ops).
    - `lit.cfg.py` – lit configuration for `mlir_tool/test/`.

## Prerequisites

- LLVM + MLIR with CMake package configs (e.g. Homebrew `llvm`):
  - `MLIR_DIR` must point to the MLIR CMake config, typically:
    - `/usr/local/opt/llvm/lib/cmake/mlir` (Homebrew)
  - `FileCheck`, `llvm-lit`, and other LLVM tools on your `PATH` if you
    want to run lit tests.

## Build

From the `mlir_tool/` directory:

```bash
cd mlir_tool
cmake -S . -B build -DMLIR_DIR=/usr/local/opt/llvm/lib/cmake/mlir
cmake --build build
```

This builds:

- `libPTOMLIRDialect.a` – the PTO dialect library.
- `libTritonToPTOPass.a` – the Triton→PTO conversion pass.
- `triton-to-pto-mlir` – the CLI tool:
  - `build/tools/triton-to-pto-mlir/triton-to-pto-mlir`

## Usage

Once built, you can run the tool on an MLIR file (for now, Triton
dialect ops are treated as **unregistered** but still parsed):

```bash
cd mlir_tool
build/tools/triton-to-pto-mlir/triton-to-pto-mlir test/vec_add_triton.mlir -o /tmp/vec_add_pto.mlir
cat /tmp/vec_add_pto.mlir
```

Given the current skeleton pass, you should see that:

- `tt.load` ops were renamed to `pto.tload`.
- `tt.add` ops were renamed to `pto.tadd`.
- `tt.store` ops were renamed to `pto.tstore`.

No type/layout-aware logic is implemented yet; the pass simply preserves
operands, result types, and attributes while changing the op name.

## Tests (lit / FileCheck)

If Python and lit/FileCheck are available, you can run the lit-based
test suite:

```bash
cd mlir_tool
cmake --build build --target check-mlir-tool
```

This uses:

- `test/lit.cfg.py` – to configure lit for `.mlir` tests.
- `test/vec_add_triton.mlir` – which contains:

  ```mlir
  // RUN: triton-to-pto-mlir %s -o - | FileCheck %s
  ```

  and `CHECK` lines verifying that the output contains `pto.tload`,
  `pto.tadd`, and `pto.tstore` as expected.

## Next steps / limitations

- The PTO dialect is currently **very minimal**:
  - No custom types (`!pto.tile`, `!pto.memref`, `!pto.event`) wired yet.
  - No custom assembly formats beyond the bare op names.
- The Triton→PTO pass is **name-based only**:
  - It does not yet use Triton’s real MLIR dialects or semantics.
  - It does not yet map shapes/layouts or scalar control flow.

The intent is for this prototype to serve as a starting point for a more
complete MLIR-based Triton→PTO pipeline that can eventually target the
PTO ISA / PTO-AS as specified in the upstream PTO Tile Lib.

