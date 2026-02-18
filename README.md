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
  - `CMakeLists.txt` – top-level MLIR project file; **depends on PTOAS** for
    the PTO dialect (see Build below).
  - **PTO dialect**: provided by **PTOAS** as the single source of truth. This
    repo does not ship its own PTO TableGen or dialect implementation; it
    links to PTOAS’s `PTOIR` library and uses PTOAS’s types/ops so that emitted
    `.pto` is directly consumable by `ptoas`.
  - `lib/Conversion/TritonToPTO/` – **Triton→PTO conversion pass**:
    - `TritonToPTO.cpp` – lowers Triton IR to PTO dialect (e.g.
      `tt.load`→`pto.tload`, `tt.store`→`pto.tstore`, `tt.add`→`pto.tadd`).
  - `tools/triton-to-pto-mlir/` – CLI driver:
    - `triton-to-pto-mlir.cpp` – parses an MLIR module, runs the
      `convert-triton-to-pto` pass, and prints the result.
  - `test/` – lit/FileCheck tests and E2E script:
    - `vec_add_triton.mlir`, `reduce_sum_triton.mlir`, etc. – lit tests.
    - `e2e_run_ptoas_sim.sh` – Triton → converter → .pto → PTOAS docker/sim.

## Prerequisites

- **LLVM + MLIR** with CMake package configs (e.g. Homebrew `llvm`):
  - `MLIR_DIR` must point to the MLIR CMake config.
  - `FileCheck` and `lit` on `PATH` (or set `LIT_FILECHECK` / `LLVM_EXTERNAL_LIT`) for lit tests.
- **LLVM 19 + MLIR** – PTOAS expects **LLVM 19** (see PTOAS `CMakeLists.txt`). Use a build or install of LLVM 19 with MLIR enabled. From your `llvm-project` checkout you can build LLVM 19 + MLIR using the script under **Building LLVM 19** below.
- **Triton** – the Triton MLIR dialects are built from a separate Triton source tree. Set `TRITON_SOURCE_DIR` (default: `../../triton` from `mlir_tool/`) to that checkout. **For LLVM 19**, use a Triton version that is compatible with LLVM 19 (e.g. a branch or tag that supports it); check Triton’s repo for the appropriate revision.
- **PTOAS** – the PTO dialect and TableGen come from PTOAS. Either:
  - **Option A (subproject):** set `PTOAS_SOURCE_DIR` to the PTOAS repo root. CMake will add PTOAS as a subproject and build `PTOIR`; you must point `LLVM_DIR` and `MLIR_DIR` at an LLVM 19 (+ MLIR) build or install.
  - **Option B (installed):** install PTOAS, then set `PTOAS_DIR` to `lib/cmake/PTOAS` so `find_package(PTOAS)` succeeds.

## Building LLVM 19

PTOAS requires **LLVM 19**. With `llvm-project` at `LLVM_PROJECT_ROOT` (e.g. `../llvm-project` from `triton-to-pto`):

1. **Optional:** use a clean LLVM 19 tree via worktree (avoids touching main):
   ```bash
   cd "${LLVM_PROJECT_ROOT}"
   git fetch origin release/19.x
   git worktree add llvm-19-worktree origin/release/19.x
   ```
2. **Build and install LLVM 19 + MLIR:**
   ```bash
   cd mlir_tool
   LLVM_PROJECT_ROOT=/path/to/llvm-project ./scripts/build_llvm19.sh
   ```
   This configures and builds LLVM with `-DLLVM_ENABLE_PROJECTS=mlir`, then installs to `llvm-project/install-19` (or `LLVM_INSTALL_PREFIX`). Use that install for `MLIR_DIR` and `LLVM_DIR` below.

## Build

From the `mlir_tool/` directory:

**With PTOAS as subproject (Option A), using your LLVM 19 install:**

```bash
cd mlir_tool
# For LLVM 19, use an LLVM-19-compatible Triton checkout (set TRITON_SOURCE_DIR if not ../../triton)
cmake -S . -B build \
  -DMLIR_DIR=/path/to/llvm-project/install-19/lib/cmake/mlir \
  -DLLVM_DIR=/path/to/llvm-project/install-19/lib/cmake/llvm \
  -DPTOAS_SOURCE_DIR=/path/to/PTOAS
cmake --build build
```

**With installed PTOAS (Option B):**

```bash
cd mlir_tool
cmake -S . -B build -DMLIR_DIR=/path/to/mlir -DPTOAS_DIR=/path/to/install/lib/cmake/PTOAS
cmake --build build
```

This builds:

- `libTritonToPTOPass.a` – the Triton→PTO conversion pass (links to PTOAS’s PTOIR).
- `triton-to-pto-mlir` – the CLI tool: `build/tools/triton-to-pto-mlir/triton-to-pto-mlir`

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

## E2E testing with PTOAS simulator

You can run the full pipeline **Triton → triton-to-pto-mlir → PTOAS (.pto → ptoas → sim → compare)** using:

- **PTOAS repo** with `docker/run_sim_example.sh` and `test/npu_validation/scripts/generate_testcase.py`
- **Docker image** `ptoas:py3.11` (build per PTOAS’s `docker/README.md`)
- **Converter binary** on `PATH` or set `TRITON_TO_PTO_MLIR`

From the repo root:

```bash
export PTOAS_ROOT=/path/to/PTOAS
# Optional: export TRITON_TO_PTO_MLIR=/path/to/triton-to-pto-mlir
mlir_tool/test/e2e_run_ptoas_sim.sh
```

Default input is `mlir_tool/test/vec_add_e2e_triton.mlir` (32×32 vec_add, 3 args, single block). The converter emits **native PTOAS** `.pto` (same dialect and assembly format as PTOAS; e.g. `!pto.tile_buf<loc=vec, dtype=..., rows=..., cols=...>`). The script writes it and runs `docker run ... ptoas:py3.11 bash docker/run_sim_example.sh <path>.pto`. Optionally set `USE_PTO_BRIDGE=1` to run the Python bridge on converter output (legacy). Success means the container exits 0 and the simulator compare passes.

## Reproducible Docker build

A Docker-based build in `docker/` fixes **LLVM** to the same version as PTOAS (`llvmorg-19.1.7`) and adds a **Triton** checkout at a configurable ref to form a reproducible environment. The image extends the PTOAS runtime (`ptoas:py3.11`) with `triton-to-pto-mlir` installed.

- Build the PTOAS image first, then from the triton-to-pto repo root:  
  `docker build -f docker/Dockerfile -t triton-to-pto:py3.11 .`
- Build args: `LLVM_TAG` (default `llvmorg-19.1.7`), `TRITON_REF` (default `main`; use a known LLVM-19-compatible ref for reproducibility), optional `PTOAS_REF`.
- See [docker/README.md](docker/README.md) for build order, build args, and how to run the converter and E2E inside the image.

## Notes

- **LLVM/MLIR version:** PTOAS and triton-to-pto must use the same LLVM/MLIR build or install to avoid ABI/header mismatches.
- The Triton→PTO pass lowers Triton IR to PTOAS’s dialect ops and types; further ISA/layout mapping is out of scope of this repo.

