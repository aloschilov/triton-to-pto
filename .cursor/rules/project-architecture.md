# Project Architecture

## What This Project Does

`triton-to-pto` lowers **Triton MLIR** (`tt.*` dialect) into the **PTO dialect** (from PTOAS), which then gets assembled into Ascend NPU kernels via the PTOAS toolchain.

## Pipeline

```
Triton Python  -->  Triton MLIR (.mlir)
                        |
                   triton-to-pto-mlir        (this repo)
                        |
                    PTO MLIR (.pto)
                        |
                      ptoas                   (PTOAS repo)
                        |
              PTO-ISA C++ kernel (.cpp)
                     /       \
           bisheng -xcce     g++-14 -D__CPU_SIM
              |                    |
      Ascend device binary    x86 CPU-sim binary
              |                    |
         camodel sim          native execution
        (packaging only)      (numerical output)
```

## Directory Structure

```
triton-to-pto/
├── mlir_tool/
│   ├── CMakeLists.txt              # Top-level MLIR build
│   ├── lib/
│   │   └── Conversion/
│   │       └── TritonToPTO/
│   │           └── TritonToPTO.cpp # The conversion pass (main logic)
│   ├── tools/
│   │   └── triton-to-pto-mlir/
│   │       └── triton-to-pto-mlir.cpp  # CLI driver
│   └── test/
│       ├── e2e_all.sh              # Run all 4 kernels end-to-end
│       ├── e2e_triton_to_sim.sh    # Single-kernel Docker e2e
│       ├── e2e_run_ptoas_sim.sh    # Single-kernel with local converter
│       ├── cpu_sim/
│       │   └── cpu_sim_run.py      # CPU-sim (x86 numerical verification)
│       ├── golden/
│       │   └── golden_check.py     # NumPy golden reference comparison
│       ├── vec_add_e2e_triton.mlir
│       ├── reduce_sum_triton.mlir
│       ├── unary_exp_triton.mlir
│       └── softmax_triton.mlir
├── docker/
│   ├── Dockerfile                  # Builds triton-to-pto:py3.11
│   └── README.md
└── .cursor/rules/                  # These rules
```

## Key Source Files

| File | Language | Purpose |
|------|----------|---------|
| `TritonToPTO.cpp` | C++ | Triton -> PTO dialect conversion pass (~1700 lines) |
| `triton-to-pto-mlir.cpp` | C++ | CLI tool that loads MLIR, runs the pass, emits output |
| `cpu_sim_run.py` | Python | Builds and runs PTO kernel on x86 via __CPU_SIM |
| `golden_check.py` | Python | NumPy-based golden reference accuracy checks |
| `e2e_all.sh` | Bash | Orchestrates full e2e test pipeline for all kernels |

## Supported Kernels

| Kernel | Triton Pattern | PTO Mapping |
|--------|---------------|-------------|
| vec_add | `tt.load` + `tt.add` + `tt.store` | `pto.tload` + `pto.tadd` + `pto.tstore` |
| reduce_sum | `tt.load` + `tt.reduce(add)` + `tt.store` | `pto.tload` + `pto.treduce` + `pto.tstore` |
| unary_exp | `tt.load` + `math.exp` + `tt.store` | `pto.tload` + `pto.texp` + `pto.tstore` |
| softmax | `tt.load` + row-max + exp + sum + div + `tt.store` | `pto.tload` + `pto.trowmax` + `pto.texp` + ... |

## Dependencies

- **LLVM 19 + MLIR**: shared between PTOAS and this project (must be same build)
- **Triton**: provides `tt.*` dialect definitions
- **PTOAS**: provides `pto.*` dialect (linked as subproject or installed)
- **pto-isa**: PTO instruction set implementation (headers for device compilation, C++ fallbacks for CPU-sim)
- **CANN toolkit**: Ascend compiler (`bisheng`), runtime, simulator -- available inside Docker images

## Docker Images

| Image | Base | Contains |
|-------|------|----------|
| `ptoas-llvm:19` | Ubuntu | Pre-built LLVM 19 + MLIR |
| `ptoas:py3.11` | `ptoas-llvm:19` | PTOAS + CANN toolkit + pto-isa |
| `triton-to-pto:py3.11` | `ptoas:py3.11` | + Triton + triton-to-pto-mlir |
