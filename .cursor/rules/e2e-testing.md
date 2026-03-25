# E2E Testing Rules

## Running All E2E Tests

From the `triton-to-pto` repo root:

```bash
PTOAS_ROOT=/Users/aloschilov/articles-workspace/PTOAS mlir_tool/test/e2e_all.sh
```

Required Docker images: `triton-to-pto:py3.11` and `ptoas:py3.11`.

## Pipeline Steps (per kernel)

`e2e_all.sh` processes each kernel through four stages:

1. **Triton IR -> PTO**: `triton-to-pto-mlir` converts `.mlir` to `.pto` (runs in `triton-to-pto:py3.11`).
2. **PTOAS + sim**: `run_sim_example.sh` assembles PTO -> C++, compiles, and runs the Ascend camodel simulator (runs in `ptoas:py3.11`). This verifies the kernel compiles, but **does not produce usable output** (see camodel limitations).
3. **CPU-sim**: `cpu_sim/cpu_sim_run.py` compiles the kernel for x86 via `-D__CPU_SIM` and runs natively. This produces the actual numerical output (`*_cpu.bin` files).
4. **Golden check**: `golden/golden_check.py` compares CPU-sim output against NumPy reference values.

## Kernel Registry

The kernel list lives in `e2e_all.sh` line 25:

```
KERNELS="vec_add_e2e_triton.mlir:VectorAddition:vec_add
         reduce_sum_triton.mlir:Reduction:reduce_sum
         unary_exp_triton.mlir:UnaryExp:unary_exp
         softmax_triton.mlir:Softmax:softmax"
```

Format: `<input_mlir>:<PTOAS_subdir>:<golden_check_name>`

## Running a Single Kernel

```bash
PTOAS_ROOT=/path/to/PTOAS mlir_tool/test/e2e_run_ptoas_sim.sh mlir_tool/test/vec_add_e2e_triton.mlir
```

Or the all-in-Docker variant:

```bash
PTOAS_ROOT=/path/to/PTOAS mlir_tool/test/e2e_triton_to_sim.sh mlir_tool/test/softmax_triton.mlir
```

## Generated Directories

The `npu_validation/e2e_triton_to/` directories under each PTOAS sample are **regenerated every run** and must NOT be committed. They contain:
- `.bin` files (inputs/outputs)
- `*_cpu.bin` files (CPU-sim output)
- Generated C++ source (`main.cpp`, `launch.cpp`, `host_stub.cpp`, etc.)
- Build artifacts

## Accuracy Expectations

All four kernels achieve **exact match** (max_abs_err < 1e-6) via CPU-sim:

| Kernel | Max Abs Error | Mean Abs Error |
|--------|---------------|----------------|
| vec_add (4096 elems) | 0 | 0 |
| reduce_sum (256 -> 8 partial sums) | 0 | 0 |
| unary_exp (4096 elems) | ~1.5e-7 | ~6.4e-9 |
| softmax (16x256 row-wise) | ~1.19e-7 | ~1.73e-8 |

## Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `PTOAS_ROOT` | *required* | Path to PTOAS repo (upstream or fork) |
| `TRITON_TO_PTO_IMAGE` | `triton-to-pto:py3.11` | Docker image with converter |
| `PTOAS_DOCKER_IMAGE` | `ptoas:py3.11` | Docker image with PTOAS+CANN |
| `PTO_ISA_ROOT` | `/sources/pto-isa` (in Docker) | PTO-ISA include root |
| `CPU_SIM_CXX` | `g++-14` | C++23 compiler for CPU-sim |
