# TritonToPTO MLIR tests

- **Lit tests**: `lit.cfg.py` configures lit for `.mlir` tests. Run via `cmake --build build --target check-mlir-tool`.
- **E2E**: three flows below.

## Supported kernels

| Kernel | Test file | PTOAS subdirectory |
|---|---|---|
| Vector add | `vec_add_e2e_triton.mlir` | `VectorAddition` |
| Reduction (sum) | `reduce_sum_triton.mlir` | `Reduction` |
| Unary (exp) | `unary_exp_triton.mlir` | `UnaryExp` |
| Softmax (fused) | `softmax_triton.mlir` | `Softmax` |
| Matrix multiply | `matmul_triton.mlir` | `MatMulSimple` |

## E2E flows

### Flow 1: All kernels (recommended)

Runs all 5 holistic-rewrite kernels through the full pipeline inside Docker:

```bash
PTOAS_ROOT=/path/to/PTOAS mlir_tool/test/e2e_all.sh
```

For each kernel the script:

1. Converts Triton IR to `.pto` (triton-to-pto-mlir in Docker).
2. Runs ptoas assembly, C++ compilation, and Ascend camodel simulator (PTOAS Docker) -- verifies the kernel compiles and the simulator loads/runs it.
2b. Runs **CPU-sim** (`cpu_sim/cpu_sim_run.py`) -- compiles the kernel for x86 via pto-isa's `__CPU_SIM` path and executes it natively to produce actual numerical output.
3. Runs an **independent golden accuracy check** (`golden/golden_check.py`) -- compares the CPU-sim output against NumPy-computed expected values.

Reports per-kernel pass/fail and overall status.

Requires images `triton-to-pto:py3.11` and `ptoas:py3.11`.

### Flow 2: PTOAS-only (no converter)

Runs entirely inside `ptoas:py3.11` with the PTOAS repo mounted. Generate a `.pto` from a PTOAS sample, then run the simulator via `run_sim_example.sh`.

From the **PTOAS** repo root:

```bash
docker run --rm -v "$(pwd)":/workspace -w /workspace ptoas:py3.11 bash -c "cd test/samples/Abs && python3 abs.py > abs.pto && bash /workspace/docker/run_sim_example.sh /workspace/test/samples/Abs/abs.pto"
```

Requires: Docker image `ptoas:py3.11` (build from PTOAS repo per its `docker/README.md`).

### Flow 3: Single kernel (full Triton → execution)

[Triton MLIR] → triton-to-pto-mlir → `.pto` → PTOAS sim (same as Flow 2 inside Docker).

From the **triton-to-pto** repo root:

```bash
export PTOAS_ROOT=/path/to/PTOAS
# Optional: export TRITON_TO_PTO_MLIR=/path/to/triton-to-pto-mlir
mlir_tool/test/e2e_run_ptoas_sim.sh [optional_triton.mlir]
```

- **PTOAS_ROOT**: PTOAS repo (must contain `docker/run_sim_example.sh` and `test/npu_validation/scripts/generate_testcase.py`).
- **Converter**: on `PATH` or set `TRITON_TO_PTO_MLIR` (e.g. from `triton-to-pto:py3.11` or local build).
- **Image**: `ptoas:py3.11`.

Default input: `mlir_tool/test/vec_add_e2e_triton.mlir`. The script auto-routes the generated `.pto` to the correct PTOAS sample subdirectory based on the kernel name. Optionally set `USE_PTO_BRIDGE=1` for the legacy Python bridge on converter output.

**All-in-Docker (no converter on host):** run converter and sim entirely in containers:

```bash
PTOAS_ROOT=/path/to/PTOAS mlir_tool/test/e2e_triton_to_sim.sh [optional_triton.mlir]
```

Requires images `triton-to-pto:py3.11` and `ptoas:py3.11`.

## CPU-sim (numerical accuracy)

The Ascend camodel simulator does not transfer device output back to host memory (`aclrtMemcpy DEVICE_TO_HOST` is a no-op), so a CPU-sim step (`cpu_sim/cpu_sim_run.py`) compiles the ptoas-generated kernel for x86 using pto-isa's built-in CPU emulation (`-D__CPU_SIM`).

How it works:

1. Parses the kernel signature and launch configuration from the ptoas output.
2. Generates a `cpu_sim_main.cpp` that reads `.bin` inputs, calls the kernel directly (with `get_block_idx`/`get_block_num` stubs), and writes `*_cpu.bin` output files.
3. Compiles with `g++-14 -std=c++23` (installed on the fly if absent).
4. Runs natively on the host CPU. All PTO instructions (TLoad, TStore, TAdd, TExp, TRowMax, etc.) execute through pto-isa's C++ host implementations.

Requirements: `g++-14` (C++23), installed automatically from `ppa:ubuntu-toolchain-r/test` the first time the CPU-sim step runs inside the Docker container.

## Golden reference check

`golden/golden_check.py` compares the CPU-sim kernel output (`*_cpu.bin`) against NumPy-computed expected values. When CPU-sim output is available the check compares the **full output array** and reports max/mean absolute error.

| Kernel | Input Shape | Output Shape | Formula | Elements |
|---|---|---|---|---|
| vec_add | (4096,) + (4096,) | (4096,) | `v1 + v2` | 4096 |
| reduce_sum | (256,) | (8,) | `v2[k] = sum(v1[k*32:(k+1)*32])` | 8 |
| unary_exp | (4096,) | (4096,) | `exp(v1)` | 4096 |
| softmax | (16, 256) | (16, 256) | row-wise `exp(x-max)/sum(exp(x-max))` | 4096 |
| matmul | (64, 64) x (64, 64) | (64, 64) | `A @ B` | 4096 |

The CPU-sim regenerates `.bin` files with non-trivial shapes for reduce_sum and softmax (see `KERNEL_CONFIGS` in `cpu_sim/cpu_sim_run.py`). The golden check auto-detects the configuration from input file sizes.

To add a new kernel, add a `check_<name>` function to `golden/golden_check.py` and a new entry to the `KERNELS` string in `e2e_all.sh`.

Both the CPU-sim and golden check run inside the ptoas Docker container and require no changes to PTOAS.

## Docker build order (for triton-to-pto image)

1. Build **ptoas-llvm:19** once (from PTOAS repo): `docker build -f docker/Dockerfile.llvm -t ptoas-llvm:19 docker/empty`
2. Build **ptoas:py3.11** (from PTOAS repo)
3. Build **triton-to-pto:py3.11** (from this repo): `docker build -f docker/Dockerfile -t triton-to-pto:py3.11 .`

See [docker/README.md](../../docker/README.md) for details. Using the pre-built LLVM base avoids rebuilding LLVM when iterating on triton-to-pto.
