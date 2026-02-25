# TritonToPTO MLIR tests

- **Lit tests**: `lit.cfg.py` configures lit for `.mlir` tests. Run via `cmake --build build --target check-mlir-tool`.
- **E2E**: two flows below (PTOAS-only and full Triton → execution).

## E2E flows

### Flow 1: PTOAS-only (no converter)

Runs entirely inside `ptoas:py3.11` with the PTOAS repo mounted. Generate a `.pto` from a PTOAS sample, then run the simulator via `run_sim_example.sh`.

From the **PTOAS** repo root:

```bash
docker run --rm -v "$(pwd)":/workspace -w /workspace ptoas:py3.11 bash -c "cd test/samples/Abs && python3 abs.py > abs.pto && bash /workspace/docker/run_sim_example.sh /workspace/test/samples/Abs/abs.pto"
```

Requires: Docker image `ptoas:py3.11` (build from PTOAS repo per its `docker/README.md`).

### Flow 2: Full Triton → execution

[Triton MLIR] → triton-to-pto-mlir → `.pto` → PTOAS sim (same as Flow 1 inside Docker).

From the **triton-to-pto** repo root:

```bash
export PTOAS_ROOT=/path/to/PTOAS
# Optional: export TRITON_TO_PTO_MLIR=/path/to/triton-to-pto-mlir
mlir_tool/test/e2e_run_ptoas_sim.sh [optional_triton.mlir]
```

- **PTOAS_ROOT**: PTOAS repo (must contain `docker/run_sim_example.sh` and `test/npu_validation/scripts/generate_testcase.py`).
- **Converter**: on `PATH` or set `TRITON_TO_PTO_MLIR` (e.g. from `triton-to-pto:py3.11` or local build).
- **Image**: `ptoas:py3.11`.

Default input: `mlir_tool/test/vec_add_e2e_triton.mlir`. The script writes the generated `.pto` and runs `docker run ... ptoas:py3.11 bash docker/run_sim_example.sh <path>.pto`. Optionally set `USE_PTO_BRIDGE=1` for the legacy Python bridge on converter output.

**All-in-Docker (no converter on host):** run converter and sim entirely in containers:

```bash
PTOAS_ROOT=/path/to/PTOAS mlir_tool/test/e2e_triton_to_sim.sh [optional_triton.mlir]
```

Requires images `triton-to-pto:py3.11` and `ptoas:py3.11`.

## Docker build order (for triton-to-pto image)

1. Build **ptoas-llvm:19** once (from PTOAS repo): `docker build -f docker/Dockerfile.llvm -t ptoas-llvm:19 docker/empty`
2. Build **ptoas:py3.11** (from PTOAS repo)
3. Build **triton-to-pto:py3.11** (from this repo): `docker build -f docker/Dockerfile -t triton-to-pto:py3.11 .`

See [docker/README.md](../../docker/README.md) for details. Using the pre-built LLVM base avoids rebuilding LLVM when iterating on triton-to-pto.
