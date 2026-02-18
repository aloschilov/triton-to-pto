# TritonToPTO MLIR tests

- **Lit tests**: `lit.cfg.py` configures lit for `.mlir` tests. Run via `cmake --build build --target check-mlir-tool`.
- **E2E with PTOAS**: see repo root [README](../../README.md) section “E2E testing with PTOAS simulator”.

## E2E requirements

- **PTOAS_ROOT**: Path to the PTOAS repo (must contain `docker/run_sim_example.sh` and `test/npu_validation/scripts/generate_testcase.py`).
- **Docker image** `ptoas:py3.11` (build per PTOAS `docker/README.md`).
- **Converter** on `PATH` or set `TRITON_TO_PTO_MLIR`.

Run:

```bash
PTOAS_ROOT=/path/to/PTOAS mlir_tool/test/e2e_run_ptoas_sim.sh [optional_triton.mlir]
```

Default input: `vec_add_e2e_triton.mlir` (32×32 vec_add, 3 args, single block). The converter emits **native PTOAS** `.pto` (same dialect/assembly as PTOAS). The script writes it and runs the PTOAS simulator and compare inside Docker. Optionally set `USE_PTO_BRIDGE=1` to run `expand_pto_to_ptoas.py` on converter output (legacy bridge).
