# Adding a New Kernel

## Checklist

When adding support for a new Triton kernel through the full pipeline:

### 1. Conversion Pass (this repo)

- [ ] Add pattern recognition in `mlir_tool/lib/Conversion/TritonToPTO/TritonToPTO.cpp`
- [ ] Create a `.mlir` test file in `mlir_tool/test/<name>_triton.mlir` with `// RUN:` and `// CHECK:` lines

### 2. PTOAS Sample Directory (PTOAS-aloschilov)

- [ ] Create `test/samples/<SampleName>/` in `PTOAS-aloschilov` with a Python generator (e.g., `<name>.py > <name>.pto`)
- [ ] The directory name becomes the `PTOAS_SUBDIR` in `e2e_all.sh`

### 3. E2E Registration (this repo)

- [ ] Add entry to `KERNELS` in `mlir_tool/test/e2e_all.sh`:
  ```
  <input_mlir>:<PTOAS_subdir>:<golden_check_name>
  ```

### 4. CPU-sim Configuration (this repo, if non-trivial shapes)

- [ ] Add config entry to `KERNEL_CONFIGS` in `mlir_tool/test/cpu_sim/cpu_sim_run.py`
- [ ] Include **both** base name and `_0` variant:
  ```python
  KERNEL_CONFIGS = {
      "<name>_kernel": _CONFIG,
      "<name>_kernel_0": _CONFIG,
  }
  ```
- [ ] Define buffer shapes (`n_elems`, `init`, `seed`) and scalar values

### 5. Golden Check (this repo)

- [ ] Add `check_<name>(d, atol)` function in `mlir_tool/test/golden/golden_check.py`
- [ ] Implement the NumPy reference computation
- [ ] Register in the `KERNELS` dict at the bottom of the file:
  ```python
  KERNELS = {
      ...
      "<name>": check_<name>,
  }
  ```

### 6. Documentation (this repo)

- [ ] Update the kernel table in `mlir_tool/test/README.md`

## Notes

- The golden check compares `*_cpu.bin` (from CPU-sim) against NumPy expected values
- For simple element-wise ops (like vec_add, unary_exp), CPU-sim auto-detects shapes from `.bin` file sizes -- no `KERNEL_CONFIGS` entry needed
- For kernels with non-trivial reduction patterns or multi-dimensional shapes, an explicit `KERNEL_CONFIGS` entry ensures correct test data generation
- The `atol` default is `1e-4` but individual checks can override it
