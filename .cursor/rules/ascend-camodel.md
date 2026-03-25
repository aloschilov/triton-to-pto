# Ascend Camodel Simulator Limitations

## Key Finding

The Ascend camodel simulator **cannot execute PTO-ISA kernels**. This is a fundamental limitation, not a packaging or API usage bug.

## What Works

- `aclInit` / `aclFinalize` -- initialization succeeds
- `aclrtMemcpy HOST_TO_DEVICE` -- copies data to simulated device memory
- `aclrtMemcpy DEVICE_TO_HOST` -- **works correctly** (returns whatever is in device memory)
- `RegisterAscendBinary` -- succeeds (returns 0) when kernel packaging is correct
- `LaunchAscendKernel` -- succeeds (returns 0) but enqueues work the simulator cannot execute
- **Official AscendC kernels** (compiled with `ascendc_library`) execute correctly on the `Ascend910B1` PEM simulator

## What Does NOT Work

- **PTO-ISA kernel execution**: crashes with `std::out_of_range: _Map_base::at` during `aclrtSynchronizeStream`
- **Ascend310P3 (CA MODEL)**: reports `0x19a unknown register encode` for PTO-ISA instructions
- **Ascend910B1 (PEM MODEL)**: successfully initializes but crashes during PTO-ISA instruction dispatch
- The compiled device binary from `bisheng -xcce` targets `dav-c310-vec`; neither simulator variant can decode these instructions

## Consequences for Testing

Because the simulator cannot run PTO-ISA kernels:

1. **Numerical accuracy** is verified exclusively through the **CPU-sim** path (`-D__CPU_SIM` with `g++-14`).
2. The camodel step in `e2e_all.sh` only verifies that the kernel **compiles and packages** correctly.
3. Output `.bin` files from the camodel are **not meaningful** (contain initial values, not computed results).
4. Real numerical verification on hardware requires an actual Ascend NPU.

## Simulator Models

| SoC | Simulator Type | PTO-ISA Support |
|-----|---------------|-----------------|
| Ascend310P3 | CA MODEL | No -- unknown register encode errors |
| Ascend910B1 | PEM MODEL | No -- `_Map_base::at` crash |

## LD_LIBRARY_PATH for Simulator

The simulator requires specific library paths. The `run_sh_template.sh` handles this by prepending:
- `${ROOT_DIR}/${BUILD_DIR}` (for `lib*_kernel.so`)
- Simulator-specific lib directories (auto-detected from `ASCEND_HOME_PATH`)

## Do NOT

- Assume camodel output values are correct
- Spend time debugging `RegisterAscendBinary failed: 107000` when it occurs during `aclrtSynchronizeStream` (this is the instruction incompatibility, not a packaging error)
- Attempt to use `--cce-aicore-only` for PTO-ISA kernels (incompatible instruction set)
