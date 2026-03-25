# Kernel Naming Conventions

## Device Kernel Names

Ascend's `RegisterAscendBinary` API **requires** kernel function names to:

1. Use **`extern "C"`** linkage (no C++ name mangling).
2. End with a **`_<digit>` suffix** (e.g., `add_kernel_0`, `softmax_kernel_0`).

If either condition is violated, `RegisterAscendBinary` fails with error code `107000`.

## Naming Flow

```
Triton kernel name      ->  PTO kernel name          ->  Device symbol
add_kernel              ->  add_kernel               ->  extern "C" void add_kernel_0(...)
softmax_kernel          ->  softmax_kernel           ->  extern "C" void softmax_kernel_0(...)
```

The `_0` suffix and `extern "C"` are applied in `generate_testcase.py` during kernel source transformation. The host-side launch wrapper is named `aclrtlaunch_<kernel_name>_0`.

## CPU-sim Config Keys

`cpu_sim_run.py`'s `KERNEL_CONFIGS` dictionary must include **both** the base name and the `_0` variant so CPU-sim works regardless of which compilation path produced the kernel:

```python
KERNEL_CONFIGS = {
    "reduce_sum_kernel": _REDUCE_SUM_CONFIG,
    "reduce_sum_kernel_0": _REDUCE_SUM_CONFIG,
    "softmax_kernel": _SOFTMAX_CONFIG,
    "softmax_kernel_0": _SOFTMAX_CONFIG,
}
```

When adding a new kernel config, always add both variants.

## Launch Wrapper Convention

The host stub must define:
- `extern "C" void aclrtlaunch_<kernel_name>_0(...)` -- called by the CANN runtime
- `void <original_launch_name>(...)` -- C++ wrapper matching the `main.cpp` interface

## Kernel Argument Structure

When compiled with `-mllvm -cce-aicore-record-overflow=true`, the kernel binary expects an extra 8-byte `__ascendc_overflow` pointer as the **last** argument. The host stub must allocate 256 bytes of device memory for it and append it to the argument buffer.

Argument buffer layout:
```
[ptr_arg_1] [ptr_arg_2] ... [scalar_arg_1] ... [__ascendc_overflow]
```

Each pointer is 8 bytes (device address). Each `int32_t` scalar is 8 bytes (zero-extended).
