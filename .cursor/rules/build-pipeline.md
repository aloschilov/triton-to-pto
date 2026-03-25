# Kernel Build Pipeline (RegisterAscendBinary Approach)

## Overview

PTOAS-generated kernels are packaged for the Ascend runtime using `RegisterAscendBinary` instead of the default `--cce-fatobj-link` approach (which uses a no-op stub on camodel).

The build pipeline is implemented in `generate_testcase.py` (in `PTOAS-aloschilov`).

## 6-Step CMake Build Pipeline

For each testcase, the CMakeLists.txt implements:

```
1. bisheng -xcce  →  fat object (.o)
2. objcopy --dump-section __aicore_rel_binary  →  device.bin
3. ld.lld -r -b binary -m <emulation>  →  device.o (ELF wrapper)
4. g++ -c host_stub.cpp  →  host_stub.o
5. ascendc_pack_kernel.sh device.o host_stub.o  →  packed host_stub.o
6. g++ -shared host_stub.o + libascendc_runtime.a  →  lib<testcase>_kernel.so
```

### Step Details

1. **Compile kernel**: `bisheng -xcce` compiles the PTO kernel to a fat object containing both host and device code. The `-xcce` flag is required (not `--cce-aicore-only`).

2. **Extract device binary**: `objcopy --dump-section=__aicore_rel_binary=device.bin` pulls out the raw device binary.

3. **ELF wrapper**: `ld.lld -r -b binary -m <emulation> -o device.o device.bin` creates a relocatable ELF object from the raw binary. The emulation is `elf_x86_64` on x86_64, `aarch64linux` on aarch64.

4. **Host stub**: `host_stub.cpp` contains:
   - `RegisterAscendBinary` call in a `__attribute__((constructor))` function
   - `aclrtlaunch_<kernel>_0()` launch function
   - `.ascend.kernel.<soc>.kernels` ELF section with appropriately sized `aiv_buf`
   - `__ascendc_overflow` allocation (256 bytes)

5. **Pack**: `ascendc_pack_kernel.sh` embeds `device.o` into `host_stub.o`'s `.ascend.kernel.*` section. This modifies `host_stub.o` in-place.

6. **Link**: Final shared library linked with `libascendc_runtime.a`.

## Key Tools

| Tool | Path (in Docker) | Purpose |
|------|-------------------|---------|
| `bisheng` | `/usr/local/Ascend/ascend-toolkit/latest/compiler/ccec_compiler/bin/bisheng` | Ascend Clang compiler |
| `objcopy` | System binutils | Extract ELF sections |
| `ld.lld` | Found via `find_program` | Create ELF wrapper |
| `ascendc_pack_kernel.sh` | `${ASCEND_HOME_PATH}/compiler/tikcpp/ascendc_pack_kernel.sh` | Pack device binary into host stub |

## Include Paths

PTO-ISA kernels need headers from `/sources/pto-isa/include` (in Docker), not from the system CANN headers. The CMake template adds:
- `-I${PTO_ISA_ROOT}/include`
- `-I${PTO_ISA_ROOT}/tests/common`
- `-I${ASCEND_HOME_PATH}/compiler/tikcpp/tikcfw`

## Kernel Source Transformation

`generate_testcase.py` transforms the ptoas-generated kernel before compilation:
1. Wraps the kernel body in `#if defined(__CCE_AICORE__)` guards
2. Adds `extern "C"` linkage
3. Appends `_0` suffix to the function name
4. Prepends the `INCLUDE_REPLACEMENT` compatibility block (FP8 type stubs, `MrgSortExecutedNumList` fallback)

## Files Modified (in PTOAS-aloschilov)

- `test/npu_validation/scripts/generate_testcase.py` -- main generator
- `test/npu_validation/templates/run_sh_template.sh` -- runtime LD_LIBRARY_PATH fix
