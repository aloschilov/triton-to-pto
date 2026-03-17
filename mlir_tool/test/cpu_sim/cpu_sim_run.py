#!/usr/bin/env python3
"""Build and run a PTO kernel on the host CPU via pto-isa's __CPU_SIM path.

The Ascend camodel simulator cannot transfer device output back to host
memory, so this script provides an alternative: compile the ptoas-generated
kernel with g++-14 / -D__CPU_SIM (which routes all PTO instructions through
pure-C++ host implementations) and run it natively on x86.

Usage (inside the ptoas Docker container):
    python3 cpu_sim_run.py <kernel.cpp> <npu_validation_dir>

Example:
    python3 cpu_sim_run.py \
        /workspace/test/samples/VectorAddition/e2e_triton_to_pto.cpp \
        /workspace/test/samples/VectorAddition/npu_validation/e2e_triton_to/

Outputs *_cpu.bin files in the npu_validation directory.
"""

import argparse
import os
import re
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

PTO_ISA_ROOT = os.environ.get("PTO_ISA_ROOT", "/sources/pto-isa")
CXX = os.environ.get("CPU_SIM_CXX", "g++-14")

# ---------------------------------------------------------------------------
# Per-kernel non-trivial test configurations
#
# When a kernel is listed here, the CPU-sim regenerates .bin files with the
# specified shapes and uses the given scalar values instead of the trivial
# defaults produced by PTOAS generate_testcase.py (which sets all scalars
# to 1).  Kernels not listed fall back to the single-scalar heuristic.
# ---------------------------------------------------------------------------

KERNEL_CONFIGS = {
    "reduce_sum_kernel": {
        "scalars": {"v3": 256},
        "buffers": {
            "v1": {"n_elems": 256, "init": "random", "seed": 42},
            "v2": {"n_elems": 8, "init": "zeros"},
        },
    },
    "softmax_kernel": {
        "scalars": {"v3": 256, "v4": 256, "v5": 16, "v6": 256},
        "buffers": {
            "v2": {"n_elems": 4096, "init": "random", "seed": 42},
            "v1": {"n_elems": 4096, "init": "zeros"},
        },
    },
}


def _regenerate_bins(nv_dir: Path, buf_specs: dict) -> None:
    """Write fresh .bin files according to *buf_specs* before kernel execution."""
    for name, spec in buf_specs.items():
        n = spec["n_elems"]
        if spec["init"] == "random":
            rng = np.random.RandomState(spec.get("seed", 42))
            arr = rng.random(size=(n,)).astype(np.float32)
        else:
            arr = np.zeros(n, dtype=np.float32)
        path = nv_dir / f"{name}.bin"
        arr.tofile(str(path))
        print(f"[cpu-sim] regenerated {name}.bin ({n} floats, {spec['init']})")


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def _parse_kernel_signature(kernel_cpp: str) -> Tuple[str, list]:
    """Return (func_name, params) from the ptoas-generated kernel .cpp.

    Each param is a dict with keys: name, ctype, kind ('ptr' or 'scalar').
    """
    sig_match = re.search(
        r"void\s+(\w+)\s*\(([^)]+)\)", kernel_cpp
    )
    if not sig_match:
        raise RuntimeError("Cannot find kernel function signature")
    func_name = sig_match.group(1)
    raw_params = sig_match.group(2)

    params: list = []
    for token in raw_params.split(","):
        token = token.strip()
        token = token.replace("__gm__", "").strip()
        parts = token.rsplit(None, 1)
        if len(parts) != 2:
            continue
        ctype, name = parts[0].strip(), parts[1].strip()
        name = name.lstrip("*").strip()
        is_ptr = "*" in ctype or "*" in parts[1]
        params.append({
            "name": name,
            "ctype": "float" if is_ptr else ctype,
            "kind": "ptr" if is_ptr else "scalar",
        })
    return func_name, params


def _parse_block_count(launch_cpp: str) -> int:
    m = re.search(r"<<<\s*(\d+)\s*,", launch_cpp)
    return int(m.group(1)) if m else 4


def _parse_scalar_defaults(main_cpp: str) -> dict:
    """Extract scalar default values from the generated main.cpp."""
    defaults: dict = {}
    for m in re.finditer(r"int32_t\s+(v\d+)\s*=\s*(\d+)\s*;", main_cpp):
        defaults[m.group(1)] = int(m.group(2))
    return defaults


def _bin_file_sizes(nv_dir: Path) -> dict:
    """Return {name: byte_size} for all .bin files."""
    sizes: dict = {}
    for f in nv_dir.glob("*.bin"):
        if f.name.startswith("golden_") or f.name.startswith("numpy_golden_"):
            continue
        sizes[f.stem] = f.stat().st_size
    return sizes


# ---------------------------------------------------------------------------
# Code generation
# ---------------------------------------------------------------------------

def _generate_cpu_sim_main(
    func_name: str,
    params: list,
    block_count: int,
    scalar_values: dict,
    bin_sizes: dict,
    outputs: List[str],
) -> str:
    """Generate the cpu_sim_main.cpp source."""

    # Preamble
    lines = [textwrap.dedent("""\
        #define __CPU_SIM 1
        #include <cstdio>
        #include <cstdlib>
        #include <cstring>
        #include <cstdint>
        #include <fstream>
        #include <sys/stat.h>

        static int64_t _cpu_blk_idx = 0;
        static int64_t _cpu_blk_num = 1;
        inline int64_t get_block_idx() { return _cpu_blk_idx; }
        inline int64_t get_block_num() { return _cpu_blk_num; }

        #include <pto/pto-inst.hpp>
        using namespace pto;

        #include "kernel_cpu.cpp"

        static size_t fsize(const char *p) {
            struct stat st;
            return (stat(p, &st) == 0) ? (size_t)st.st_size : 0;
        }
        static void *read_bin(const char *p) {
            size_t sz = fsize(p);
            if (!sz) { std::fprintf(stderr, "Cannot read %s\\n", p); return nullptr; }
            void *buf = std::malloc(sz);
            std::ifstream f(p, std::ios::binary);
            f.read(static_cast<char*>(buf), sz);
            return buf;
        }
        static bool write_bin(const char *p, const void *buf, size_t sz) {
            std::ofstream f(p, std::ios::binary);
            if (!f) return false;
            f.write(static_cast<const char*>(buf), sz);
            return f.good();
        }

        int main() {
    """)]

    ptr_params = [p for p in params if p["kind"] == "ptr"]
    scalar_params = [p for p in params if p["kind"] == "scalar"]

    # Read ALL pointer buffers from their .bin files.  The kernel decides
    # which ones to read and which to overwrite; we don't rely on
    # outputs.txt because generate_testcase sometimes misidentifies the
    # output (e.g. softmax).
    for p in ptr_params:
        name = p["name"]
        lines.append(f"    size_t sz_{name} = fsize(\"./{name}.bin\");\n")
        lines.append(f"    float *{name} = (float*)read_bin(\"./{name}.bin\");\n")

    # Declare scalars
    for p in scalar_params:
        val = scalar_values.get(p["name"], 1)
        lines.append(f"    int32_t {p['name']} = {val};\n")

    lines.append(f"\n    _cpu_blk_num = {block_count};\n")
    lines.append(f"    for (int64_t _b = 0; _b < {block_count}; _b++) {{\n")
    lines.append("        _cpu_blk_idx = _b;\n")

    # Build call args
    call_args = ", ".join(p["name"] for p in params)
    lines.append(f"        {func_name}({call_args});\n")
    lines.append("    }\n\n")

    # Write ALL pointer buffers as *_cpu.bin (the golden check knows
    # which one is the actual output for each kernel).
    for p in ptr_params:
        name = p["name"]
        lines.append(
            f"    write_bin(\"./{name}_cpu.bin\", {name}, sz_{name});\n"
        )
        lines.append(
            f'    std::printf("[CPU-SIM] {name}_cpu.bin written (%zu bytes)\\n", sz_{name});\n'
        )

    # Cleanup
    for p in ptr_params:
        lines.append(f"    std::free({p['name']});\n")
    lines.append("    return 0;\n}\n")

    return "".join(lines)


# ---------------------------------------------------------------------------
# Build & run
# ---------------------------------------------------------------------------

def _ensure_compiler() -> str:
    """Return the path to a working C++23 compiler, installing g++-14 if needed."""
    if _compiler_ok(CXX):
        return CXX

    print("[cpu-sim] g++-14 not found, installing...")
    cmds = [
        "apt-get update -qq",
        "apt-get install -y -qq software-properties-common",
        "add-apt-repository -y ppa:ubuntu-toolchain-r/test",
        "apt-get update -qq",
        "apt-get install -y -qq g++-14",
    ]
    for cmd in cmds:
        subprocess.run(cmd, shell=True, stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL, check=True)

    if not _compiler_ok(CXX):
        raise RuntimeError(f"Failed to install {CXX}")
    return CXX


def _compiler_ok(cxx: str) -> bool:
    try:
        r = subprocess.run([cxx, "--version"], capture_output=True)
        return r.returncode == 0
    except FileNotFoundError:
        return False


def build_and_run(
    kernel_cpp_path: Path,
    nv_dir: Path,
) -> int:
    """Main entry point: generate, compile, and run the CPU-sim binary."""

    kernel_src = kernel_cpp_path.read_text()
    func_name, params = _parse_kernel_signature(kernel_src)
    print(f"[cpu-sim] kernel: {func_name}, params: {[p['name'] for p in params]}")

    # Read launch.cpp for block count
    launch_path = nv_dir / "launch.cpp"
    block_count = 4
    if launch_path.is_file():
        block_count = _parse_block_count(launch_path.read_text())
    print(f"[cpu-sim] block_count: {block_count}")

    # Read outputs.txt
    outputs_path = nv_dir / "outputs.txt"
    outputs: List[str] = []
    if outputs_path.is_file():
        outputs = [
            line.strip()
            for line in outputs_path.read_text().splitlines()
            if line.strip()
        ]
    if not outputs:
        ptr_names = [p["name"] for p in params if p["kind"] == "ptr"]
        outputs = [ptr_names[-1]] if ptr_names else []
    print(f"[cpu-sim] outputs: {outputs}")

    ptr_params = [p for p in params if p["kind"] == "ptr"]
    scalar_params = [p for p in params if p["kind"] == "scalar"]

    # Use KERNEL_CONFIGS for non-trivial test shapes when available.
    config = KERNEL_CONFIGS.get(func_name)
    if config is not None:
        scalar_values = dict(config["scalars"])
        _regenerate_bins(nv_dir, config["buffers"])
    else:
        # Fallback: read scalar defaults from main.cpp.  When there is
        # exactly one scalar param it is almost always n_elements; override
        # it with the element count of the first pointer buffer.
        main_path = nv_dir / "main.cpp"
        scalar_defaults = {}
        if main_path.is_file():
            scalar_defaults = _parse_scalar_defaults(main_path.read_text())

        scalar_values = {}
        if len(scalar_params) == 1:
            bin_sizes_local = _bin_file_sizes(nv_dir)
            first_ptr_elems = bin_sizes_local.get(
                ptr_params[0]["name"], 16384
            ) // 4 if ptr_params else 1
            scalar_values[scalar_params[0]["name"]] = first_ptr_elems
        else:
            for p in scalar_params:
                scalar_values[p["name"]] = scalar_defaults.get(p["name"], 1)

    bin_sizes = _bin_file_sizes(nv_dir)
    print(f"[cpu-sim] scalar_values: {scalar_values}")

    # Prepare kernel source: strip __CCE_AICORE__ guard
    stripped = re.sub(
        r"#if\s+defined\(__CCE_AICORE__\)",
        "#if 1",
        kernel_src,
    )
    kernel_cpu_path = nv_dir / "kernel_cpu.cpp"
    kernel_cpu_path.write_text(stripped)

    # Generate cpu_sim_main.cpp
    main_src = _generate_cpu_sim_main(
        func_name, params, block_count, scalar_values, bin_sizes, outputs,
    )
    sim_main_path = nv_dir / "cpu_sim_main.cpp"
    sim_main_path.write_text(main_src)

    # Ensure compiler
    cxx = _ensure_compiler()

    # Compile
    binary = nv_dir / "cpu_sim_test"
    compile_cmd = [
        cxx, "-std=c++23", "-D__CPU_SIM",
        f"-I{PTO_ISA_ROOT}/include",
        f"-I{nv_dir}",
        "-Wno-ignored-attributes",
        "-O2", "-lpthread",
        "-o", str(binary),
        str(sim_main_path),
    ]
    print(f"[cpu-sim] compiling...")
    result = subprocess.run(compile_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[cpu-sim] COMPILE ERROR:\n{result.stderr}", file=sys.stderr)
        return 1

    # Run
    print(f"[cpu-sim] running...")
    result = subprocess.run(
        [str(binary)], cwd=str(nv_dir), capture_output=True, text=True,
    )
    print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr)
    if result.returncode != 0:
        print(f"[cpu-sim] RUNTIME ERROR (exit {result.returncode})", file=sys.stderr)
        return 1

    # Verify at least one *_cpu.bin was produced
    ok = any(
        (nv_dir / f"{p['name']}_cpu.bin").is_file()
        for p in params if p["kind"] == "ptr"
    )
    if not ok:
        print("[cpu-sim] ERROR: no *_cpu.bin files produced", file=sys.stderr)
    return 0 if ok else 1


def main():
    parser = argparse.ArgumentParser(description="CPU-sim for PTO kernels")
    parser.add_argument("kernel_cpp", help="Path to ptoas-generated kernel .cpp")
    parser.add_argument("nv_dir", help="Path to npu_validation/e2e_triton_to/")
    args = parser.parse_args()

    rc = build_and_run(Path(args.kernel_cpp), Path(args.nv_dir))
    sys.exit(rc)


if __name__ == "__main__":
    main()
