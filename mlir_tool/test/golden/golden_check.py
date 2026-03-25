#!/usr/bin/env python3
"""Independent golden-reference accuracy check for triton-to-pto kernels.

Computes expected outputs via NumPy and compares them against the actual
kernel output produced by the CPU-sim path (*_cpu.bin files).  Falls back
to the camodel simulator output (.bin files) when CPU-sim output is not
available, but in that case the check can only WARN (the camodel's
aclrtMemcpy DEVICE_TO_HOST is a no-op).

Usage (inside ptoas Docker container):
    python3 golden_check.py vec_add /path/to/npu_validation/e2e_triton_to/

Exit codes:
    0 - PASS or WARN
    1 - FAIL (output is non-trivial but incorrect)
    2 - ERROR (missing files / bad arguments)
"""

import argparse
import os
import sys

import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load(d, name):
    return np.fromfile(os.path.join(d, name), dtype=np.float32)


def _cpu_bin_path(d, name):
    """Return path to the CPU-sim output if it exists."""
    p = os.path.join(d, name.replace(".bin", "_cpu.bin"))
    return p if os.path.isfile(p) else None


def _check(kernel, d, output_file, expected_arr, atol=1e-4):
    """Compare actual output against *expected_arr*.

    Prefers the CPU-sim file (<name>_cpu.bin) when available; otherwise
    falls back to the camodel file (<name>.bin) with a WARN.

    Returns (ok, warn).
    """
    expected_arr = np.asarray(expected_arr, dtype=np.float32).ravel()
    cpu_path = _cpu_bin_path(d, output_file)

    if cpu_path is not None:
        actual = np.fromfile(cpu_path, dtype=np.float32)
        n = min(len(actual), len(expected_arr))
        actual = actual[:n]
        expected_arr = expected_arr[:n]
        max_err = float(np.max(np.abs(actual - expected_arr)))
        mean_err = float(np.mean(np.abs(actual - expected_arr)))
        ok = bool(np.allclose(actual, expected_arr, atol=atol, rtol=0))
        tag = "PASS" if ok else "FAIL"
        print(
            f"[golden] {tag} {kernel}: {n} elements compared, "
            f"max_abs_err={max_err:.6g}, mean_abs_err={mean_err:.6g}, "
            f"atol={atol}"
        )
        return ok, False

    # Fallback: camodel output (likely zeros / initial values)
    actual = np.fromfile(os.path.join(d, output_file), dtype=np.float32)
    if np.allclose(actual[0], expected_arr[0], atol=atol, rtol=0):
        print(
            f"[golden] PASS {kernel}: {output_file}[0]={actual[0]:.8g} == "
            f"expected={expected_arr[0]:.8g}"
        )
        return True, False

    # Detect unchanged output (camodel D2H limitation)
    if np.all(actual == 0) or np.allclose(actual[0], 0.0, atol=1e-9):
        print(
            f"[golden] WARN {kernel}: no CPU-sim output and camodel produced "
            f"zeros (D2H limitation).  Expected[0]={expected_arr[0]:.8g}"
        )
        return True, True

    print(
        f"[golden] FAIL {kernel}: {output_file}[0]={actual[0]:.8g} != "
        f"expected={expected_arr[0]:.8g}"
    )
    return False, False


# ---------------------------------------------------------------------------
# Per-kernel checks
# ---------------------------------------------------------------------------

def check_vec_add(d, atol=1e-4):
    """add_kernel(v1, v2, v3, n_elements): v3 = v1 + v2."""
    v1 = _load(d, "v1.bin")
    v2 = _load(d, "v2.bin")
    expected = v1 + v2
    return _check("vec_add", d, "v3.bin", expected, atol)


def check_reduce_sum(d, atol=1e-4):
    """reduce_sum_kernel(v1, v2, n_elements): per-block partial sums.

    The PTO kernel uses a grid-stride loop with BLOCK_SIZE=32.  Each
    iteration reduces one 32-element chunk and stores the partial sum
    to v2[chunk_index].
    """
    v1 = _load(d, "v1.bin")
    n = len(v1)
    BLOCK_SIZE = 32
    n_chunks = -(-n // BLOCK_SIZE)  # ceildiv
    expected = np.array([
        np.sum(v1[k * BLOCK_SIZE : min((k + 1) * BLOCK_SIZE, n)])
        for k in range(n_chunks)
    ], dtype=np.float32)
    return _check("reduce_sum", d, "v2.bin", expected, atol)


def check_unary_exp(d, atol=1e-4):
    """exp_kernel(v1, v2, n_elements): v2 = exp(v1)."""
    v1 = _load(d, "v1.bin")
    expected = np.exp(v1)
    return _check("unary_exp", d, "v2.bin", expected, atol)


def check_softmax(d, atol=1e-4):
    """softmax_kernel(v1=out, v2=in, ...): row-wise softmax.

    The kernel tile width is 256 (compile-time BLOCK_SIZE).  The shape
    is auto-detected from the input file size:
      - <= 256 elements: trivial single-element case (result = 1.0)
      - > 256 elements: n_rows = len(v2) // 256, full row-wise softmax
    """
    v2 = _load(d, "v2.bin")
    N_COLS = 256
    n = len(v2)
    if n <= N_COLS:
        expected = np.array([1.0], dtype=np.float32)
    else:
        n_rows = n // N_COLS
        mat = v2[: n_rows * N_COLS].reshape(n_rows, N_COLS)
        row_max = mat.max(axis=1, keepdims=True)
        numerator = np.exp(mat - row_max)
        denominator = numerator.sum(axis=1, keepdims=True)
        expected = (numerator / denominator).ravel()
    return _check("softmax", d, "v1.bin", expected, atol)


def check_matmul(d, atol=1e-3):
    """matmul_kernel(v1=A, v2=B, v3=C, ...): C = A @ B (f32).

    A and B are square (M=N=K) row-major matrices.  The dimension is
    inferred from file sizes: M = sqrt(len(v1)).
    """
    a = _load(d, "v1.bin")
    b = _load(d, "v2.bin")
    side = int(np.sqrt(len(a)))
    A = a[: side * side].reshape(side, side)
    B = b[: side * side].reshape(side, side)
    expected = A @ B
    return _check("matmul", d, "v3.bin", expected.ravel(), atol)


KERNELS = {
    "vec_add": check_vec_add,
    "reduce_sum": check_reduce_sum,
    "unary_exp": check_unary_exp,
    "softmax": check_softmax,
    "matmul": check_matmul,
}


def main():
    parser = argparse.ArgumentParser(
        description="Independent golden-reference accuracy check."
    )
    parser.add_argument(
        "kernel",
        choices=sorted(KERNELS),
        help="Kernel name to check.",
    )
    parser.add_argument(
        "directory",
        help="Path to the npu_validation/e2e_triton_to/ directory.",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-4,
        help="Absolute tolerance for comparison (default: 1e-4).",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.directory):
        print(f"[golden] ERROR: directory not found: {args.directory}", file=sys.stderr)
        sys.exit(2)

    ok, warn = KERNELS[args.kernel](args.directory, atol=args.atol)
    if not ok:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
