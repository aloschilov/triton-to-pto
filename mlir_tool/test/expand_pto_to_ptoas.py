#!/usr/bin/env python3
"""
Format bridge: convert triton-to-pto PTO MLIR to PTOAS-acceptable .pto.

Reads MLIR from stdin or from a file; rewrites simplified tile_buf types
(!pto.tile_buf<RxCxdtype> and bare <RxCxdtype> in type positions) to the
full PTOAS form with loc=ub, dtype, rows, cols, v_row, v_col, blayout, etc.

Usage:
  expand_pto_to_ptoas.py [input.mlir] [-o output.pto]
  cat input.mlir | expand_pto_to_ptoas.py -o output.pto
"""

import argparse
import re
import sys


def ptoas_tile_buf(rows: int, cols: int, dtype: str) -> str:
    """Build PTOAS-style !pto.tile_buf<...> string."""
    return (
        "!pto.tile_buf<loc=ub, dtype={dtype}, rows={rows}, cols={cols}, "
        "v_row={rows}, v_col={cols}, blayout=row_major, slayout=none_box, "
        "fractal=512, pad=0>".format(rows=rows, cols=cols, dtype=dtype)
    )


def expand_tile_buf_types(text: str) -> str:
    # Expand !pto.tile_buf<rows>x<cols>x<dtype>
    text = re.sub(
        r"!pto\.tile_buf<(\d+)x(\d+)x(f32|f16|i32)>",
        lambda m: ptoas_tile_buf(int(m.group(1)), int(m.group(2)), m.group(3)),
        text,
    )
    # Expand bare <rows>x<cols>x<dtype> (our Tile/TileBuf short form in ops)
    text = re.sub(
        r"<(\d+)x(\d+)x(f32|f16|i32)>",
        lambda m: ptoas_tile_buf(int(m.group(1)), int(m.group(2)), m.group(3)),
        text,
    )
    return text


def main() -> int:
    parser = argparse.ArgumentParser(description="Expand triton-to-pto PTO MLIR to PTOAS .pto format")
    parser.add_argument("input", nargs="?", default="-", help="Input MLIR file (default: stdin)")
    parser.add_argument("-o", "--output", default="-", help="Output .pto file (default: stdout)")
    args = parser.parse_args()

    if args.input == "-":
        content = sys.stdin.read()
    else:
        with open(args.input, "r") as f:
            content = f.read()

    out = expand_tile_buf_types(content)

    if args.output == "-":
        sys.stdout.write(out)
    else:
        with open(args.output, "w") as f:
            f.write(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
