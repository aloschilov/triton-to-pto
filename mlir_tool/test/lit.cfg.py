import os

import lit.formats

# Basic lit configuration for the Triton→PTO MLIR tool tests.

config.name = "TritonToPTO-MLIR-Tests"
config.test_format = lit.formats.ShTest()
config.suffixes = [".mlir"]

# Treat this directory as the test source root.
config.test_source_root = os.path.dirname(__file__)

# Use explicit tool paths from the environment when set (by CMake), so tests
# do not rely on PATH. Fall back to bare names for ad-hoc lit runs.
triton_to_pto_mlir = os.environ.get("TRITON_TO_PTO_MLIR", "triton-to-pto-mlir")
filecheck = os.environ.get("LIT_FILECHECK", "FileCheck")
config.substitutions.append(("%triton_to_pto_mlir", triton_to_pto_mlir))
config.substitutions.append(("%filecheck", filecheck))

