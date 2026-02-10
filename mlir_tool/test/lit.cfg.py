import os

import lit.formats

# Basic lit configuration for the Triton→PTO MLIR tool tests.

config.name = "TritonToPTO-MLIR-Tests"
config.test_format = lit.formats.ShTest()
config.suffixes = [".mlir"]

# Treat this directory as the test source root.
config.test_source_root = os.path.dirname(__file__)

