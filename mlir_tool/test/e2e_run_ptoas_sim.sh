#!/usr/bin/env bash
# E2E: Triton → triton-to-pto-mlir → .pto → PTOAS (ptoas + sim + compare).
#
# The converter emits PTOAS-compatible .pto (TileBufType with loc=, dtype=, rows=, etc.).
# Optional: set USE_PTO_BRIDGE=1 to run the legacy Python bridge on converter output.
#
# Usage:
#   PTOAS_ROOT=/path/to/PTOAS [TRITON_TO_PTO_MLIR=/path/to/triton-to-pto-mlir] \
#     mlir_tool/test/e2e_run_ptoas_sim.sh [triton_input.mlir]
#
# Default Triton input: mlir_tool/test/vec_add_e2e_triton.mlir (32x32 vec_add, 3 args).
# Requires: PTOAS repo at PTOAS_ROOT, docker image ptoas:py3.11, run_sim_example.sh in PTOAS.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
TRITON_INPUT="${1:-${SCRIPT_DIR}/vec_add_e2e_triton.mlir}"
CONVERTER="${TRITON_TO_PTO_MLIR:-triton-to-pto-mlir}"
BRIDGE="${SCRIPT_DIR}/expand_pto_to_ptoas.py"
DOCKER_IMAGE="${PTOAS_DOCKER_IMAGE:-ptoas:py3.11}"
USE_BRIDGE="${USE_PTO_BRIDGE:-0}"

if [[ -z "${PTOAS_ROOT:-}" ]]; then
  echo "ERROR: PTOAS_ROOT not set. Point it at the PTOAS repo root (for docker mount and run_sim_example.sh)." >&2
  exit 1
fi
PTOAS_ROOT="$(cd "${PTOAS_ROOT}" && pwd)"
if [[ ! -f "${PTOAS_ROOT}/docker/run_sim_example.sh" ]]; then
  echo "ERROR: PTOAS_ROOT/docker/run_sim_example.sh not found. Is PTOAS_ROOT correct?" >&2
  exit 1
fi
if [[ ! -f "${TRITON_INPUT}" ]]; then
  echo "ERROR: Triton input not found: ${TRITON_INPUT}" >&2
  exit 1
fi
if ! command -v "${CONVERTER}" >/dev/null 2>&1; then
  echo "ERROR: Converter not found: ${CONVERTER}. Set TRITON_TO_PTO_MLIR or add it to PATH." >&2
  exit 1
fi
if [[ "${USE_BRIDGE}" == "1" ]] && [[ ! -f "${BRIDGE}" ]]; then
  echo "ERROR: Bridge script not found: ${BRIDGE} (USE_PTO_BRIDGE=1)." >&2
  exit 1
fi

# Output .pto: converter writes PTOAS-compatible MLIR; optionally run legacy bridge.
GENERATED_PTO="${SCRIPT_DIR}/e2e_generated.pto"

echo "[e2e] Converting: ${TRITON_INPUT}"
if [[ "${USE_BRIDGE}" == "1" ]]; then
  "${CONVERTER}" "${TRITON_INPUT}" -o - 2>/dev/null | python3 "${BRIDGE}" -o "${GENERATED_PTO}"
  echo "[e2e] Generated (via bridge): ${GENERATED_PTO}"
else
  "${CONVERTER}" "${TRITON_INPUT}" -o "${GENERATED_PTO}" 2>/dev/null
  echo "[e2e] Generated: ${GENERATED_PTO}"
fi

# Copy into PTOAS tree so docker (mounting PTOAS at /workspace) can see it.
PTO_IN_PTOAS="${PTOAS_ROOT}/test/samples/VectorAddition/e2e_triton_to_pto.pto"
mkdir -p "$(dirname "${PTO_IN_PTOAS}")"
cp "${GENERATED_PTO}" "${PTO_IN_PTOAS}"
WORKSPACE_PATH="/workspace/test/samples/VectorAddition/e2e_triton_to_pto.pto"

echo "[e2e] Running PTOAS sim (docker image: ${DOCKER_IMAGE})..."
if ! docker run --rm \
  -v "${PTOAS_ROOT}:/workspace" \
  -w /workspace \
  "${DOCKER_IMAGE}" \
  bash docker/run_sim_example.sh "${WORKSPACE_PATH}"; then
  echo "[e2e] FAILED: docker or compare failed." >&2
  exit 1
fi
echo "[e2e] PASSED: compare passed."
