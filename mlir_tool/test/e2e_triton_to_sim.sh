#!/usr/bin/env bash
# All-in-Docker e2e: run converter inside triton-to-pto image, then run PTOAS sim
# inside ptoas image. No host need for triton-to-pto-mlir; only Docker and PTOAS mount.
#
# Usage:
#   PTOAS_ROOT=/path/to/PTOAS mlir_tool/test/e2e_triton_to_sim.sh [triton_input.mlir]
#
# Requires: ptoas:py3.11, triton-to-pto:py3.11. Default input: mlir_tool/test/vec_add_e2e_triton.mlir
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
TRITON_INPUT="${1:-${SCRIPT_DIR}/vec_add_e2e_triton.mlir}"
TRITON_INPUT_ABS="$(cd "$(dirname "${TRITON_INPUT}")" && pwd)/$(basename "${TRITON_INPUT}")"
INPUT_DIR="$(dirname "${TRITON_INPUT_ABS}")"
INPUT_NAME="$(basename "${TRITON_INPUT_ABS}")"
PTO_OUT="/workspace/test/samples/VectorAddition/e2e_triton_to_pto.pto"
N_ELEMENTS="${N_ELEMENTS:-98432}"
TRITON_IMAGE="${TRITON_TO_PTO_IMAGE:-triton-to-pto:py3.11}"
PTOAS_IMAGE="${PTOAS_DOCKER_IMAGE:-ptoas:py3.11}"

if [[ -z "${PTOAS_ROOT:-}" ]]; then
  echo "ERROR: PTOAS_ROOT not set. Point it at the PTOAS repo root." >&2
  exit 1
fi
PTOAS_ROOT="$(cd "${PTOAS_ROOT}" && pwd)"
if [[ ! -f "${PTOAS_ROOT}/docker/run_sim_example.sh" ]]; then
  echo "ERROR: PTOAS_ROOT/docker/run_sim_example.sh not found." >&2
  exit 1
fi
if [[ ! -f "${TRITON_INPUT}" ]]; then
  echo "ERROR: Triton input not found: ${TRITON_INPUT}" >&2
  exit 1
fi

# Step 1: run converter in triton-to-pto image; write .pto into PTOAS tree (mounted at /workspace)
echo "[e2e_triton_to_sim] Converting (in ${TRITON_IMAGE}): ${TRITON_INPUT} -> ${PTO_OUT}"
docker run --rm \
  -v "${PTOAS_ROOT}:/workspace" \
  -v "${INPUT_DIR}:/input:ro" \
  -w /workspace \
  "${TRITON_IMAGE}" \
  bash -c "mkdir -p /workspace/test/samples/VectorAddition && triton-to-pto-mlir /input/${INPUT_NAME} -o ${PTO_OUT}"

# Step 2: run PTOAS sim in ptoas image
echo "[e2e_triton_to_sim] Running PTOAS sim (${PTOAS_IMAGE})..."
if ! docker run --rm \
  -v "${PTOAS_ROOT}:/workspace" \
  -w /workspace \
  "${PTOAS_IMAGE}" \
  bash docker/run_sim_example.sh "${PTO_OUT}" --n-elements "${N_ELEMENTS}"; then
  echo "[e2e_triton_to_sim] FAILED." >&2
  exit 1
fi
echo "[e2e_triton_to_sim] PASSED."
