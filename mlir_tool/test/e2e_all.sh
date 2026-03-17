#!/usr/bin/env bash
# Run all holistic-rewrite kernels through the full e2e pipeline:
#   Triton IR -> triton-to-pto-mlir -> .pto -> ptoas -> C++ -> Ascend sim
#
# Usage:
#   PTOAS_ROOT=/path/to/PTOAS mlir_tool/test/e2e_all.sh
#
# Requires: Docker images triton-to-pto:py3.11 and ptoas:py3.11.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
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

KERNELS="vec_add_e2e_triton.mlir:VectorAddition:vec_add reduce_sum_triton.mlir:Reduction:reduce_sum unary_exp_triton.mlir:UnaryExp:unary_exp softmax_triton.mlir:Softmax:softmax"

PASS=0
FAIL=0
FAILED_KERNELS=""

for ENTRY in ${KERNELS}; do
  INPUT_NAME="${ENTRY%%:*}"
  rest="${ENTRY#*:}"
  PTOAS_SUBDIR="${rest%%:*}"
  KERNEL_NAME="${rest##*:}"
  INPUT_FILE="${SCRIPT_DIR}/${INPUT_NAME}"
  PTO_OUT="/workspace/test/samples/${PTOAS_SUBDIR}/e2e_triton_to_pto.pto"

  if [[ ! -f "${INPUT_FILE}" ]]; then
    echo "[e2e_all] SKIP: ${INPUT_NAME} (file not found)"
    continue
  fi

  echo ""
  echo "========================================"
  echo "[e2e_all] Testing: ${INPUT_NAME} -> ${PTOAS_SUBDIR}"
  echo "========================================"

  # Step 1: convert Triton IR to PTO
  echo "[e2e_all] Converting..."
  if ! docker run --rm \
    -v "${PTOAS_ROOT}:/workspace" \
    -v "${SCRIPT_DIR}:/input:ro" \
    -w /workspace \
    "${TRITON_IMAGE}" \
    bash -c "mkdir -p /workspace/test/samples/${PTOAS_SUBDIR} && triton-to-pto-mlir /input/${INPUT_NAME} -o ${PTO_OUT}"; then
    echo "[e2e_all] FAIL: ${INPUT_NAME} (converter failed)"
    FAIL=$((FAIL + 1))
    FAILED_KERNELS="${FAILED_KERNELS} ${INPUT_NAME}"
    continue
  fi

  # Step 2: run ptoas + sim
  echo "[e2e_all] Running ptoas + sim..."
  if ! docker run --rm \
    -v "${PTOAS_ROOT}:/workspace" \
    -w /workspace \
    "${PTOAS_IMAGE}" \
    bash docker/run_sim_example.sh "${PTO_OUT}"; then
    echo "[e2e_all] FAIL: ${INPUT_NAME} (ptoas/sim failed)"
    FAIL=$((FAIL + 1))
    FAILED_KERNELS="${FAILED_KERNELS} ${INPUT_NAME}"
    continue
  fi

  # Step 2b: CPU-sim for numerical accuracy
  NV_DIR="/workspace/test/samples/${PTOAS_SUBDIR}/npu_validation/e2e_triton_to/"
  PTO_CPP="/workspace/test/samples/${PTOAS_SUBDIR}/e2e_triton_to_pto.cpp"
  echo "[e2e_all] Running CPU sim..."
  if ! docker run --rm \
    -v "${PTOAS_ROOT}:/workspace" \
    -v "${SCRIPT_DIR}/cpu_sim:/cpu_sim:ro" \
    "${PTOAS_IMAGE}" \
    python3 /cpu_sim/cpu_sim_run.py "${PTO_CPP}" "${NV_DIR}"; then
    echo "[e2e_all] FAIL: ${INPUT_NAME} (cpu-sim failed)"
    FAIL=$((FAIL + 1))
    FAILED_KERNELS="${FAILED_KERNELS} ${INPUT_NAME}"
    continue
  fi

  # Step 3: independent golden accuracy check
  echo "[e2e_all] Running golden accuracy check..."
  if ! docker run --rm \
    -v "${PTOAS_ROOT}:/workspace" \
    -v "${SCRIPT_DIR}/golden:/golden:ro" \
    "${PTOAS_IMAGE}" \
    python3 /golden/golden_check.py "${KERNEL_NAME}" "${NV_DIR}"; then
    echo "[e2e_all] FAIL: ${INPUT_NAME} (golden check failed)"
    FAIL=$((FAIL + 1))
    FAILED_KERNELS="${FAILED_KERNELS} ${INPUT_NAME}"
    continue
  fi

  echo "[e2e_all] PASS: ${INPUT_NAME}"
  PASS=$((PASS + 1))
done

echo ""
echo "========================================"
echo "[e2e_all] Results: ${PASS} passed, ${FAIL} failed (of $((PASS + FAIL)) total)"
if [[ ${FAIL} -gt 0 ]]; then
  echo "[e2e_all] Failed:${FAILED_KERNELS}"
  exit 1
fi
echo "[e2e_all] All kernels passed."
