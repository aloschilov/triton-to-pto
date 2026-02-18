#!/usr/bin/env bash
# Check whether any Triton ref pins the same llvm-project commit as llvmorg-19.1.7.
# Usage: ./check_triton_llvm_hash.sh
# Requires: curl, git (for ls-remote). Run with network access.

set -e

LLVM_TAG="${LLVM_TAG:-llvmorg-19.1.7}"
TRITON_REFS="${TRITON_REFS:-v3.0.0 v3.1.0 v3.2.0 v3.3.0 v3.3.1 v3.4.0 v3.5.0 v3.5.1 v3.6.0 main}"

echo "Resolving ${LLVM_TAG} to commit..."
H_19_1_7=$(git ls-remote https://github.com/llvm/llvm-project "refs/tags/${LLVM_TAG}^{}" 2>/dev/null | awk '{print $1}')
if [ -z "$H_19_1_7" ]; then
  echo "Failed to resolve ${LLVM_TAG}. Try: git ls-remote https://github.com/llvm/llvm-project refs/tags/${LLVM_TAG}^{}" >&2
  exit 1
fi
echo "  ${LLVM_TAG} -> ${H_19_1_7}"
echo ""

echo "Fetching cmake/llvm-hash.txt for Triton refs..."
MATCH_REF=""
for ref in $TRITON_REFS; do
  h=$(curl -sL "https://raw.githubusercontent.com/triton-lang/triton/${ref}/cmake/llvm-hash.txt" 2>/dev/null | tr -d ' \n\r' | head -c 40)
  if [ -z "$h" ]; then
    echo "  ${ref}: (failed or missing)"
    continue
  fi
  if [ "$h" = "$H_19_1_7" ]; then
    echo "  ${ref}: ${h}  <-- EXACT MATCH"
    MATCH_REF="$ref"
  else
    echo "  ${ref}: ${h}"
  fi
done

echo ""
if [ -n "$MATCH_REF" ]; then
  echo "Exact hash match found: use TRITON_REF=${MATCH_REF} with LLVM_TAG=${LLVM_TAG}"
else
  echo "No exact hash match for ${LLVM_TAG} (${H_19_1_7}). Use a Triton ref that pins LLVM 19 (e.g. v3.1.0) and document the minor version difference."
fi
