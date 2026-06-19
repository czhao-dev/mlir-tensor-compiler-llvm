#!/usr/bin/env bash
# Lower a JIT-runnable example down to LLVM dialect with tensor-pipeline-opt,
# then execute it with mlir-runner.
#
# Usage: scripts/run_jit_example.sh <path/to/example_main.mlir> [build-dir]
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <path/to/example_main.mlir> [build-dir]" >&2
  exit 1
fi

INPUT="$1"
BUILD_DIR="${2:-build}"

LLVM_PREFIX="${LLVM_PREFIX:-$(brew --prefix llvm 2>/dev/null || true)}"
if [[ -z "${LLVM_PREFIX}" ]]; then
  echo "error: could not locate LLVM prefix; set LLVM_PREFIX explicitly" >&2
  exit 1
fi

TENSOR_PIPELINE_OPT="${BUILD_DIR}/tools/tensor-pipeline-opt/tensor-pipeline-opt"
MLIR_RUNNER="${LLVM_PREFIX}/bin/mlir-runner"
RUNNER_UTILS="${LLVM_PREFIX}/lib/libmlir_runner_utils.dylib"
C_RUNNER_UTILS="${LLVM_PREFIX}/lib/libmlir_c_runner_utils.dylib"

"${TENSOR_PIPELINE_OPT}" "${INPUT}" \
  -one-shot-bufferize="bufferize-function-boundaries" \
  -convert-linalg-to-loops \
  -lower-affine \
  -convert-scf-to-cf \
  -convert-cf-to-llvm \
  -convert-arith-to-llvm \
  -finalize-memref-to-llvm \
  -convert-func-to-llvm \
  -reconcile-unrealized-casts \
  | "${MLIR_RUNNER}" -e main -entry-point-result=void \
      --shared-libs="${RUNNER_UTILS},${C_RUNNER_UTILS}"
