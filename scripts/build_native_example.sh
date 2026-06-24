#!/usr/bin/env bash
# Build a JIT-runnable example into a standalone native executable, instead
# of JIT-executing it: lower to the llvm dialect with tensor-pipeline-opt,
# translate to LLVM IR with mlir-translate, compile to an object file with
# llc, then link it with clang against the MLIR runner utils shared
# libraries (the same ones mlir-runner loads for JIT execution, here linked
# ahead of time instead).
#
# Usage: scripts/build_native_example.sh <path/to/example_main.mlir> [build-dir]
#
# Examples that embed a transform dialect schedule (tiling, fusion,
# vectorization) need that schedule applied before bufferization. Set
# PRE_PASSES to inject passes at the front of the pipeline, same as
# run_jit_example.sh, e.g.:
#
#   PRE_PASSES="-transform-interpreter" scripts/build_native_example.sh examples/tiled_matmul_main.mlir
#
# The resulting executable is run once at the end, so output can be diffed
# against the JIT path's output for the same example.
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <path/to/example_main.mlir> [build-dir]" >&2
  exit 1
fi

INPUT="$1"
BUILD_DIR="${2:-build}"
PRE_PASSES="${PRE_PASSES:-}"

LLVM_PREFIX="${LLVM_PREFIX:-$(brew --prefix llvm 2>/dev/null || true)}"
if [[ -z "${LLVM_PREFIX}" ]]; then
  echo "error: could not locate LLVM prefix; set LLVM_PREFIX explicitly" >&2
  exit 1
fi

TENSOR_PIPELINE_OPT="${BUILD_DIR}/tools/tensor-pipeline-opt/tensor-pipeline-opt"
MLIR_TRANSLATE="${LLVM_PREFIX}/bin/mlir-translate"
LLC="${LLVM_PREFIX}/bin/llc"
RUNNER_UTILS="${LLVM_PREFIX}/lib/libmlir_runner_utils.dylib"
C_RUNNER_UTILS="${LLVM_PREFIX}/lib/libmlir_c_runner_utils.dylib"

NAME="$(basename "${INPUT}" .mlir)"
OUT_DIR="${BUILD_DIR}/native/${NAME}"
mkdir -p "${OUT_DIR}"

LLVM_DIALECT_MLIR="${OUT_DIR}/${NAME}.llvm.mlir"
LLVM_IR="${OUT_DIR}/${NAME}.ll"
OBJECT="${OUT_DIR}/${NAME}.o"
EXE="${OUT_DIR}/${NAME}"

"${TENSOR_PIPELINE_OPT}" "${INPUT}" ${PRE_PASSES} \
  -one-shot-bufferize="bufferize-function-boundaries" \
  -convert-linalg-to-loops \
  -expand-strided-metadata \
  -convert-vector-to-scf \
  -lower-affine \
  -convert-scf-to-cf \
  -convert-cf-to-llvm \
  -convert-vector-to-llvm \
  -convert-arith-to-llvm \
  -convert-ub-to-llvm \
  -finalize-memref-to-llvm \
  -convert-func-to-llvm \
  -reconcile-unrealized-casts \
  -symbol-dce \
  > "${LLVM_DIALECT_MLIR}"

"${MLIR_TRANSLATE}" --mlir-to-llvmir "${LLVM_DIALECT_MLIR}" > "${LLVM_IR}"
"${LLC}" -filetype=obj "${LLVM_IR}" -o "${OBJECT}"
clang "${OBJECT}" -o "${EXE}" \
  -L"${LLVM_PREFIX}/lib" -lmlir_runner_utils -lmlir_c_runner_utils \
  -Wl,-rpath,"${LLVM_PREFIX}/lib"

echo "Built ${EXE}"

# The example's @main returns void in MLIR, but is linked here as the
# process's actual C `main` symbol (which must return an int); the resulting
# exit code is therefore whatever happened to be left in the return
# register, not a meaningful success/failure signal. Run it without letting
# that propagate as this script's own exit status -- correctness is judged
# by the printed memref output, the same as the JIT path.
set +e
"${EXE}"
set -e
