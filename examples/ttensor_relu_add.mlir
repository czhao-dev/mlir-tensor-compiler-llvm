// relu(A + B), expressed with the custom ttensor dialect instead of spelling
// out tensor.empty/linalg.add/linalg.fill/linalg.max by hand (compare with
// fused_relu_add.mlir, which computes the same thing using only upstream
// dialects).
//
// ttensor.relu_add is a single high-level op defined in include/TTensor/
// purely as a teaching example of MLIR's dialect/ODS system: it carries no
// new lowering machinery of its own. -convert-ttensor-to-linalg expands it
// into exactly the same tensor/linalg sequence as fused_relu_add.mlir, so
// the rest of the pipeline (bufferization, linalg-to-loops, ..., llvm) never
// has to know the ttensor dialect exists. Lower it end to end with:
//
//   tensor-pipeline-opt examples/ttensor_relu_add.mlir \
//     -convert-ttensor-to-linalg \
//     -one-shot-bufferize="bufferize-function-boundaries" \
//     -convert-linalg-to-loops \
//     -expand-strided-metadata \
//     -convert-vector-to-scf \
//     -lower-affine \
//     -convert-scf-to-cf \
//     -convert-cf-to-llvm \
//     -convert-vector-to-llvm \
//     -convert-arith-to-llvm \
//     -convert-ub-to-llvm \
//     -finalize-memref-to-llvm \
//     -convert-func-to-llvm \
//     -reconcile-unrealized-casts \
//     -symbol-dce

func.func @ttensor_relu_add(%A: tensor<4x4xf32>, %B: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %result = ttensor.relu_add %A, %B : tensor<4x4xf32>
  return %result : tensor<4x4xf32>
}
