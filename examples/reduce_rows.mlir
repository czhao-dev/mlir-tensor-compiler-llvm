// Reduction over one dimension: sums each row of a 4x4 tensor down to a
// length-4 vector, e.g. row i of the output is sum_j(A[i][j]).
//
// linalg.reduce takes a combiner region (here, scalar addition) and a list
// of dimensions to reduce away; dimensions = [1] reduces the second
// (column) dimension, leaving one result per row.
//
// Lower it end to end with the same pass pipeline as elementwise_add.mlir:
//
//   tensor-pipeline-opt examples/reduce_rows.mlir \
//     -one-shot-bufferize="bufferize-function-boundaries" \
//     -convert-linalg-to-loops \
//     -lower-affine \
//     -convert-scf-to-cf \
//     -convert-cf-to-llvm \
//     -convert-arith-to-llvm \
//     -finalize-memref-to-llvm \
//     -convert-func-to-llvm \
//     -reconcile-unrealized-casts

func.func @reduce_rows(%A: tensor<4x4xf32>) -> tensor<4xf32> {
  %init = tensor.empty() : tensor<4xf32>
  %cst = arith.constant 0.0 : f32
  %zeroed = linalg.fill ins(%cst : f32) outs(%init : tensor<4xf32>) -> tensor<4xf32>
  %result = linalg.reduce { arith.addf } ins(%A : tensor<4x4xf32>) outs(%zeroed : tensor<4xf32>) dimensions = [1]
  return %result : tensor<4xf32>
}
