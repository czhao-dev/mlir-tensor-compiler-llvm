// Matrix multiplication, vectorized via the transform dialect: the same 4x4
// C = A x B as matmul.mlir, but rewritten as vector.contract over
// vector.transfer_read/transfer_write instead of scalar loops.
//
// Because the shapes are small and static (4x4), vectorization replaces the
// entire linalg.matmul with a single vector.contract -- no loops at all.
// Run it with:
//
//   tensor-pipeline-opt examples/vectorized_matmul.mlir -transform-interpreter
//
// then continue with the usual lowering pipeline (see README "Running the
// Compiler") to take the vector ops all the way to the llvm dialect; that
// pipeline includes -convert-vector-to-scf and -convert-vector-to-llvm
// specifically to handle the multi-dimensional vector ops vectorization
// introduces.

module attributes {transform.with_named_sequence} {
  func.func @vectorized_matmul(%A: tensor<4x4xf32>, %B: tensor<4x4xf32>) -> tensor<4x4xf32> {
    %init = tensor.empty() : tensor<4x4xf32>
    %cst = arith.constant 0.0 : f32
    %zeroed = linalg.fill ins(%cst : f32) outs(%init : tensor<4x4xf32>) -> tensor<4x4xf32>
    %result = linalg.matmul ins(%A, %B : tensor<4x4xf32>, tensor<4x4xf32>) outs(%zeroed : tensor<4x4xf32>) -> tensor<4x4xf32>
    return %result : tensor<4x4xf32>
  }

  transform.named_sequence private @__transform_main(%module: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %module : (!transform.any_op) -> !transform.any_op
    %vectorized = transform.structured.vectorize_children_and_apply_patterns %func : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}
