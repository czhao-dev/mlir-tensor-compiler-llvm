// Fused elementwise operation: relu(A + B) = max(A + B, 0)
//
// High-level entry point expressed with linalg-on-tensors. Lower it end to
// end with the same pass pipeline as elementwise_add.mlir:
//
//   tensor-pipeline-opt examples/fused_relu_add.mlir \
//     -one-shot-bufferize="bufferize-function-boundaries" \
//     -convert-linalg-to-loops \
//     -lower-affine \
//     -convert-scf-to-cf \
//     -convert-cf-to-llvm \
//     -convert-arith-to-llvm \
//     -finalize-memref-to-llvm \
//     -convert-func-to-llvm \
//     -reconcile-unrealized-casts
//
// linalg.add and linalg.max are two separate named ops here. Run
//
//   tensor-pipeline-opt examples/fused_relu_add.mlir \
//     -linalg-generalize-named-ops -linalg-fuse-elementwise-ops
//
// to see them fused into a single linalg.generic that computes both the
// sum and the max in one pass over the data, with the zero-fill folded
// away entirely.

func.func @fused_relu_add(%A: tensor<4x4xf32>, %B: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %sum_init = tensor.empty() : tensor<4x4xf32>
  %sum = linalg.add ins(%A, %B : tensor<4x4xf32>, tensor<4x4xf32>) outs(%sum_init : tensor<4x4xf32>) -> tensor<4x4xf32>
  %zero_init = tensor.empty() : tensor<4x4xf32>
  %cst = arith.constant 0.0 : f32
  %zeroed = linalg.fill ins(%cst : f32) outs(%zero_init : tensor<4x4xf32>) -> tensor<4x4xf32>
  %relu_init = tensor.empty() : tensor<4x4xf32>
  %relu = linalg.max ins(%sum, %zeroed : tensor<4x4xf32>, tensor<4x4xf32>) outs(%relu_init : tensor<4x4xf32>) -> tensor<4x4xf32>
  return %relu : tensor<4x4xf32>
}
