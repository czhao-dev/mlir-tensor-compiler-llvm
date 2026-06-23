// RUN: "%tensor-pipeline-opt" "%s" \
// RUN:   -linalg-generalize-named-ops \
// RUN:   -linalg-fuse-elementwise-ops | "FileCheck" "%s"

// linalg.add and linalg.max should fuse into a single linalg.generic that
// computes the sum and the max in one pass, with the zero-fill folded
// away entirely.

// CHECK-LABEL: func.func @fused_relu_add
// CHECK: linalg.generic
// CHECK: arith.addf
// CHECK: arith.maximumf
// CHECK-NOT: linalg.generic
// CHECK-NOT: linalg.fill
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
