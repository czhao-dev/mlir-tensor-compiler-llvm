// RUN: "%tensor-pipeline-opt" "%s" -transform-interpreter | "FileCheck" "%s"

// The transform dialect schedule should vectorize the whole 4x4 matmul into
// a single vector.contract over vector.transfer_read/transfer_write -- no
// loops, since the shape is small and static.

// CHECK-LABEL: func.func @vectorized_matmul
// CHECK: vector.transfer_read
// CHECK: vector.transfer_read
// CHECK: vector.contract
// CHECK: vector.transfer_write
// CHECK-NOT: linalg.matmul
// CHECK-NOT: linalg.fill
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
