// RUN: "%tensor-pipeline-opt" "%s" -convert-ttensor-to-linalg | "FileCheck" "%s"

// ttensor.relu_add should expand into the same tensor.empty / linalg.add /
// linalg.fill / linalg.max sequence as fused_relu_add.mlir, with no
// ttensor ops remaining.

// CHECK-LABEL: func.func @ttensor_relu_add
// CHECK: linalg.add
// CHECK: linalg.fill
// CHECK: linalg.max
// CHECK-NOT: ttensor.
func.func @ttensor_relu_add(%A: tensor<4x4xf32>, %B: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %result = ttensor.relu_add %A, %B : tensor<4x4xf32>
  return %result : tensor<4x4xf32>
}
