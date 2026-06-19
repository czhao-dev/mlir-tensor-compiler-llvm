// RUN: "%tensor-pipeline-opt" "%s" | "FileCheck" "%s"

// Sanity check that the driver can load and round-trip a high-level
// tensor/linalg input without running any passes.

// CHECK-LABEL: func.func @elementwise_add
// CHECK: linalg.add
func.func @elementwise_add(%A: tensor<4x4xf32>, %B: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %init = tensor.empty() : tensor<4x4xf32>
  %sum = linalg.add ins(%A, %B : tensor<4x4xf32>, tensor<4x4xf32>) outs(%init : tensor<4x4xf32>) -> tensor<4x4xf32>
  return %sum : tensor<4x4xf32>
}
