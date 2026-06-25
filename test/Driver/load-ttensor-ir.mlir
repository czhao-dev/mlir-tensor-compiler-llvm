// RUN: "%tensor-pipeline-opt" "%s" | "FileCheck" "%s"

// Sanity check that the driver can load and round-trip the custom ttensor
// dialect without running any passes.

// CHECK-LABEL: func.func @ttensor_relu_add
// CHECK: ttensor.relu_add
func.func @ttensor_relu_add(%A: tensor<4x4xf32>, %B: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %result = ttensor.relu_add %A, %B : tensor<4x4xf32>
  return %result : tensor<4x4xf32>
}
