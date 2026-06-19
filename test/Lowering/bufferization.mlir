// RUN: "%tensor-pipeline-opt" "%s" -one-shot-bufferize="bufferize-function-boundaries" | "FileCheck" "%s"

// Tensors at the function boundary should become memrefs after
// one-shot-bufferize, and the linalg op should now operate on memrefs.

// CHECK-LABEL: func.func @elementwise_add
// CHECK-SAME: memref<4x4xf32
// CHECK-SAME: memref<4x4xf32
// CHECK: memref.alloc
// CHECK: linalg.add
// CHECK-SAME: memref<4x4xf32
func.func @elementwise_add(%A: tensor<4x4xf32>, %B: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %init = tensor.empty() : tensor<4x4xf32>
  %sum = linalg.add ins(%A, %B : tensor<4x4xf32>, tensor<4x4xf32>) outs(%init : tensor<4x4xf32>) -> tensor<4x4xf32>
  return %sum : tensor<4x4xf32>
}
