// RUN: "%tensor-pipeline-opt" "%s" \
// RUN:   -one-shot-bufferize="bufferize-function-boundaries" \
// RUN:   -convert-linalg-to-loops | "FileCheck" "%s"

// Elementwise add should lower to a doubly-nested scf.for loop performing
// a scalar arith.addf per element, with no linalg ops remaining.

// CHECK-LABEL: func.func @elementwise_add
// CHECK: scf.for
// CHECK: scf.for
// CHECK: memref.load
// CHECK: memref.load
// CHECK: arith.addf
// CHECK: memref.store
// CHECK-NOT: linalg.add
func.func @elementwise_add(%A: tensor<4x4xf32>, %B: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %init = tensor.empty() : tensor<4x4xf32>
  %sum = linalg.add ins(%A, %B : tensor<4x4xf32>, tensor<4x4xf32>) outs(%init : tensor<4x4xf32>) -> tensor<4x4xf32>
  return %sum : tensor<4x4xf32>
}
