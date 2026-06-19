// RUN: "%tensor-pipeline-opt" "%s" \
// RUN:   -one-shot-bufferize="bufferize-function-boundaries" \
// RUN:   -convert-linalg-to-loops | "FileCheck" "%s"

// Matrix multiplication should lower to a zero-fill loop nest followed by
// a triply-nested scf.for loop computing the dot products, with no
// linalg ops remaining.

// CHECK-LABEL: func.func @matmul
// CHECK: scf.for
// CHECK:   scf.for
// CHECK: scf.for
// CHECK:   scf.for
// CHECK:     scf.for
// CHECK: arith.mulf
// CHECK: arith.addf
// CHECK-NOT: linalg.matmul
// CHECK-NOT: linalg.fill
func.func @matmul(%A: tensor<4x4xf32>, %B: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %init = tensor.empty() : tensor<4x4xf32>
  %cst = arith.constant 0.0 : f32
  %zeroed = linalg.fill ins(%cst : f32) outs(%init : tensor<4x4xf32>) -> tensor<4x4xf32>
  %result = linalg.matmul ins(%A, %B : tensor<4x4xf32>, tensor<4x4xf32>) outs(%zeroed : tensor<4x4xf32>) -> tensor<4x4xf32>
  return %result : tensor<4x4xf32>
}
