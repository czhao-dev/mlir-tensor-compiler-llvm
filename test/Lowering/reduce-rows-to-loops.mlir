// RUN: "%tensor-pipeline-opt" "%s" \
// RUN:   -one-shot-bufferize="bufferize-function-boundaries" \
// RUN:   -convert-linalg-to-loops | "FileCheck" "%s"

// Row reduction should lower to a zero-fill loop followed by a doubly-nested
// scf.for loop that accumulates each row with arith.addf, with no linalg ops
// remaining.

// CHECK-LABEL: func.func @reduce_rows
// CHECK: scf.for
// CHECK: scf.for
// CHECK:   scf.for
// CHECK:     arith.addf
// CHECK-NOT: linalg.reduce
// CHECK-NOT: linalg.fill
func.func @reduce_rows(%A: tensor<4x4xf32>) -> tensor<4xf32> {
  %init = tensor.empty() : tensor<4xf32>
  %cst = arith.constant 0.0 : f32
  %zeroed = linalg.fill ins(%cst : f32) outs(%init : tensor<4xf32>) -> tensor<4xf32>
  %result = linalg.reduce { arith.addf } ins(%A : tensor<4x4xf32>) outs(%zeroed : tensor<4xf32>) dimensions = [1]
  return %result : tensor<4xf32>
}
