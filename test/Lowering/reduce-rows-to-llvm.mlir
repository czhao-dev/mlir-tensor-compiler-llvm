// RUN: "%tensor-pipeline-opt" "%s" \
// RUN:   -one-shot-bufferize="bufferize-function-boundaries" \
// RUN:   -convert-linalg-to-loops \
// RUN:   -expand-strided-metadata \
// RUN:   -convert-vector-to-scf \
// RUN:   -lower-affine \
// RUN:   -convert-scf-to-cf \
// RUN:   -convert-cf-to-llvm \
// RUN:   -convert-vector-to-llvm \
// RUN:   -convert-arith-to-llvm \
// RUN:   -convert-ub-to-llvm \
// RUN:   -finalize-memref-to-llvm \
// RUN:   -convert-func-to-llvm \
// RUN:   -reconcile-unrealized-casts \
// RUN:   -symbol-dce | "FileCheck" "%s"

// End-to-end lowering of the row reduction from tensor/linalg all the way
// down to the llvm dialect. No tensor, linalg, or scf ops should remain.

// CHECK-LABEL: llvm.func @reduce_rows
// CHECK: llvm.fadd
// CHECK-NOT: linalg.
// CHECK-NOT: scf.
// CHECK-NOT: tensor.
func.func @reduce_rows(%A: tensor<4x4xf32>) -> tensor<4xf32> {
  %init = tensor.empty() : tensor<4xf32>
  %cst = arith.constant 0.0 : f32
  %zeroed = linalg.fill ins(%cst : f32) outs(%init : tensor<4xf32>) -> tensor<4xf32>
  %result = linalg.reduce { arith.addf } ins(%A : tensor<4x4xf32>) outs(%zeroed : tensor<4xf32>) dimensions = [1]
  return %result : tensor<4xf32>
}
