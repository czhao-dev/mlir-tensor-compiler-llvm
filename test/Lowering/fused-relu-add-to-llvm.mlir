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

// End-to-end lowering of relu(A + B) from tensor/linalg all the way down
// to the llvm dialect. No tensor, linalg, or scf ops should remain.

// CHECK-LABEL: llvm.func @fused_relu_add
// CHECK: llvm.fadd
// CHECK: llvm.intr.maximum
// CHECK-NOT: linalg.
// CHECK-NOT: scf.
// CHECK-NOT: tensor.
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
