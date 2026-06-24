// RUN: "%tensor-pipeline-opt" "%s" \
// RUN:   -transform-interpreter \
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

// End-to-end lowering of the transform-dialect-vectorized matmul from
// tensor/linalg all the way down to the llvm dialect. No tensor, linalg, or
// vector ops should remain; the vector.contract becomes a sequence of
// extractvalue/fmul/fadd/vector-reduce operating on 1-D LLVM vectors.

// CHECK-LABEL: llvm.func @vectorized_matmul
// CHECK: llvm.intr.vector.reduce.fadd
// CHECK-NOT: linalg.
// CHECK-NOT: tensor.
// CHECK-NOT: vector.contract
// CHECK-NOT: vector.transfer
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
