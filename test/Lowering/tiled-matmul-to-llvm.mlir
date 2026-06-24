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

// End-to-end lowering of the transform-dialect-tiled matmul from tensor/
// linalg all the way down to the llvm dialect. No tensor, linalg, scf, or
// memref ops should remain.

// CHECK-LABEL: llvm.func @tiled_matmul
// CHECK: llvm.call @malloc
// CHECK: llvm.fmul
// CHECK: llvm.fadd
// CHECK-NOT: linalg.
// CHECK-NOT: scf.
// CHECK-NOT: tensor.
// CHECK-NOT: memref.
module attributes {transform.with_named_sequence} {
  func.func @tiled_matmul(%A: tensor<4x4xf32>, %B: tensor<4x4xf32>) -> tensor<4x4xf32> {
    %init = tensor.empty() : tensor<4x4xf32>
    %cst = arith.constant 0.0 : f32
    %zeroed = linalg.fill ins(%cst : f32) outs(%init : tensor<4x4xf32>) -> tensor<4x4xf32>
    %result = linalg.matmul ins(%A, %B : tensor<4x4xf32>, tensor<4x4xf32>) outs(%zeroed : tensor<4x4xf32>) -> tensor<4x4xf32>
    return %result : tensor<4x4xf32>
  }

  transform.named_sequence private @__transform_main(%module: !transform.any_op {transform.readonly}) {
    %matmul = transform.structured.match ops{["linalg.matmul"]} in %module : (!transform.any_op) -> !transform.any_op
    %tiled, %loops:2 = transform.structured.tile_using_for %matmul tile_sizes [2, 2]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    transform.yield
  }
}
