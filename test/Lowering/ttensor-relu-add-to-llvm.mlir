// RUN: "%tensor-pipeline-opt" "%s" \
// RUN:   -convert-ttensor-to-linalg \
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

// End-to-end lowering of ttensor.relu_add from the custom ttensor dialect
// all the way down to the llvm dialect. No ttensor, tensor, linalg, or scf
// ops should remain.

// CHECK-LABEL: llvm.func @ttensor_relu_add
// CHECK: llvm.fadd
// CHECK: llvm.intr.maximum
// CHECK-NOT: ttensor.
// CHECK-NOT: linalg.
// CHECK-NOT: scf.
// CHECK-NOT: tensor.
func.func @ttensor_relu_add(%A: tensor<4x4xf32>, %B: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %result = ttensor.relu_add %A, %B : tensor<4x4xf32>
  return %result : tensor<4x4xf32>
}
