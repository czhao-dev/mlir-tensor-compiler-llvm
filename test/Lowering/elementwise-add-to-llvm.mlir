// RUN: "%tensor-pipeline-opt" "%s" \
// RUN:   -one-shot-bufferize="bufferize-function-boundaries" \
// RUN:   -convert-linalg-to-loops \
// RUN:   -lower-affine \
// RUN:   -convert-scf-to-cf \
// RUN:   -convert-cf-to-llvm \
// RUN:   -convert-arith-to-llvm \
// RUN:   -finalize-memref-to-llvm \
// RUN:   -convert-func-to-llvm \
// RUN:   -reconcile-unrealized-casts | "FileCheck" "%s"

// End-to-end lowering of elementwise add from tensor/linalg all the way
// down to the llvm dialect. No tensor, linalg, scf, memref, or arith ops
// should remain; the function becomes an llvm.func with malloc/fadd/gep.

// CHECK-LABEL: llvm.func @elementwise_add
// CHECK: llvm.call @malloc
// CHECK: llvm.fadd
// CHECK-NOT: linalg.
// CHECK-NOT: scf.
// CHECK-NOT: tensor.
func.func @elementwise_add(%A: tensor<4x4xf32>, %B: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %init = tensor.empty() : tensor<4x4xf32>
  %sum = linalg.add ins(%A, %B : tensor<4x4xf32>, tensor<4x4xf32>) outs(%init : tensor<4x4xf32>) -> tensor<4x4xf32>
  return %sum : tensor<4x4xf32>
}
