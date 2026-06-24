// RUN: "%tensor-pipeline-opt" "%s" -transform-interpreter | "FileCheck" "%s"

// The transform dialect schedule should tile the 4x4 linalg.matmul into a
// 2x2 grid of 2x2-tile matmuls, expressed as a pair of nested scf.for loops
// over tensor.extract_slice / linalg.matmul / tensor.insert_slice.

// CHECK-LABEL: func.func @tiled_matmul
// CHECK: scf.for
// CHECK:   scf.for
// CHECK:     tensor.extract_slice
// CHECK:     tensor.extract_slice
// CHECK:     tensor.extract_slice
// CHECK:     linalg.matmul ins(%{{.*}}, %{{.*}} : tensor<2x4xf32>, tensor<4x2xf32>) outs(%{{.*}} : tensor<2x2xf32>) -> tensor<2x2xf32>
// CHECK:     tensor.insert_slice
// CHECK-NOT: linalg.matmul ins(%{{.*}}, %{{.*}} : tensor<4x4xf32>, tensor<4x4xf32>)
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
