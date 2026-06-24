// RUN: "%tensor-pipeline-opt" "%s" -transform-interpreter | "FileCheck" "%s"

// Tiling linalg.max and fusing linalg.add into the resulting loop should
// produce a single scf.for whose body contains both ops operating on
// matching tile-sized slices, with no full-size linalg.add/linalg.max left
// outside the loop.

// CHECK-LABEL: func.func @tiled_fused_relu_add
// CHECK: scf.for
// CHECK:   linalg.add ins(%{{.*}}, %{{.*}} : tensor<2x8xf32>, tensor<2x8xf32>)
// CHECK:   linalg.max ins(%{{.*}}, %{{.*}} : tensor<2x8xf32>, tensor<2x8xf32>)
// CHECK-NOT: linalg.add ins(%{{.*}}, %{{.*}} : tensor<8x8xf32>, tensor<8x8xf32>)
// CHECK-NOT: linalg.max ins(%{{.*}}, %{{.*}} : tensor<8x8xf32>, tensor<8x8xf32>)
module attributes {transform.with_named_sequence} {
  func.func @tiled_fused_relu_add(%A: tensor<8x8xf32>, %B: tensor<8x8xf32>) -> tensor<8x8xf32> {
    %sum_init = tensor.empty() : tensor<8x8xf32>
    %sum = linalg.add ins(%A, %B : tensor<8x8xf32>, tensor<8x8xf32>) outs(%sum_init : tensor<8x8xf32>) -> tensor<8x8xf32>
    %zero_init = tensor.empty() : tensor<8x8xf32>
    %cst = arith.constant 0.0 : f32
    %zeroed = linalg.fill ins(%cst : f32) outs(%zero_init : tensor<8x8xf32>) -> tensor<8x8xf32>
    %relu_init = tensor.empty() : tensor<8x8xf32>
    %relu = linalg.max ins(%sum, %zeroed : tensor<8x8xf32>, tensor<8x8xf32>) outs(%relu_init : tensor<8x8xf32>) -> tensor<8x8xf32>
    return %relu : tensor<8x8xf32>
  }

  transform.named_sequence private @__transform_main(%module: !transform.any_op {transform.readonly}) {
    %max = transform.structured.match ops{["linalg.max"]} in %module : (!transform.any_op) -> !transform.any_op
    %tiled, %loop = transform.structured.tile_using_for %max tile_sizes [2]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    %add = transform.structured.match ops{["linalg.add"]} in %module : (!transform.any_op) -> !transform.any_op
    %fused_add, %new_loop = transform.structured.fuse_into_containing_op %add into %loop
      : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}
