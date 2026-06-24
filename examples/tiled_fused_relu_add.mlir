// Tiled and loop-fused relu(A + B) = max(A + B, 0) on 8x8 tensors.
//
// This is a different flavor of fusion from fused_relu_add.mlir: instead of
// fusing the two elementwise *ops* into one linalg.generic
// (-linalg-fuse-elementwise-ops), this tiles the consumer (linalg.max) into
// a loop nest and fuses the producer (linalg.add) *into that loop*, so both
// ops share one set of tile-sized slices instead of each materializing a
// full-size intermediate tensor. Run it with:
//
//   tensor-pipeline-opt examples/tiled_fused_relu_add.mlir -transform-interpreter
//
// then continue with the same full lowering pipeline as tiled_matmul.mlir to
// take the fused, tiled loop all the way to the llvm dialect.

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
