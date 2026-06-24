// JIT-runnable version of tiled_fused_relu_add.mlir. A is a constant 2.0
// everywhere; B is -10.0 in its top half and +10.0 in its bottom half. So
// relu(A + B) is 0 in the top half (2 - 10 clamped to 0) and 12 in the
// bottom half (2 + 10), exercising the tiled+fused loop across the boundary
// between the two halves.
//
// Run end to end with:
//
//   PRE_PASSES="-transform-interpreter" scripts/run_jit_example.sh examples/tiled_fused_relu_add_main.mlir

module attributes {transform.with_named_sequence} {
  func.func private @printMemrefF32(memref<*xf32>)

  func.func @main() {
    %A = arith.constant dense<2.0> : tensor<8x8xf32>
    %B = arith.constant dense<[
      [-10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0],
      [-10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0],
      [-10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0],
      [-10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0],
      [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0],
      [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0],
      [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0],
      [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0]]> : tensor<8x8xf32>
    %sum_init = tensor.empty() : tensor<8x8xf32>
    %sum = linalg.add ins(%A, %B : tensor<8x8xf32>, tensor<8x8xf32>) outs(%sum_init : tensor<8x8xf32>) -> tensor<8x8xf32>
    %zero_init = tensor.empty() : tensor<8x8xf32>
    %cst = arith.constant 0.0 : f32
    %zeroed = linalg.fill ins(%cst : f32) outs(%zero_init : tensor<8x8xf32>) -> tensor<8x8xf32>
    %relu_init = tensor.empty() : tensor<8x8xf32>
    %relu = linalg.max ins(%sum, %zeroed : tensor<8x8xf32>, tensor<8x8xf32>) outs(%relu_init : tensor<8x8xf32>) -> tensor<8x8xf32>
    %mem = bufferization.to_buffer %relu : tensor<8x8xf32> to memref<8x8xf32>
    %umem = memref.cast %mem : memref<8x8xf32> to memref<*xf32>
    call @printMemrefF32(%umem) : (memref<*xf32>) -> ()
    return
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
