// JIT-runnable version of tiled_matmul.mlir. Computes A x I for a constant
// 4x4 matrix A and the 4x4 identity I, tiled into a 2x2 grid of 2x2 tiles via
// the transform dialect. Since A x I = A, the printed result should be A
// itself unchanged -- a simple way to confirm the tiling/slicing is correct.
//
// Run end to end with:
//
//   PRE_PASSES="-transform-interpreter" scripts/run_jit_example.sh examples/tiled_matmul_main.mlir

module attributes {transform.with_named_sequence} {
  func.func private @printMemrefF32(memref<*xf32>)

  func.func @main() {
    %A = arith.constant dense<[[1.0, 2.0, 3.0, 4.0],
                                [5.0, 6.0, 7.0, 8.0],
                                [9.0, 10.0, 11.0, 12.0],
                                [13.0, 14.0, 15.0, 16.0]]> : tensor<4x4xf32>
    %I = arith.constant dense<[[1.0, 0.0, 0.0, 0.0],
                                [0.0, 1.0, 0.0, 0.0],
                                [0.0, 0.0, 1.0, 0.0],
                                [0.0, 0.0, 0.0, 1.0]]> : tensor<4x4xf32>
    %init = tensor.empty() : tensor<4x4xf32>
    %cst = arith.constant 0.0 : f32
    %zeroed = linalg.fill ins(%cst : f32) outs(%init : tensor<4x4xf32>) -> tensor<4x4xf32>
    %result = linalg.matmul ins(%A, %I : tensor<4x4xf32>, tensor<4x4xf32>) outs(%zeroed : tensor<4x4xf32>) -> tensor<4x4xf32>
    %mem = bufferization.to_buffer %result : tensor<4x4xf32> to memref<4x4xf32>
    %umem = memref.cast %mem : memref<4x4xf32> to memref<*xf32>
    call @printMemrefF32(%umem) : (memref<*xf32>) -> ()
    return
  }

  transform.named_sequence private @__transform_main(%module: !transform.any_op {transform.readonly}) {
    %matmul = transform.structured.match ops{["linalg.matmul"]} in %module : (!transform.any_op) -> !transform.any_op
    %tiled, %loops:2 = transform.structured.tile_using_for %matmul tile_sizes [2, 2]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    transform.yield
  }
}
