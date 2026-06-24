// JIT-runnable version of vectorized_matmul.mlir. Same inputs as
// matmul_main.mlir, so the result should match: A x B = [[19,22],[43,50]].
//
// Run end to end with:
//
//   PRE_PASSES="-transform-interpreter" scripts/run_jit_example.sh examples/vectorized_matmul_main.mlir

module attributes {transform.with_named_sequence} {
  func.func private @printMemrefF32(memref<*xf32>)

  func.func @main() {
    %A = arith.constant dense<[[1.0, 2.0], [3.0, 4.0]]> : tensor<2x2xf32>
    %B = arith.constant dense<[[5.0, 6.0], [7.0, 8.0]]> : tensor<2x2xf32>
    %init = tensor.empty() : tensor<2x2xf32>
    %cst = arith.constant 0.0 : f32
    %zeroed = linalg.fill ins(%cst : f32) outs(%init : tensor<2x2xf32>) -> tensor<2x2xf32>
    %result = linalg.matmul ins(%A, %B : tensor<2x2xf32>, tensor<2x2xf32>) outs(%zeroed : tensor<2x2xf32>) -> tensor<2x2xf32>
    %mem = bufferization.to_buffer %result : tensor<2x2xf32> to memref<2x2xf32>
    %umem = memref.cast %mem : memref<2x2xf32> to memref<*xf32>
    call @printMemrefF32(%umem) : (memref<*xf32>) -> ()
    return
  }

  transform.named_sequence private @__transform_main(%module: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %module : (!transform.any_op) -> !transform.any_op
    %vectorized = transform.structured.vectorize_children_and_apply_patterns %func : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}
