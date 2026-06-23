// JIT-runnable version of fused_relu_add.mlir. Computes relu(A + B) for
// two constant 2x2 tensors chosen so that the sum has both positive and
// negative entries, exercising the clamp.
//
// A + B = [[3, -4], [-7, 3]], so relu(A + B) = [[3, 0], [0, 3]].
//
// Run end to end with:
//
//   scripts/run_jit_example.sh examples/fused_relu_add_main.mlir

func.func private @printMemrefF32(memref<*xf32>)

func.func @main() {
  %A = arith.constant dense<[[1.0, -5.0], [3.0, -2.0]]> : tensor<2x2xf32>
  %B = arith.constant dense<[[2.0, 1.0], [-10.0, 5.0]]> : tensor<2x2xf32>
  %sum_init = tensor.empty() : tensor<2x2xf32>
  %sum = linalg.add ins(%A, %B : tensor<2x2xf32>, tensor<2x2xf32>) outs(%sum_init : tensor<2x2xf32>) -> tensor<2x2xf32>
  %zero_init = tensor.empty() : tensor<2x2xf32>
  %cst = arith.constant 0.0 : f32
  %zeroed = linalg.fill ins(%cst : f32) outs(%zero_init : tensor<2x2xf32>) -> tensor<2x2xf32>
  %relu_init = tensor.empty() : tensor<2x2xf32>
  %relu = linalg.max ins(%sum, %zeroed : tensor<2x2xf32>, tensor<2x2xf32>) outs(%relu_init : tensor<2x2xf32>) -> tensor<2x2xf32>
  %mem = bufferization.to_buffer %relu : tensor<2x2xf32> to memref<2x2xf32>
  %umem = memref.cast %mem : memref<2x2xf32> to memref<*xf32>
  call @printMemrefF32(%umem) : (memref<*xf32>) -> ()
  return
}
