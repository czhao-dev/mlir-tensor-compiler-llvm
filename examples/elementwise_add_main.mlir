// JIT-runnable version of elementwise_add.mlir. Computes A + B for two
// constant 2x2 tensors and prints the result via the MLIR runner utils.
//
// Run end to end with:
//
//   scripts/run_jit_example.sh examples/elementwise_add_main.mlir

func.func private @printMemrefF32(memref<*xf32>)

func.func @main() {
  %A = arith.constant dense<[[1.0, 2.0], [3.0, 4.0]]> : tensor<2x2xf32>
  %B = arith.constant dense<[[5.0, 6.0], [7.0, 8.0]]> : tensor<2x2xf32>
  %init = tensor.empty() : tensor<2x2xf32>
  %sum = linalg.add ins(%A, %B : tensor<2x2xf32>, tensor<2x2xf32>) outs(%init : tensor<2x2xf32>) -> tensor<2x2xf32>
  %mem = bufferization.to_buffer %sum : tensor<2x2xf32> to memref<2x2xf32>
  %umem = memref.cast %mem : memref<2x2xf32> to memref<*xf32>
  call @printMemrefF32(%umem) : (memref<*xf32>) -> ()
  return
}
