// JIT-runnable version of reduce_rows.mlir. Sums each row of a constant 2x3
// tensor and prints the result via the MLIR runner utils.
//
// Run end to end with:
//
//   scripts/run_jit_example.sh examples/reduce_rows_main.mlir

func.func private @printMemrefF32(memref<*xf32>)

func.func @main() {
  %A = arith.constant dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]> : tensor<2x3xf32>
  %init = tensor.empty() : tensor<2xf32>
  %cst = arith.constant 0.0 : f32
  %zeroed = linalg.fill ins(%cst : f32) outs(%init : tensor<2xf32>) -> tensor<2xf32>
  %sum = linalg.reduce { arith.addf } ins(%A : tensor<2x3xf32>) outs(%zeroed : tensor<2xf32>) dimensions = [1]
  %mem = bufferization.to_buffer %sum : tensor<2xf32> to memref<2xf32>
  %umem = memref.cast %mem : memref<2xf32> to memref<*xf32>
  call @printMemrefF32(%umem) : (memref<*xf32>) -> ()
  return
}
