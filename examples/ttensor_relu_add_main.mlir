// JIT-runnable version of ttensor_relu_add.mlir. Same inputs as
// fused_relu_add_main.mlir, so the result should match: relu(A + B) =
// [[3, 0], [0, 3]] -- the point being that ttensor.relu_add and the
// hand-written tensor.empty/linalg.add/linalg.fill/linalg.max sequence in
// fused_relu_add_main.mlir are computing exactly the same thing.
//
// Run end to end with:
//
//   PRE_PASSES="-convert-ttensor-to-linalg" scripts/run_jit_example.sh examples/ttensor_relu_add_main.mlir

func.func private @printMemrefF32(memref<*xf32>)

func.func @main() {
  %A = arith.constant dense<[[1.0, -5.0], [3.0, -2.0]]> : tensor<2x2xf32>
  %B = arith.constant dense<[[2.0, 1.0], [-10.0, 5.0]]> : tensor<2x2xf32>
  %result = ttensor.relu_add %A, %B : tensor<2x2xf32>
  %mem = bufferization.to_buffer %result : tensor<2x2xf32> to memref<2x2xf32>
  %umem = memref.cast %mem : memref<2x2xf32> to memref<*xf32>
  call @printMemrefF32(%umem) : (memref<*xf32>) -> ()
  return
}
