// JIT-runnable version of conv2d.mlir. Convolves a constant 5x5
// single-channel input (values 1..25, row-major) with a 3x3 all-ones
// filter (a box-sum filter) and prints the 3x3 result.
//
// Expected result (computed by hand, see README):
//   [ 63,  72,  81]
//   [108, 117, 126]
//   [153, 162, 171]
//
// Run end to end with:
//
//   scripts/run_jit_example.sh examples/conv2d_main.mlir

func.func private @printMemrefF32(memref<*xf32>)

func.func @main() {
  %input = arith.constant dense<[[
      [[1.0], [2.0], [3.0], [4.0], [5.0]],
      [[6.0], [7.0], [8.0], [9.0], [10.0]],
      [[11.0], [12.0], [13.0], [14.0], [15.0]],
      [[16.0], [17.0], [18.0], [19.0], [20.0]],
      [[21.0], [22.0], [23.0], [24.0], [25.0]]
  ]]> : tensor<1x5x5x1xf32>
  %filter = arith.constant dense<1.0> : tensor<3x3x1x1xf32>
  %init = tensor.empty() : tensor<1x3x3x1xf32>
  %cst = arith.constant 0.0 : f32
  %zeroed = linalg.fill ins(%cst : f32) outs(%init : tensor<1x3x3x1xf32>) -> tensor<1x3x3x1xf32>
  %result = linalg.conv_2d_nhwc_hwcf
    {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
    ins(%input, %filter : tensor<1x5x5x1xf32>, tensor<3x3x1x1xf32>)
    outs(%zeroed : tensor<1x3x3x1xf32>) -> tensor<1x3x3x1xf32>
  %mem = bufferization.to_buffer %result : tensor<1x3x3x1xf32> to memref<1x3x3x1xf32>
  %umem = memref.cast %mem : memref<1x3x3x1xf32> to memref<*xf32>
  call @printMemrefF32(%umem) : (memref<*xf32>) -> ()
  return
}
