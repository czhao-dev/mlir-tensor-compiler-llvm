// 2D convolution (NHWC input, HWCF filter), no padding, unit stride.
//
// High-level entry point expressed with linalg-on-tensors. Lower it end to
// end with the same pass pipeline as elementwise_add.mlir:
//
//   tensor-pipeline-opt examples/conv2d.mlir \
//     -one-shot-bufferize="bufferize-function-boundaries" \
//     -convert-linalg-to-loops \
//     -lower-affine \
//     -convert-scf-to-cf \
//     -convert-cf-to-llvm \
//     -convert-arith-to-llvm \
//     -finalize-memref-to-llvm \
//     -convert-func-to-llvm \
//     -reconcile-unrealized-casts

func.func @conv2d(%input: tensor<1x5x5x1xf32>, %filter: tensor<3x3x1x1xf32>) -> tensor<1x3x3x1xf32> {
  %init = tensor.empty() : tensor<1x3x3x1xf32>
  %cst = arith.constant 0.0 : f32
  %zeroed = linalg.fill ins(%cst : f32) outs(%init : tensor<1x3x3x1xf32>) -> tensor<1x3x3x1xf32>
  %result = linalg.conv_2d_nhwc_hwcf
    {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
    ins(%input, %filter : tensor<1x5x5x1xf32>, tensor<3x3x1x1xf32>)
    outs(%zeroed : tensor<1x3x3x1xf32>) -> tensor<1x3x3x1xf32>
  return %result : tensor<1x3x3x1xf32>
}
