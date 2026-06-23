// RUN: "%tensor-pipeline-opt" "%s" \
// RUN:   -one-shot-bufferize="bufferize-function-boundaries" \
// RUN:   -convert-linalg-to-loops | "FileCheck" "%s"

// 2D convolution should lower to nested scf.for loops performing a
// multiply-accumulate per output element, with no linalg ops remaining.

// CHECK-LABEL: func.func @conv2d
// CHECK: scf.for
// CHECK: scf.for
// CHECK: arith.mulf
// CHECK: arith.addf
// CHECK-NOT: linalg.conv_2d_nhwc_hwcf
// CHECK-NOT: linalg.fill
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
