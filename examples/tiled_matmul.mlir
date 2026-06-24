// Matrix multiplication, tiled via the transform dialect: C = A x B, computed
// as a 2x2 grid of 2x2 tiles instead of a single 4x4 linalg.matmul.
//
// The transform.named_sequence below runs before bufferization (it operates
// on tensor-level linalg ops) and rewrites the single linalg.matmul into a
// pair of nested scf.for tiling loops, each iteration computing one 2x2
// output tile via tensor.extract_slice / linalg.matmul / tensor.insert_slice.
// Run it with:
//
//   tensor-pipeline-opt examples/tiled_matmul.mlir -transform-interpreter
//
// then continue with the usual lowering pipeline to take the tiled loops all
// the way to the llvm dialect (the tiling introduces memref.subview ops once
// bufferized, which need -expand-strided-metadata before -lower-affine):
//
//   tensor-pipeline-opt examples/tiled_matmul.mlir \
//     -transform-interpreter \
//     -one-shot-bufferize="bufferize-function-boundaries" \
//     -convert-linalg-to-loops \
//     -expand-strided-metadata \
//     -convert-vector-to-scf \
//     -lower-affine \
//     -convert-scf-to-cf \
//     -convert-cf-to-llvm \
//     -convert-vector-to-llvm \
//     -convert-arith-to-llvm \
//     -convert-ub-to-llvm \
//     -finalize-memref-to-llvm \
//     -convert-func-to-llvm \
//     -reconcile-unrealized-casts \
//     -symbol-dce

module attributes {transform.with_named_sequence} {
  func.func @tiled_matmul(%A: tensor<4x4xf32>, %B: tensor<4x4xf32>) -> tensor<4x4xf32> {
    %init = tensor.empty() : tensor<4x4xf32>
    %cst = arith.constant 0.0 : f32
    %zeroed = linalg.fill ins(%cst : f32) outs(%init : tensor<4x4xf32>) -> tensor<4x4xf32>
    %result = linalg.matmul ins(%A, %B : tensor<4x4xf32>, tensor<4x4xf32>) outs(%zeroed : tensor<4x4xf32>) -> tensor<4x4xf32>
    return %result : tensor<4x4xf32>
  }

  transform.named_sequence private @__transform_main(%module: !transform.any_op {transform.readonly}) {
    %matmul = transform.structured.match ops{["linalg.matmul"]} in %module : (!transform.any_op) -> !transform.any_op
    %tiled, %loops:2 = transform.structured.tile_using_for %matmul tile_sizes [2, 2]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    transform.yield
  }
}
