# Tiling: `transform.structured.tile_using_for`

Tiling splits one `linalg` op into a loop over smaller `linalg` ops, each
operating on a slice of the original tensors. `tiled_matmul.mlir` tiles a
4x4 `linalg.matmul` into a 2x2 grid of 2x2-tile matmuls:

```mlir
%matmul = transform.structured.match ops{["linalg.matmul"]} in %module : (!transform.any_op) -> !transform.any_op
%tiled, %loops:2 = transform.structured.tile_using_for %matmul tile_sizes [2, 2]
  : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
```

```mlir
scf.for %i = %c0 to %c4 step %c2 iter_args(%acc = %zeroed) -> (tensor<4x4xf32>) {
  scf.for %j = %c0 to %c4 step %c2 iter_args(%acc2 = %acc) -> (tensor<4x4xf32>) {
    %a_tile = tensor.extract_slice %A[%i, 0] [2, 4] [1, 1] : tensor<4x4xf32> to tensor<2x4xf32>
    %b_tile = tensor.extract_slice %B[0, %j] [4, 2] [1, 1] : tensor<4x4xf32> to tensor<4x2xf32>
    %c_tile = tensor.extract_slice %acc2[%i, %j] [2, 2] [1, 1] : tensor<4x4xf32> to tensor<2x2xf32>
    %result = linalg.matmul ins(%a_tile, %b_tile : ...) outs(%c_tile : ...) -> tensor<2x2xf32>
    %inserted = tensor.insert_slice %result into %acc2[%i, %j] [2, 2] [1, 1] : tensor<2x2xf32> into tensor<4x4xf32>
    scf.yield %inserted : tensor<4x4xf32>
  }
  scf.yield ... : tensor<4x4xf32>
}
```

This is the building block real tiling strategies are made of: tile sizes
control cache/register footprint, and a tiled op is what
[fusion](06-fusion.md) and parallelization passes actually operate on.

**Driver requirement**: the transform dialect's `transform.structured.*` ops
aren't registered by `registerAllDialects`/`registerAllPasses` alone --
`tensor-pipeline-opt.cpp` calls `mlir::registerAllExtensions(registry)`
specifically to unlock them (see `tools/tensor-pipeline-opt/tensor-pipeline-opt.cpp`).

**Pipeline requirement**: tiling on tensors lowers (after bufferization) to
`memref.subview` with a dynamic, strided layout. `-finalize-memref-to-llvm`
can't convert that directly -- `-expand-strided-metadata` has to run first
to turn the subview's offset/stride into explicit `arith`/`affine`
computations, which is why it appears in the pipeline right after
`-convert-linalg-to-loops` (see [stage 5](05-llvm-conversion.md)).

See `examples/tiled_matmul.mlir`, `test/Lowering/tiled-matmul-tiling.mlir`,
and `test/Lowering/tiled-matmul-to-llvm.mlir`.
