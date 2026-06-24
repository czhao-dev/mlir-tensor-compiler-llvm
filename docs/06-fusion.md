# Fusion: op-level and loop-level

This project demonstrates two distinct flavors of fusion:

## Op fusion: `-linalg-fuse-elementwise-ops`

Merges two elementwise `linalg` ops into a single `linalg.generic` that
computes both in one pass over the data, instead of materializing an
intermediate tensor between them. `fused_relu_add.mlir` generalizes
`linalg.add` + `linalg.max` into `linalg.generic` ops first
(`-linalg-generalize-named-ops`) and then fuses them:

```mlir
// before: two ops, one intermediate tensor
%sum = linalg.add ins(%A, %B : ...) outs(%sum_init : ...) -> tensor<4x4xf32>
%relu = linalg.max ins(%sum, %zeroed : ...) outs(%relu_init : ...) -> tensor<4x4xf32>

// after -linalg-fuse-elementwise-ops: one linalg.generic, no intermediate
%relu = linalg.generic {...} ins(%A, %B, %zeroed : ...) outs(%relu_init : ...) {
  ^bb0(%a: f32, %b: f32, %z: f32, %out: f32):
    %sum = arith.addf %a, %b : f32
    %max = arith.maximumf %sum, %z : f32
    linalg.yield %max : f32
}
```

See `examples/fused_relu_add.mlir` and
`test/Lowering/fused-relu-add-fusion.mlir`.

## Loop fusion: `transform.structured.fuse_into_containing_op`

A different problem: once a consumer is [tiled](07-tiling.md) into a loop,
should its producer compute its *entire* output up front, or just the slice
each tile actually needs? `tiled_fused_relu_add.mlir` tiles `linalg.max` and
then fuses `linalg.add` into the resulting loop, so each iteration computes
one tile-sized slice of the sum and immediately applies relu to it -- no
full-size intermediate tensor exists at any point:

```mlir
scf.for %i = ... {
  %a_slice = tensor.extract_slice %A[%i, 0] [2, 8] [1, 1] : ... to tensor<2x8xf32>
  %b_slice = tensor.extract_slice %B[%i, 0] [2, 8] [1, 1] : ... to tensor<2x8xf32>
  %sum = linalg.add ins(%a_slice, %b_slice : ...) outs(...) -> tensor<2x8xf32>
  %relu = linalg.max ins(%sum, %zero_slice : ...) outs(...) -> tensor<2x8xf32>
  // ... insert %relu back into the result ...
}
```

See `examples/tiled_fused_relu_add.mlir` and
`test/Lowering/tiled-fused-relu-add-fusion.mlir`.
