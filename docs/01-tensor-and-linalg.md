# Stage 1: tensor + linalg

Every example starts as a `func.func` operating on `tensor` values, with the
actual computation expressed via `linalg` named ops (`linalg.add`,
`linalg.matmul`, `linalg.conv_2d_nhwc_hwcf`, `linalg.reduce`, ...) or
`linalg.generic`. Tensors are immutable SSA values: each op produces a brand
new tensor rather than mutating one in place, and shapes are static
throughout this project (see [Design Principles](../README.md#design-principles)).

```mlir
func.func @elementwise_add(%A: tensor<4x4xf32>, %B: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %init = tensor.empty() : tensor<4x4xf32>
  %sum = linalg.add ins(%A, %B : tensor<4x4xf32>, tensor<4x4xf32>) outs(%init : tensor<4x4xf32>) -> tensor<4x4xf32>
  return %sum : tensor<4x4xf32>
}
```

`tensor.empty()` declares the shape of the result without allocating
anything real -- there's no memory yet, only a placeholder that
[bufferization](02-bufferization.md) later turns into an actual allocation.

This representation is what every other stage in the pipeline lowers away.
See `examples/elementwise_add.mlir` and `test/Driver/load-tensor-ir.mlir`
(which just checks the driver round-trips this input unchanged).
