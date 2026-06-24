# Vectorization: `transform.structured.vectorize_children_and_apply_patterns`

For small, fully static shapes, an entire `linalg` op can become a single
SIMD operation instead of a scalar loop nest. `vectorized_matmul.mlir`
vectorizes a whole function in one shot:

```mlir
%func = transform.structured.match ops{["func.func"]} in %module : (!transform.any_op) -> !transform.any_op
%vectorized = transform.structured.vectorize_children_and_apply_patterns %func : (!transform.any_op) -> !transform.any_op
```

The 4x4 `linalg.matmul` becomes:

```mlir
%a = vector.transfer_read %A[%c0, %c0], %pad : tensor<4x4xf32>, vector<4x4xf32>
%b = vector.transfer_read %B[%c0, %c0], %pad : tensor<4x4xf32>, vector<4x4xf32>
%result = vector.contract {indexing_maps = [...], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>}
  %a, %b, %zero : vector<4x4xf32>, vector<4x4xf32> into vector<4x4xf32>
%out = vector.transfer_write %result, %init[%c0, %c0] : vector<4x4xf32>, tensor<4x4xf32>
```

No `scf.for` at all -- the whole computation is one `vector.contract`. Once
lowered further, `vector.contract` becomes a sequence of `llvm.fmul` +
`llvm.intr.vector.reduce.fadd` over 1-D LLVM vectors (LLVM has no native
multi-dimensional vector type, so anything beyond 1-D gets decomposed into
an `!llvm.array` of 1-D vectors along the way).

**Pipeline requirement**: getting from `vector.contract` down to the `llvm`
dialect needs two extra passes beyond the ones in [stage 5](05-llvm-conversion.md):
`-convert-vector-to-scf` (to unroll any remaining multi-dimensional
transfer ops into rank-1 transfers) and `-convert-vector-to-llvm` itself --
and the latter must run *before* `-convert-arith-to-llvm`, since a
multi-dimensional `arith.constant` (e.g. the zero accumulator) can only be
decomposed by the vector-to-llvm patterns. See [stage 5](05-llvm-conversion.md)
for the full ordering rationale.

See `examples/vectorized_matmul.mlir`,
`test/Lowering/vectorized-matmul-vectorization.mlir`, and
`test/Lowering/vectorized-matmul-to-llvm.mlir`.
