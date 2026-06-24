# Stage 2: bufferization

`-one-shot-bufferize="bufferize-function-boundaries"` turns `tensor` values
into `memref` values -- i.e. it decides where every tensor actually lives in
memory. `tensor.empty()` becomes `memref.alloc()`; function arguments and
results become memrefs too (that's what `bufferize-function-boundaries`
controls).

Before:
```mlir
%init = tensor.empty() : tensor<4x4xf32>
%sum = linalg.add ins(%A, %B : tensor<4x4xf32>, tensor<4x4xf32>) outs(%init : tensor<4x4xf32>) -> tensor<4x4xf32>
return %sum : tensor<4x4xf32>
```

After:
```mlir
%alloc = memref.alloc() {alignment = 64 : i64} : memref<4x4xf32>
linalg.add ins(%A, %B : memref<4x4xf32>, memref<4x4xf32>) outs(%alloc : memref<4x4xf32>)
return %alloc : memref<4x4xf32>
```

Without this stage, `linalg-to-loops` ([stage 3](03-linalg-to-loops.md)) has
nothing to generate `memref.load`/`memref.store` against -- loop bodies need
real memory, not abstract tensor values. See
`test/Lowering/bufferization.mlir`.
