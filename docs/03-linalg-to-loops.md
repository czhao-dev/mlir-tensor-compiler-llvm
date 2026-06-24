# Stage 3: linalg to loops

`-convert-linalg-to-loops` expands each `linalg` op on memrefs into explicit
`scf.for` loops with scalar `memref.load`/`arith.*`/`memref.store` in the
body. This is where the actual iteration structure of an op becomes visible:
elementwise ops become one loop nest, `linalg.matmul` becomes three nested
loops with a multiply-accumulate, `linalg.reduce` becomes a fill loop plus an
accumulation loop, and so on.

```mlir
scf.for %i = %c0 to %c4 step %c1 {
  scf.for %j = %c0 to %c4 step %c1 {
    %a = memref.load %A[%i, %j] : memref<4x4xf32>
    %b = memref.load %B[%i, %j] : memref<4x4xf32>
    %sum = arith.addf %a, %b : f32
    memref.store %sum, %out[%i, %j] : memref<4x4xf32>
  }
}
```

This is the stage where you can see *what an op actually computes*, loop by
loop -- it's the most useful stage to inspect by hand when debugging a new
example. See `test/Lowering/elementwise-add-to-loops.mlir`,
`matmul-to-loops.mlir`, `conv2d-to-loops.mlir`, and
`reduce-rows-to-loops.mlir`.

[Tiling](07-tiling.md) rewrites the *tensor-level* op before this stage runs
(splitting one `linalg.matmul` into several smaller ones over slices), so
the loop nest this stage emits for a tiled example has tile-sized bounds
instead of the full tensor shape.
