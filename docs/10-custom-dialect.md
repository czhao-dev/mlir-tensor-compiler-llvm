# A small custom dialect: ttensor

`ttensor` is a teaching dialect defined in `include/TTensor/` with exactly
one op, `ttensor.relu_add`, which computes `max(lhs + rhs, 0)` on two
equally-shaped tensors as a single high-level op:

```mlir
%result = ttensor.relu_add %A, %B : tensor<4x4xf32>
```

It exists to demonstrate MLIR's dialect/ODS system end to end -- TableGen
op definitions, generated parser/printer/verifier boilerplate, dialect
registration, and a lowering pass -- not to add a new abstraction the rest
of the pipeline depends on. `-convert-ttensor-to-linalg` expands it into
exactly the same sequence `fused_relu_add.mlir` writes by hand:

```mlir
%sum_init = tensor.empty() : tensor<4x4xf32>
%sum = linalg.add ins(%A, %B : ...) outs(%sum_init : ...) -> tensor<4x4xf32>
%zero_init = tensor.empty() : tensor<4x4xf32>
%cst = arith.constant 0.0 : f32
%zeroed = linalg.fill ins(%cst : f32) outs(%zero_init : ...) -> tensor<4x4xf32>
%relu_init = tensor.empty() : tensor<4x4xf32>
%relu = linalg.max ins(%sum, %zeroed : ...) outs(%relu_init : ...) -> tensor<4x4xf32>
```

so every later pipeline stage only ever sees standard upstream dialects --
`-convert-ttensor-to-linalg` just needs to run first, before bufferization,
the same way `-transform-interpreter` does for the [tiled/vectorized
examples](07-tiling.md).

## Layout

```text
include/TTensor/TTensorOps.td   # Dialect + op definition (ODS/TableGen)
include/TTensor/TTensorOps.h    # Hand-written header pulling in the generated .inc files
include/TTensor/Passes.h        # Pass creation/registration declarations
include/TTensor/CMakeLists.txt  # add_mlir_dialect(TTensorOps ttensor)
lib/TTensor/TTensorDialect.cpp  # Dialect registration + generated op definitions
lib/TTensor/ConvertTTensorToLinalg.cpp  # The -convert-ttensor-to-linalg pass
lib/TTensor/CMakeLists.txt      # add_mlir_dialect_library(MLIRTTensor ...)
```

`add_mlir_dialect(TTensorOps ttensor)` runs `mlir-tblgen` against
`TTensorOps.td` to generate the dialect and op C++ boilerplate (decls and
defs, for both ops and any typedefs) into `build/include/TTensor/`.
`add_mlir_dialect_library` registers the resulting library in MLIR's global
`MLIR_DIALECT_LIBS` property, which `tools/tensor-pipeline-opt`'s
`CMakeLists.txt` already collects automatically -- the only place the
driver itself needs to change is `tensor-pipeline-opt.cpp`, which calls
`registry.insert<mlir::ttensor::TTensorDialect>()` and
`mlir::ttensor::registerConvertTTensorToLinalgPass()` next to the existing
`registerAllDialects`/`registerAllExtensions`/`registerAllPasses` calls.

One non-obvious requirement: `ConvertTTensorToLinalgPass` creates `tensor`,
`linalg`, and `arith` ops, but the input IR it runs on may only have
`ttensor` and `func` loaded into the context. Without overriding
`getDependentDialects()` to declare those three as dependencies, the pass
crashes the first time it tries to build an op from an unloaded dialect --
this is a general MLIR pass-authoring rule, not specific to this dialect.

See `examples/ttensor_relu_add.mlir`, `test/Driver/load-ttensor-ir.mlir`,
`test/Lowering/ttensor-relu-add-to-linalg.mlir`, and
`test/Lowering/ttensor-relu-add-to-llvm.mlir`.
