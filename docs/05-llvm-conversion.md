# Stage 5: conversion to the llvm dialect

The final stretch is a sequence of per-dialect "convert X to llvm" passes,
each rewriting one remaining dialect's ops into their `llvm` dialect
equivalent (`cf.br` -> `llvm.br`, `arith.addf` -> `llvm.fadd`,
`memref.alloc` -> `llvm.call @malloc` + struct bookkeeping, etc.):

```text
-convert-cf-to-llvm
-convert-vector-to-llvm
-convert-arith-to-llvm
-convert-ub-to-llvm
-finalize-memref-to-llvm
-convert-func-to-llvm
-reconcile-unrealized-casts
-symbol-dce
```

The order of `-convert-vector-to-llvm` and `-convert-arith-to-llvm` matters
and is easy to get backwards: a multi-dimensional `vector<NxMxf32>`
`arith.constant` (which [vectorization](08-vectorization.md) produces) can
only be lowered by the vector-to-llvm patterns, which know how to decompose
it into the `!llvm.array<N x vector<Mxf32>>` shape LLVM actually supports.
Running `-convert-arith-to-llvm` first leaves it half-converted and stuck.
`-convert-ub-to-llvm` exists for the same reason: vectorization introduces
`ub.poison` (the out-of-bounds padding value for `vector.transfer_read`),
which needs its own pass to become `llvm.mlir.poison`.

`-reconcile-unrealized-casts` cleans up the `builtin.unrealized_conversion_cast`
ops that dialect conversion leaves behind when one pattern produces a type
another pattern hasn't converted yet -- by the end of the pipeline these
should all cancel out. `-symbol-dce` is last because [transform dialect
schedules](07-tiling.md) (tiling/fusion/vectorization) leave their
`transform.named_sequence` behind as dead code once interpreted; marking it
`private` and running `-symbol-dce` removes it.

After this stage, `--mlir-print-ir-after-all` should show zero `tensor.`,
`linalg.`, `scf.`, or `memref.` ops left -- every `*-to-llvm.mlir` test under
`test/Lowering/` checks exactly that.
