# Stage 4: affine and control flow

Two passes bridge the gap between structured loops and the unstructured
control flow LLVM understands:

- `-lower-affine` rewrites any remaining `affine.apply`/`affine.for`-style
  index arithmetic into plain `arith` ops on `index` values. Tiling
  ([stage 7](07-tiling.md)) and `-expand-strided-metadata` both introduce
  `affine.apply` for offset/stride computations, so this pass has to run
  after them.
- `-convert-scf-to-cf` rewrites `scf.for`/`scf.if` into explicit basic blocks
  and branches (`cf.br`, `cf.cond_br`) -- the loop's induction variable
  becomes a block argument, and the loop condition becomes a conditional
  branch back to the loop header or out to the exit block.

```mlir
^bb1(%i: index):
  %cond = arith.cmpi slt, %i, %c4 : index
  cf.cond_br %cond, ^bb2, ^bb3
^bb2:
  // ... loop body ...
  %next = arith.addi %i, %c1 : index
  cf.br ^bb1(%next : index)
^bb3:
  // ... after the loop ...
```

After this stage there's no more "structure" left in the IR -- just blocks
and branches, which is exactly the shape LLVM IR expects. See any of the
`*-to-llvm.mlir` tests under `test/Lowering/`, which all run these two
passes en route to the `llvm` dialect.
