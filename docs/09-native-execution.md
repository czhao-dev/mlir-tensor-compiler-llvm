# Native execution vs. JIT

Every example can be run two ways, both starting from the same llvm-dialect
IR that `tensor-pipeline-opt` produces:

- **JIT** (`scripts/run_jit_example.sh`): pipe the llvm-dialect IR straight
  into `mlir-runner`, which JIT-compiles it in memory and calls `@main`
  immediately. Nothing is written to disk.
- **Native** (`scripts/build_native_example.sh`): translate the same IR to
  LLVM IR ahead of time, compile it to a real object file, and link a
  standalone executable -- the same path a production compiler would take.

```text
tensor-pipeline-opt ...        # same for both
        |
        v
   llvm dialect IR
    /          \
   v            v
mlir-runner   mlir-translate --mlir-to-llvmir
(JIT, runs       |
 immediately)    v
              LLVM IR (.ll)
                 |
                 v
         llc -filetype=obj
                 |
                 v
            object file (.o)
                 |
                 v
   clang -lmlir_runner_utils -lmlir_c_runner_utils
                 |
                 v
          native executable
```

The native path links against the same `libmlir_runner_utils`/
`libmlir_c_runner_utils` shared libraries `mlir-runner` loads for JIT
execution -- `@printMemrefF32` is the same function either way, just bound
ahead of time by the linker instead of at JIT load time.

One quirk worth knowing: every example's `@main` returns `void` in MLIR
(that's what `mlir-runner -entry-point-result=void` expects), but the
native path links it directly as the process's C `main` symbol, which is
supposed to return an `int`. The resulting exit code is therefore whatever
happened to be left in the return register -- not a real success/failure
signal. `build_native_example.sh` deliberately doesn't propagate it;
correctness is judged by the printed memref output, same as the JIT path.

Run it with:

```sh
scripts/build_native_example.sh examples/matmul_main.mlir
PRE_PASSES="-transform-interpreter" scripts/build_native_example.sh examples/tiled_matmul_main.mlir
```
