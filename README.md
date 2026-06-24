# Tensor Compiler Mini-Pipeline

A small MLIR-based compiler project for learning how high-level tensor programs become lower-level executable code. The pipeline starts with tensor operations expressed in the `linalg`/`tensor` dialects, optionally rewrites them with the `transform` dialect (tiling, fusion, vectorization), and lowers them through MLIR's standard dialects — `linalg`, `bufferization`, `scf`, `affine`, `vector`, `memref`, `arith`, `cf`, and finally `llvm` — down to code that can be JIT-executed or compiled into a standalone native executable.

This project is intentionally scoped as a learning compiler rather than a production framework. Each stage is meant to be easy to understand, test, and debug by hand.

## Status

Milestone 1 is complete: elementwise tensor addition, matrix multiplication, 2D convolution, and a fused `relu(add(A, B))` example all lower end to end from tensor-level IR to the `llvm` dialect and execute correctly under MLIR's JIT runner.

Milestone 2 adds tiling, loop fusion, vectorization, a reduction example, per-stage documentation, and a native (non-JIT) execution path:

- `examples/tiled_matmul.mlir` — tiles a `linalg.matmul` via the transform dialect (`transform.structured.tile_using_for`).
- `examples/tiled_fused_relu_add.mlir` — tiles `relu(A + B)` and fuses the producer into the tiled loop (`transform.structured.fuse_into_containing_op`), a different flavor of fusion from the op-level fusion below.
- `examples/vectorized_matmul.mlir` — vectorizes a whole matmul into a single `vector.contract` (`transform.structured.vectorize_children_and_apply_patterns`).
- `examples/reduce_rows.mlir` — reduction over one dimension (`linalg.reduce`).
- `docs/` — one short doc per pipeline stage, including the non-obvious pass-ordering constraints tiling and vectorization introduce.
- `scripts/build_native_example.sh` — builds a standalone native executable (object file + link) instead of JIT-executing, as an alternative to `scripts/run_jit_example.sh`.

8 examples, 15 lit tests, all passing. See [Test Results](#test-results) below for verified output.

## Goals

- Represent tensor computations in a high-level MLIR dialect (`tensor` + `linalg`).
- Lower tensor operations through standard MLIR dialects.
- Provide a custom compiler driver (`tensor-pipeline-opt`) that accepts `.mlir` input and runs a configurable pass pipeline, mirroring `mlir-opt`.
- Generate runnable code through the LLVM lowering path and execute it via MLIR's JIT runner.
- Add simple optimization passes such as tiling, fusion, and vectorization.
- Keep examples small enough that every pass can be inspected by hand.

## Pipeline

The current pipeline, exercised end to end by the examples and tests in this repo:

```text
tensor + linalg input
  -> [optional] transform-interpreter (tiling / fusion / vectorization)
  -> linalg on tensors
  -> one-shot-bufferize (tensor -> memref)
  -> convert-linalg-to-loops (linalg -> scf.for)
  -> expand-strided-metadata, convert-vector-to-scf
  -> lower-affine, convert-scf-to-cf
  -> convert-{cf,vector,arith,ub,memref,func}-to-llvm
  -> reconcile-unrealized-casts, symbol-dce
  -> llvm dialect
  -> JIT execution via mlir-runner, or a native executable
```

Each stage is documented in [`docs/`](docs/), including why some of these passes have to run in this specific order (e.g. vector-to-llvm before arith-to-llvm).

## Example Programs

Implemented:

- `examples/elementwise_add.mlir` / `elementwise_add_main.mlir` — elementwise tensor addition (`linalg.add`).
- `examples/matmul.mlir` / `matmul_main.mlir` — matrix multiplication (`linalg.fill` + `linalg.matmul`).
- `examples/conv2d.mlir` / `conv2d_main.mlir` — 2D convolution, NHWC input / HWCF filter (`linalg.conv_2d_nhwc_hwcf`).
- `examples/fused_relu_add.mlir` / `fused_relu_add_main.mlir` — `relu(A + B)` (`linalg.add` + `linalg.max`); also demonstrates `-linalg-fuse-elementwise-ops` fusing both into a single `linalg.generic`.
- `examples/tiled_matmul.mlir` / `tiled_matmul_main.mlir` — the same matmul, tiled into a 2x2 grid of 2x2-tile matmuls via the transform dialect (`transform.structured.tile_using_for`).
- `examples/tiled_fused_relu_add.mlir` / `tiled_fused_relu_add_main.mlir` — `relu(A + B)` again, but tiled with `linalg.add` fused into the tiled loop (`transform.structured.fuse_into_containing_op`) — loop-level fusion, as opposed to the op-level fusion in `fused_relu_add.mlir`.
- `examples/vectorized_matmul.mlir` / `vectorized_matmul_main.mlir` — the matmul vectorized into a single `vector.contract` via the transform dialect (`transform.structured.vectorize_children_and_apply_patterns`).
- `examples/reduce_rows.mlir` / `reduce_rows_main.mlir` — reduction over one dimension: sums each row of a tensor (`linalg.reduce`).

Each example has two files:

- The plain version takes tensors as function arguments and returns a tensor — used by the FileCheck lowering tests.
- The `_main` version embeds constant input tensors and prints the result — used for JIT execution via `scripts/run_jit_example.sh` or native execution via `scripts/build_native_example.sh`.

The three transform-dialect examples (`tiled_matmul`, `tiled_fused_relu_add`, `vectorized_matmul`) embed their schedule as a `transform.named_sequence` in the same file; running them needs `-transform-interpreter` passed first (see [Running the Compiler](#running-the-compiler)).

## Repository Layout

```text
.
├── tools/tensor-pipeline-opt/       # The compiler driver (mlir-opt-style tool)
├── examples/                        # Tensor-level input programs (plain + JIT-runnable)
├── test/                            # lit + FileCheck tests for the driver and pass pipeline
│   ├── Driver/                      # Parsing/round-trip tests
│   └── Lowering/                    # Per-stage and end-to-end lowering tests
├── docs/                            # One short doc per pipeline stage
├── scripts/run_jit_example.sh       # Lowers and JIT-executes a `_main` example
├── scripts/build_native_example.sh  # Lowers and builds a `_main` example into a native executable
├── CMakeLists.txt                   # Build configuration
└── README.md
```

A custom tensor dialect is not part of the current scope (see [Roadmap](#roadmap)); the project builds entirely on upstream MLIR dialects.

## Development Setup

This project builds against an LLVM/MLIR installation that includes MLIR's CMake package files, headers, and libraries.

### Option A: Homebrew (macOS, what this repo was developed and tested against)

```sh
brew install llvm
```

Homebrew's `llvm` formula ships full MLIR dev files (`lib/cmake/mlir`, headers, and static libs), so no separate MLIR build is required.

### Option B: Build LLVM/MLIR from source

```sh
git clone https://github.com/llvm/llvm-project.git
cmake -S llvm-project/llvm -B llvm-build \
  -G Ninja \
  -DLLVM_ENABLE_PROJECTS=mlir \
  -DLLVM_BUILD_EXAMPLES=ON \
  -DLLVM_TARGETS_TO_BUILD="host" \
  -DCMAKE_BUILD_TYPE=Release
cmake --build llvm-build
```

### Configuring and building this repo

```sh
cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DMLIR_DIR=$(brew --prefix llvm)/lib/cmake/mlir \
  -DLLVM_DIR=$(brew --prefix llvm)/lib/cmake/llvm
cmake --build build
```

(If you built LLVM/MLIR from source instead, point `MLIR_DIR`/`LLVM_DIR` at `llvm-build/lib/cmake/mlir` and `llvm-build/lib/cmake/llvm`.)

This produces `build/tools/tensor-pipeline-opt/tensor-pipeline-opt`.

## Running the Compiler

`tensor-pipeline-opt` accepts `.mlir` input and a sequence of pass flags, just like `mlir-opt`:

```sh
build/tools/tensor-pipeline-opt/tensor-pipeline-opt examples/elementwise_add.mlir \
  -one-shot-bufferize="bufferize-function-boundaries" \
  -convert-linalg-to-loops \
  -expand-strided-metadata \
  -convert-vector-to-scf \
  -lower-affine \
  -convert-scf-to-cf \
  -convert-cf-to-llvm \
  -convert-vector-to-llvm \
  -convert-arith-to-llvm \
  -convert-ub-to-llvm \
  -finalize-memref-to-llvm \
  -convert-func-to-llvm \
  -reconcile-unrealized-casts \
  -symbol-dce
```

This prints the fully lowered `llvm` dialect IR for elementwise add. The same pipeline works for every other example. `-expand-strided-metadata`, `-convert-vector-to-scf`, and `-convert-vector-to-llvm` only do real work for the tiled/vectorized examples (see [docs/05-llvm-conversion.md](docs/05-llvm-conversion.md)) but are harmless no-ops otherwise, so the same command line works everywhere.

The three transform-dialect examples additionally need `-transform-interpreter` run *first*, before bufferization:

```sh
build/tools/tensor-pipeline-opt/tensor-pipeline-opt examples/tiled_matmul.mlir -transform-interpreter \
  -one-shot-bufferize="bufferize-function-boundaries" \
  ... # same pipeline as above
```

Useful development modes (inherited from `mlir-opt`'s `MlirOptMain`):

- `--mlir-print-ir-after-all` — print the IR after each pass.
- Run a single pass at a time by passing only that flag.
- `--verify-each` (on by default) — verify IR after every transformation.
- `--mlir-pass-statistics` — print per-pass statistics.

### JIT execution

`scripts/run_jit_example.sh` lowers a `_main` example to the `llvm` dialect and executes it with `mlir-runner`:

```sh
scripts/run_jit_example.sh examples/elementwise_add_main.mlir
scripts/run_jit_example.sh examples/matmul_main.mlir
scripts/run_jit_example.sh examples/conv2d_main.mlir
scripts/run_jit_example.sh examples/fused_relu_add_main.mlir
scripts/run_jit_example.sh examples/reduce_rows_main.mlir

# transform-dialect examples need their schedule applied before bufferization:
PRE_PASSES="-transform-interpreter" scripts/run_jit_example.sh examples/tiled_matmul_main.mlir
PRE_PASSES="-transform-interpreter" scripts/run_jit_example.sh examples/tiled_fused_relu_add_main.mlir
PRE_PASSES="-transform-interpreter" scripts/run_jit_example.sh examples/vectorized_matmul_main.mlir
```

It locates `mlir-runner` and the MLIR runner-utils shared libraries via `brew --prefix llvm`; set `LLVM_PREFIX` to override.

### Native execution

`scripts/build_native_example.sh` takes the same `_main` examples through an ahead-of-time path instead: `mlir-translate` to LLVM IR, `llc` to an object file, then `clang` to link a standalone executable against the same runner-utils libraries `mlir-runner` loads for JIT. It takes the same arguments and `PRE_PASSES` convention as `run_jit_example.sh`:

```sh
scripts/build_native_example.sh examples/matmul_main.mlir
PRE_PASSES="-transform-interpreter" scripts/build_native_example.sh examples/tiled_matmul_main.mlir
```

The built executables and intermediate `.ll`/`.o` files land under `build/native/<example-name>/`. See [docs/09-native-execution.md](docs/09-native-execution.md) for how this compares to the JIT path.

## Testing

The test suite uses MLIR's `lit` and `FileCheck`. Each test focuses on one stage of the pipeline:

- `test/Driver/load-tensor-ir.mlir` — the driver loads and round-trips tensor/linalg input unchanged.
- `test/Lowering/bufferization.mlir` — `one-shot-bufferize` turns tensor arguments into memrefs.
- `test/Lowering/elementwise-add-to-loops.mlir` — `linalg.add` lowers to a nested `scf.for` loop with scalar `arith.addf`.
- `test/Lowering/matmul-to-loops.mlir` — `linalg.fill` + `linalg.matmul` lower to a zero-fill loop nest plus a triply-nested `scf.for` loop.
- `test/Lowering/conv2d-to-loops.mlir` — `linalg.conv_2d_nhwc_hwcf` lowers to nested `scf.for` loops with a multiply-accumulate per output element.
- `test/Lowering/fused-relu-add-fusion.mlir` — `-linalg-fuse-elementwise-ops` fuses `linalg.add` + `linalg.max` into one `linalg.generic`.
- `test/Lowering/elementwise-add-to-llvm.mlir` — the full tensor-to-`llvm` pipeline, asserting no `tensor`/`linalg`/`scf` ops remain.
- `test/Lowering/fused-relu-add-to-llvm.mlir` — same full pipeline for the fused relu(add) example, asserting an `llvm.intr.maximum` remains where the relu clamp lives.
- `test/Lowering/reduce-rows-to-loops.mlir` — `linalg.reduce` lowers to a fill loop plus a doubly-nested accumulation loop with `arith.addf`.
- `test/Lowering/reduce-rows-to-llvm.mlir` — the full pipeline for the row reduction.
- `test/Lowering/tiled-matmul-tiling.mlir` — `transform.structured.tile_using_for` tiles the matmul into nested `scf.for` loops over tile-sized `linalg.matmul`s.
- `test/Lowering/tiled-matmul-to-llvm.mlir` — the full pipeline (transform-interpreter + lowering) for the tiled matmul.
- `test/Lowering/tiled-fused-relu-add-fusion.mlir` — tiling `linalg.max` and fusing `linalg.add` into it produces a single `scf.for` containing both ops.
- `test/Lowering/vectorized-matmul-vectorization.mlir` — `transform.structured.vectorize_children_and_apply_patterns` turns the matmul into a single `vector.contract`.
- `test/Lowering/vectorized-matmul-to-llvm.mlir` — the full pipeline (transform-interpreter + lowering) for the vectorized matmul.

### Running the tests

```sh
pip install lit   # one-time; FileCheck ships with LLVM/MLIR
cmake -S . -B build -G Ninja \
  -DMLIR_DIR=$(brew --prefix llvm)/lib/cmake/mlir \
  -DLLVM_DIR=$(brew --prefix llvm)/lib/cmake/llvm \
  -DLIT_EXECUTABLE=$(command -v lit)
cmake --build build --target check
```

### Test Results

Verified on macOS (arm64) with Homebrew LLVM/MLIR 22.1.6, 2026-06-23:

```text
-- Testing: 15 tests, 8 workers --
PASS: tensor-pipeline :: Lowering/bufferization.mlir
PASS: tensor-pipeline :: Lowering/conv2d-to-loops.mlir
PASS: tensor-pipeline :: Lowering/fused-relu-add-fusion.mlir
PASS: tensor-pipeline :: Lowering/elementwise-add-to-loops.mlir
PASS: tensor-pipeline :: Lowering/matmul-to-loops.mlir
PASS: tensor-pipeline :: Driver/load-tensor-ir.mlir
PASS: tensor-pipeline :: Lowering/elementwise-add-to-llvm.mlir
PASS: tensor-pipeline :: Lowering/fused-relu-add-to-llvm.mlir
PASS: tensor-pipeline :: Lowering/reduce-rows-to-llvm.mlir
PASS: tensor-pipeline :: Lowering/reduce-rows-to-loops.mlir
PASS: tensor-pipeline :: Lowering/tiled-matmul-tiling.mlir
PASS: tensor-pipeline :: Lowering/tiled-fused-relu-add-fusion.mlir
PASS: tensor-pipeline :: Lowering/vectorized-matmul-vectorization.mlir
PASS: tensor-pipeline :: Lowering/tiled-matmul-to-llvm.mlir
PASS: tensor-pipeline :: Lowering/vectorized-matmul-to-llvm.mlir

Testing Time: 0.43s
Total Discovered Tests: 15
  Passed: 15 (100.00%)
```

JIT execution of all eight examples produces the mathematically correct result:

```text
$ scripts/run_jit_example.sh examples/elementwise_add_main.mlir
Unranked Memref base@ = 0x... rank = 2 offset = 0 sizes = [2, 2] strides = [2, 1] data =
[[6,    8],
 [10,   12]]

$ scripts/run_jit_example.sh examples/matmul_main.mlir
Unranked Memref base@ = 0x... rank = 2 offset = 0 sizes = [2, 2] strides = [2, 1] data =
[[19,   22],
 [43,   50]]

$ scripts/run_jit_example.sh examples/conv2d_main.mlir
Unranked Memref base@ = 0x... rank = 4 offset = 0 sizes = [1, 3, 3, 1] strides = [9, 3, 1, 1] data =
[[[[63],
   [72],
   [81]],
  [[108],
   [117],
   [126]],
  [[153],
   [162],
   [171]]]]

$ scripts/run_jit_example.sh examples/fused_relu_add_main.mlir
Unranked Memref base@ = 0x... rank = 2 offset = 0 sizes = [2, 2] strides = [2, 1] data =
[[3,   0],
 [0,   3]]

$ scripts/run_jit_example.sh examples/reduce_rows_main.mlir
Unranked Memref base@ = 0x... rank = 1 offset = 0 sizes = [2] strides = [1] data =
[6,   15]

$ PRE_PASSES="-transform-interpreter" scripts/run_jit_example.sh examples/tiled_matmul_main.mlir
Unranked Memref base@ = 0x... rank = 2 offset = 0 sizes = [4, 4] strides = [4, 1] data =
[[1,    2,    3,    4],
 [5,    6,    7,    8],
 [9,    10,   11,   12],
 [13,   14,   15,   16]]

$ PRE_PASSES="-transform-interpreter" scripts/run_jit_example.sh examples/tiled_fused_relu_add_main.mlir
Unranked Memref base@ = 0x... rank = 2 offset = 0 sizes = [8, 8] strides = [8, 1] data =
[[0,    0,    0,    0,    0,    0,    0,    0],
 [0,    0,    0,    0,    0,    0,    0,    0],
 [0,    0,    0,    0,    0,    0,    0,    0],
 [0,    0,    0,    0,    0,    0,    0,    0],
 [12,   12,   12,   12,   12,   12,   12,   12],
 [12,   12,   12,   12,   12,   12,   12,   12],
 [12,   12,   12,   12,   12,   12,   12,   12],
 [12,   12,   12,   12,   12,   12,   12,   12]]

$ PRE_PASSES="-transform-interpreter" scripts/run_jit_example.sh examples/vectorized_matmul_main.mlir
Unranked Memref base@ = 0x... rank = 2 offset = 0 sizes = [2, 2] strides = [2, 1] data =
[[19,   22],
 [43,   50]]
```

All confirmed correct by hand:

- `A = [[1,2],[3,4]]`, `B = [[5,6],[7,8]]`: `A+B = [[6,8],[10,12]]`, `A x B = [[19,22],[43,50]]` (also the expected result for `vectorized_matmul`, which uses the same inputs).
- `conv2d`: a 5x5 input with values `1..25` (row-major) convolved with a 3x3 all-ones filter is a box-sum filter; the 3x3 output is the sum of each 3x3 window, matching the hand-computed values above.
- `fused_relu_add`: `A = [[1,-5],[3,-2]]`, `B = [[2,1],[-10,5]]`, so `A+B = [[3,-4],[-7,3]]` and `relu(A+B) = [[3,0],[0,3]]`.
- `reduce_rows`: `A = [[1,2,3],[4,5,6]]`, row sums `[1+2+3, 4+5+6] = [6, 15]`.
- `tiled_matmul`: `A x I = A` for the 4x4 identity `I`, so the tiled-and-lowered result should equal `A = [[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]]` unchanged.
- `tiled_fused_relu_add`: `A` is `2.0` everywhere; `B` is `-10.0` in its top half and `+10.0` in its bottom half, so `relu(A+B)` is `0` in the top half (`max(2-10, 0)`) and `12` in the bottom half (`max(2+10, 0)`).

Native execution of every example via `scripts/build_native_example.sh` was also verified to produce byte-for-byte identical output to the JIT path above (including the four new examples, with `PRE_PASSES="-transform-interpreter"` for the transform-dialect ones).

## Roadmap

1. ~~Create the CMake project skeleton.~~
2. ~~Add a minimal compiler driver.~~
3. ~~Accept `.mlir` input files and run a configurable pass pipeline.~~
4. ~~Lower one elementwise tensor example end to end.~~
5. ~~Add matrix multiplication.~~
6. ~~Add pass tests with `lit` and `FileCheck`.~~
7. ~~Add more examples: 2D convolution and a fused `relu(add(A, B))` op.~~
8. ~~Implement tiling and fusion experiments~~ — `tiled_matmul.mlir` (tiling) and `tiled_fused_relu_add.mlir` (tiling + producer/consumer loop fusion), via the transform dialect.
9. ~~Add vectorization for small static shapes~~ — `vectorized_matmul.mlir`.
10. ~~Add documentation for each lowering stage~~ — see [`docs/`](docs/).
11. Optionally add a small custom tensor dialect.
12. ~~Add a path to a native executable (object file + linking), not just JIT execution~~ — `scripts/build_native_example.sh`.

## Design Principles

- Prefer clarity over cleverness.
- Keep each pass independently testable.
- Make intermediate IR easy to inspect.
- Start with static shapes before supporting dynamic shapes.
- Reuse standard MLIR dialects before inventing custom abstractions.
- Document surprising lowering behavior with small examples.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
