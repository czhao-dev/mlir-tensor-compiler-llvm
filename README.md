# Tensor Compiler Mini-Pipeline

A small MLIR-based compiler project for learning how high-level tensor programs become lower-level executable code. The pipeline starts with tensor operations expressed in the `linalg`/`tensor` dialects (or the small custom `ttensor` dialect, see below), optionally rewrites them with the `transform` dialect (tiling, fusion, vectorization), and lowers them through MLIR's standard dialects — `linalg`, `bufferization`, `scf`, `affine`, `vector`, `memref`, `arith`, `cf`, and finally `llvm` — down to code that can be JIT-executed or compiled into a standalone native executable.

This project is intentionally scoped as a learning compiler rather than a production framework. Each stage is meant to be easy to understand, test, and debug by hand.

---

## Table of Contents

- [Goals](#goals)
- [Quick Start](#quick-start)
- [Pipeline](#pipeline)
- [Example Programs](#example-programs)
- [Repository Layout](#repository-layout)
- [Documentation](#documentation)
- [Development Setup](#development-setup)
- [Running the Compiler](#running-the-compiler)
- [Testing](#testing)
- [Design Principles](#design-principles)
- [License](#license)

---

## Goals

- Represent tensor computations in a high-level MLIR dialect (`tensor` + `linalg`).
- Lower tensor operations through standard MLIR dialects.
- Provide a custom compiler driver (`tensor-pipeline-opt`) that accepts `.mlir` input and runs a configurable pass pipeline, mirroring `mlir-opt`.
- Generate runnable code through the LLVM lowering path and execute it via MLIR's JIT runner.
- Add simple optimization passes such as tiling, fusion, and vectorization.
- Keep examples small enough that every pass can be inspected by hand.

---

## Quick Start

```sh
# 1. Install LLVM/MLIR (macOS)
brew install llvm

# 2. Build
cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DMLIR_DIR=$(brew --prefix llvm)/lib/cmake/mlir \
  -DLLVM_DIR=$(brew --prefix llvm)/lib/cmake/llvm
cmake --build build

# 3. Run a test
pip install lit
cmake --build build --target check

# 4. JIT-execute an example
scripts/run_jit_example.sh examples/elementwise_add_main.mlir
```

---

## Pipeline

The current pipeline, exercised end to end by the examples and tests in this repo:

```text
tensor + linalg (or ttensor) input
  -> [optional] convert-ttensor-to-linalg   # custom dialect desugaring
  -> [optional] transform-interpreter        # tiling / fusion / vectorization
  -> linalg on tensors
  -> one-shot-bufferize                      # tensor -> memref
  -> convert-linalg-to-loops                 # linalg -> scf.for
  -> expand-strided-metadata
  -> convert-vector-to-scf
  -> lower-affine
  -> convert-scf-to-cf
  -> convert-{cf,vector,arith,ub,memref,func}-to-llvm
  -> reconcile-unrealized-casts, symbol-dce
  -> llvm dialect
  -> JIT execution via mlir-runner, or a standalone native executable
```

Each stage is documented in [`docs/`](docs/), including why some of these passes must run in a specific order (e.g., `vector-to-llvm` before `arith-to-llvm`).

---

## Example Programs

| Example | Key Op(s) | Notes |
|---|---|---|
| `elementwise_add` | `linalg.add` | Elementwise tensor addition |
| `matmul` | `linalg.fill` + `linalg.matmul` | Matrix multiplication |
| `conv2d` | `linalg.conv_2d_nhwc_hwcf` | 2D convolution, NHWC input / HWCF filter |
| `fused_relu_add` | `linalg.add` + `linalg.max` | `relu(A+B)`; `-linalg-fuse-elementwise-ops` fuses both into one `linalg.generic` |
| `reduce_rows` | `linalg.reduce` | Sums each row; reduction over one dimension |
| `tiled_matmul` | `transform.structured.tile_using_for` | Matmul tiled into a 2×2 grid of 2×2-tile matmuls via the transform dialect |
| `tiled_fused_relu_add` | `transform.structured.fuse_into_containing_op` | `relu(A+B)` tiled with `linalg.add` fused into the tiled loop (loop-level fusion) |
| `vectorized_matmul` | `transform.structured.vectorize_children_and_apply_patterns` | Matmul vectorized into a single `vector.contract` |
| `ttensor_relu_add` | `ttensor.relu_add` | `relu(A+B)` expressed with the custom `ttensor` dialect; `-convert-ttensor-to-linalg` expands it |

Each example lives in two files:

- **Plain version** (e.g., `elementwise_add.mlir`) — takes tensors as function arguments and returns a tensor. Used by the FileCheck lowering tests.
- **`_main` version** (e.g., `elementwise_add_main.mlir`) — embeds constant input tensors and prints the result. Used for JIT/native execution.

The three transform-dialect examples (`tiled_matmul`, `tiled_fused_relu_add`, `vectorized_matmul`) embed their schedule as a `transform.named_sequence` in the same file and require `-transform-interpreter` to be run before bufferization. `ttensor_relu_add` requires `-convert-ttensor-to-linalg` for the same reason.

---

## Repository Layout

```text
.
├── include/TTensor/                 # ttensor dialect ODS (TableGen) + generated-header glue
├── lib/TTensor/                     # ttensor dialect registration + -convert-ttensor-to-linalg pass
├── tools/tensor-pipeline-opt/       # The compiler driver (mlir-opt-style tool)
├── examples/                        # Tensor-level input programs (plain + JIT-runnable)
├── test/
│   ├── Driver/                      # Parsing and round-trip tests
│   └── Lowering/                    # Per-stage and end-to-end lowering tests
├── docs/                            # One short document per pipeline stage
├── scripts/
│   ├── run_jit_example.sh           # Lowers and JIT-executes a _main example
│   └── build_native_example.sh      # Lowers and builds a _main example into a native binary
├── CMakeLists.txt
└── README.md
```

The project builds entirely on upstream MLIR dialects. `ttensor` is a single small custom dialect (one op), kept deliberately minimal and documented in [docs/10-custom-dialect.md](docs/10-custom-dialect.md).

---

## Documentation

Each file in [`docs/`](docs/) covers one pipeline stage:

| Doc | Topic |
|---|---|
| `01-tensor-linalg.md` | High-level tensor/linalg input IR |
| `02-bufferization.md` | `one-shot-bufferize`: tensor → memref |
| `03-linalg-to-loops.md` | `convert-linalg-to-loops`: linalg → `scf.for` |
| `04-affine-scf-cf.md` | `lower-affine`, `convert-scf-to-cf` |
| `05-llvm-conversion.md` | Converting `cf`/`vector`/`arith`/`memref`/`func` to `llvm`; pass ordering |
| `06-fusion.md` | `-linalg-fuse-elementwise-ops` |
| `07-tiling.md` | Transform dialect tiling and loop-level fusion |
| `08-vectorization.md` | Transform dialect vectorization to `vector.contract` |
| `09-native-execution.md` | AOT path: `mlir-translate` → `llc` → `clang` vs. JIT |
| `10-custom-dialect.md` | The `ttensor` dialect: ODS, registration, lowering pass |

---

## Development Setup

### Prerequisites

- **CMake ≥ 3.20** and **Ninja**
- **LLVM/MLIR** with CMake package files, headers, and static libraries
- **Python + `lit`** for running tests (`pip install lit`)
- **`clang`** for the native execution path

### Option A: Homebrew (macOS — what this repo was developed and tested against)

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

If you built LLVM/MLIR from source, point `MLIR_DIR`/`LLVM_DIR` at `llvm-build/lib/cmake/mlir` and `llvm-build/lib/cmake/llvm` instead.

This produces `build/tools/tensor-pipeline-opt/tensor-pipeline-opt`.

---

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

This prints the fully lowered `llvm` dialect IR for elementwise add. The same pipeline works for every other example. `-expand-strided-metadata`, `-convert-vector-to-scf`, and `-convert-vector-to-llvm` are no-ops for the non-vectorized examples but are harmless to include, so the same command line works everywhere (see [docs/05-llvm-conversion.md](docs/05-llvm-conversion.md)).

The three transform-dialect examples need `-transform-interpreter` run *before* bufferization, and `ttensor_relu_add` needs `-convert-ttensor-to-linalg` run first for the same reason:

```sh
# Tiled matmul: apply transform schedule before bufferization
build/tools/tensor-pipeline-opt/tensor-pipeline-opt examples/tiled_matmul.mlir \
  -transform-interpreter \
  -one-shot-bufferize="bufferize-function-boundaries" \
  ... # same pipeline as above

# Custom dialect: desugar to linalg before bufferization
build/tools/tensor-pipeline-opt/tensor-pipeline-opt examples/ttensor_relu_add.mlir \
  -convert-ttensor-to-linalg \
  -one-shot-bufferize="bufferize-function-boundaries" \
  ... # same pipeline as above
```

### Useful development flags

These are inherited from `mlir-opt`'s `MlirOptMain`:

| Flag | Effect |
|---|---|
| `--mlir-print-ir-after-all` | Print the IR after every pass — useful for tracing transformations step by step |
| `--verify-each` | Verify IR after every transformation (on by default) |
| `--mlir-pass-statistics` | Print per-pass statistics (pattern counts, etc.) |
| (run a single pass flag) | Pass only the flag you're interested in to isolate that one stage |

### JIT execution

`scripts/run_jit_example.sh` lowers a `_main` example to the `llvm` dialect and executes it with `mlir-runner`. It locates `mlir-runner` and the MLIR runner-utils shared libraries via `brew --prefix llvm`; set `LLVM_PREFIX` to override.

```sh
# Standard examples
scripts/run_jit_example.sh examples/elementwise_add_main.mlir
scripts/run_jit_example.sh examples/matmul_main.mlir
scripts/run_jit_example.sh examples/conv2d_main.mlir
scripts/run_jit_example.sh examples/fused_relu_add_main.mlir
scripts/run_jit_example.sh examples/reduce_rows_main.mlir

# Transform-dialect examples: apply the schedule before bufferization
PRE_PASSES="-transform-interpreter" scripts/run_jit_example.sh examples/tiled_matmul_main.mlir
PRE_PASSES="-transform-interpreter" scripts/run_jit_example.sh examples/tiled_fused_relu_add_main.mlir
PRE_PASSES="-transform-interpreter" scripts/run_jit_example.sh examples/vectorized_matmul_main.mlir

# Custom-dialect example: desugar to linalg before bufferization
PRE_PASSES="-convert-ttensor-to-linalg" scripts/run_jit_example.sh examples/ttensor_relu_add_main.mlir
```

### Native execution

`scripts/build_native_example.sh` takes the same `_main` examples through an ahead-of-time path: `mlir-translate` to LLVM IR, `llc` to an object file, then `clang` to link a standalone executable against the runner-utils libraries. It takes the same arguments and `PRE_PASSES` convention as `run_jit_example.sh`:

```sh
scripts/build_native_example.sh examples/matmul_main.mlir
PRE_PASSES="-transform-interpreter" scripts/build_native_example.sh examples/tiled_matmul_main.mlir
```

Built executables and intermediate `.ll`/`.o` files land under `build/native/<example-name>/`. See [docs/09-native-execution.md](docs/09-native-execution.md) for how this compares to the JIT path.

---

## Testing

The test suite uses MLIR's `lit` and `FileCheck`. Each test focuses on one stage of the pipeline.

### Test inventory

**Driver tests** — verify the compiler driver loads and round-trips input IR:

| Test | What it checks |
|---|---|
| `Driver/load-tensor-ir.mlir` | Driver loads and round-trips `tensor`/`linalg` input unchanged |
| `Driver/load-ttensor-ir.mlir` | Driver loads and round-trips the custom `ttensor` dialect unchanged |

**Lowering tests** — verify each pass or pass sequence produces the expected IR:

| Test | What it checks |
|---|---|
| `Lowering/bufferization.mlir` | `one-shot-bufferize` converts tensor arguments to memrefs |
| `Lowering/elementwise-add-to-loops.mlir` | `linalg.add` lowers to a nested `scf.for` with scalar `arith.addf` |
| `Lowering/matmul-to-loops.mlir` | `linalg.fill` + `linalg.matmul` lower to a zero-fill loop nest plus a triply-nested `scf.for` |
| `Lowering/conv2d-to-loops.mlir` | `linalg.conv_2d_nhwc_hwcf` lowers to nested `scf.for` loops with a multiply-accumulate per output element |
| `Lowering/fused-relu-add-fusion.mlir` | `-linalg-fuse-elementwise-ops` fuses `linalg.add` + `linalg.max` into a single `linalg.generic` |
| `Lowering/reduce-rows-to-loops.mlir` | `linalg.reduce` lowers to a fill loop plus a doubly-nested accumulation loop with `arith.addf` |
| `Lowering/tiled-matmul-tiling.mlir` | `transform.structured.tile_using_for` tiles the matmul into nested `scf.for` loops over tile-sized `linalg.matmul`s |
| `Lowering/tiled-fused-relu-add-fusion.mlir` | Tiling `linalg.max` and fusing `linalg.add` into it produces a single `scf.for` containing both ops |
| `Lowering/vectorized-matmul-vectorization.mlir` | `transform.structured.vectorize_children_and_apply_patterns` turns the matmul into a single `vector.contract` |
| `Lowering/ttensor-relu-add-to-linalg.mlir` | `-convert-ttensor-to-linalg` expands `ttensor.relu_add` into `linalg.add`/`linalg.fill`/`linalg.max` |
| `Lowering/elementwise-add-to-llvm.mlir` | Full tensor→`llvm` pipeline; asserts no `tensor`/`linalg`/`scf` ops remain |
| `Lowering/fused-relu-add-to-llvm.mlir` | Full pipeline for `relu(add)`; asserts an `llvm.intr.maximum` remains for the relu clamp |
| `Lowering/reduce-rows-to-llvm.mlir` | Full pipeline for the row reduction |
| `Lowering/tiled-matmul-to-llvm.mlir` | Full pipeline (transform-interpreter + lowering) for the tiled matmul |
| `Lowering/vectorized-matmul-to-llvm.mlir` | Full pipeline (transform-interpreter + lowering) for the vectorized matmul |
| `Lowering/ttensor-relu-add-to-llvm.mlir` | Full pipeline (custom-dialect desugaring + lowering) for `ttensor.relu_add` |

### Running the tests

```sh
pip install lit   # one-time setup; FileCheck ships with LLVM/MLIR
cmake -S . -B build -G Ninja \
  -DMLIR_DIR=$(brew --prefix llvm)/lib/cmake/mlir \
  -DLLVM_DIR=$(brew --prefix llvm)/lib/cmake/llvm \
  -DLIT_EXECUTABLE=$(command -v lit)
cmake --build build --target check
```

### Test results

Verified on macOS (arm64) with Homebrew LLVM/MLIR 22.1.6, 2026-06-25:

```
✅ 18/18 tests passed  (1.20s)   Platform: macOS arm64 · LLVM/MLIR 22.1.6
```

<details>
<summary>Full test output</summary>

```
-- Testing: 18 tests, 8 workers --

Driver (2/2)
  PASS  Driver/load-tensor-ir.mlir
  PASS  Driver/load-ttensor-ir.mlir

Lowering — loops & fusion (6/6)
  PASS  Lowering/bufferization.mlir
  PASS  Lowering/elementwise-add-to-loops.mlir
  PASS  Lowering/matmul-to-loops.mlir
  PASS  Lowering/conv2d-to-loops.mlir
  PASS  Lowering/fused-relu-add-fusion.mlir
  PASS  Lowering/reduce-rows-to-loops.mlir

Lowering — transform dialect (3/3)
  PASS  Lowering/tiled-matmul-tiling.mlir
  PASS  Lowering/tiled-fused-relu-add-fusion.mlir
  PASS  Lowering/vectorized-matmul-vectorization.mlir

Lowering — custom dialect (2/2)
  PASS  Lowering/ttensor-relu-add-to-linalg.mlir
  PASS  Lowering/ttensor-relu-add-to-llvm.mlir

Lowering — end-to-end LLVM (5/5)
  PASS  Lowering/elementwise-add-to-llvm.mlir
  PASS  Lowering/fused-relu-add-to-llvm.mlir
  PASS  Lowering/reduce-rows-to-llvm.mlir
  PASS  Lowering/tiled-matmul-to-llvm.mlir
  PASS  Lowering/vectorized-matmul-to-llvm.mlir

Testing Time: 1.20s
Total Discovered Tests: 18
  Passed: 18 (100.00%)
```

</details>

### JIT execution results

All nine `_main` examples produce the mathematically correct result via JIT:

<details>
<summary>elementwise_add — A+B where A=[[1,2],[3,4]], B=[[5,6],[7,8]]</summary>

```
$ scripts/run_jit_example.sh examples/elementwise_add_main.mlir
Unranked Memref base@ = 0x... rank = 2 offset = 0 sizes = [2, 2] strides = [2, 1] data =
[[6,    8],
 [10,   12]]
```
Expected: `A+B = [[6,8],[10,12]]` ✓

</details>

<details>
<summary>matmul — A×B where A=[[1,2],[3,4]], B=[[5,6],[7,8]]</summary>

```
$ scripts/run_jit_example.sh examples/matmul_main.mlir
Unranked Memref base@ = 0x... rank = 2 offset = 0 sizes = [2, 2] strides = [2, 1] data =
[[19,   22],
 [43,   50]]
```
Expected: `A×B = [[19,22],[43,50]]` ✓

</details>

<details>
<summary>conv2d — 5×5 input (1..25) convolved with 3×3 all-ones filter</summary>

```
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
```
Expected: box-sum of each 3×3 window over a row-major `1..25` input ✓

</details>

<details>
<summary>fused_relu_add — relu(A+B) where A=[[1,-5],[3,-2]], B=[[2,1],[-10,5]]</summary>

```
$ scripts/run_jit_example.sh examples/fused_relu_add_main.mlir
Unranked Memref base@ = 0x... rank = 2 offset = 0 sizes = [2, 2] strides = [2, 1] data =
[[3,   0],
 [0,   3]]
```
Expected: `A+B = [[3,-4],[-7,3]]`, `relu(A+B) = [[3,0],[0,3]]` ✓

</details>

<details>
<summary>reduce_rows — row sums of A=[[1,2,3],[4,5,6]]</summary>

```
$ scripts/run_jit_example.sh examples/reduce_rows_main.mlir
Unranked Memref base@ = 0x... rank = 1 offset = 0 sizes = [2] strides = [1] data =
[6,   15]
```
Expected: `[1+2+3, 4+5+6] = [6, 15]` ✓

</details>

<details>
<summary>tiled_matmul — A×I where A is 4×4 and I is the identity</summary>

```
$ PRE_PASSES="-transform-interpreter" scripts/run_jit_example.sh examples/tiled_matmul_main.mlir
Unranked Memref base@ = 0x... rank = 2 offset = 0 sizes = [4, 4] strides = [4, 1] data =
[[1,    2,    3,    4],
 [5,    6,    7,    8],
 [9,    10,   11,   12],
 [13,   14,   15,   16]]
```
Expected: `A×I = A = [[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]]` ✓

</details>

<details>
<summary>tiled_fused_relu_add — relu(A+B) where A=2.0 everywhere, B=-10.0 (top half) / +10.0 (bottom half)</summary>

```
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
```
Expected: top half `max(2-10, 0) = 0`, bottom half `max(2+10, 0) = 12` ✓

</details>

<details>
<summary>vectorized_matmul — same A×B as matmul, lowered via vector.contract</summary>

```
$ PRE_PASSES="-transform-interpreter" scripts/run_jit_example.sh examples/vectorized_matmul_main.mlir
Unranked Memref base@ = 0x... rank = 2 offset = 0 sizes = [2, 2] strides = [2, 1] data =
[[19,   22],
 [43,   50]]
```
Expected: same `A×B = [[19,22],[43,50]]` ✓

</details>

<details>
<summary>ttensor_relu_add — same relu(A+B) as fused_relu_add, expressed via ttensor.relu_add</summary>

```
$ PRE_PASSES="-convert-ttensor-to-linalg" scripts/run_jit_example.sh examples/ttensor_relu_add_main.mlir
Unranked Memref base@ = 0x... rank = 2 offset = 0 sizes = [2, 2] strides = [2, 1] data =
[[3,   0],
 [0,   3]]
```
Expected: same result as `fused_relu_add`, confirming `ttensor.relu_add` desugars correctly ✓

</details>

Native execution via `scripts/build_native_example.sh` was also verified for every example and produces byte-for-byte identical output to the JIT path above.

---

## Design Principles

- Prefer clarity over cleverness.
- Keep each pass independently testable.
- Make intermediate IR easy to inspect.
- Start with static shapes before supporting dynamic shapes.
- Reuse standard MLIR dialects before inventing custom abstractions.
- Document surprising lowering behavior with small examples.

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
