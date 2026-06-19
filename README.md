# Tensor Compiler Mini-Pipeline

A small MLIR-based compiler project for learning how high-level tensor programs become lower-level executable code. The pipeline starts with tensor operations expressed in the `linalg`/`tensor` dialects and lowers them through MLIR's standard dialects — `linalg`, `bufferization`, `scf`, `affine`, `memref`, `arith`, `cf`, and finally `llvm` — down to code that can be JIT-executed.

This project is intentionally scoped as a learning compiler rather than a production framework. Each stage is meant to be easy to understand, test, and debug by hand.

## Status

Milestone 1 is complete: elementwise tensor addition and matrix multiplication both lower end to end from tensor-level IR to the `llvm` dialect and execute correctly under MLIR's JIT runner. See [Test Results](#test-results) below for verified output.

## Goals

- Represent tensor computations in a high-level MLIR dialect (`tensor` + `linalg`).
- Lower tensor operations through standard MLIR dialects.
- Provide a custom compiler driver (`tensor-pipeline-opt`) that accepts `.mlir` input and runs a configurable pass pipeline, mirroring `mlir-opt`.
- Generate runnable code through the LLVM lowering path and execute it via MLIR's JIT runner.
- Add simple optimization passes such as tiling, fusion, vectorization, and bufferization (future work).
- Keep examples small enough that every pass can be inspected by hand.

## Pipeline

The current pipeline, exercised end to end by the examples and tests in this repo:

```text
tensor + linalg input
  -> linalg on tensors
  -> one-shot-bufferize (tensor -> memref)
  -> convert-linalg-to-loops (linalg -> scf.for)
  -> lower-affine, convert-scf-to-cf
  -> convert-{cf,arith,memref,func}-to-llvm
  -> reconcile-unrealized-casts
  -> llvm dialect
  -> JIT execution via mlir-runner
```

Tiling, fusion, vectorization, and a path to a native executable (rather than JIT) remain future work — see [Roadmap](#roadmap).

## Example Programs

Implemented:

- `examples/elementwise_add.mlir` / `elementwise_add_main.mlir` — elementwise tensor addition (`linalg.add`).
- `examples/matmul.mlir` / `matmul_main.mlir` — matrix multiplication (`linalg.fill` + `linalg.matmul`).

Each example has two files:

- The plain version takes tensors as function arguments and returns a tensor — used by the FileCheck lowering tests.
- The `_main` version embeds constant input tensors and prints the result — used for JIT execution via `scripts/run_jit_example.sh`.

Planned: 2D convolution, reduction over one dimension, and a fused elementwise example such as `relu(add(A, B))`.

## Repository Layout

```text
.
├── tools/tensor-pipeline-opt/   # The compiler driver (mlir-opt-style tool)
├── examples/                    # Tensor-level input programs (plain + JIT-runnable)
├── test/                        # lit + FileCheck tests for the driver and pass pipeline
│   ├── Driver/                  # Parsing/round-trip tests
│   └── Lowering/                # Per-stage and end-to-end lowering tests
├── scripts/run_jit_example.sh   # Lowers and JIT-executes a `_main` example
├── CMakeLists.txt               # Build configuration
└── README.md
```

`docs/` and a custom tensor dialect are not part of the current scope; the project currently builds entirely on upstream MLIR dialects.

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
  -lower-affine \
  -convert-scf-to-cf \
  -convert-cf-to-llvm \
  -convert-arith-to-llvm \
  -finalize-memref-to-llvm \
  -convert-func-to-llvm \
  -reconcile-unrealized-casts
```

This prints the fully lowered `llvm` dialect IR for elementwise add. The same pipeline works for `examples/matmul.mlir`.

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
```

It locates `mlir-runner` and the MLIR runner-utils shared libraries via `brew --prefix llvm`; set `LLVM_PREFIX` to override.

## Testing

The test suite uses MLIR's `lit` and `FileCheck`. Each test focuses on one stage of the pipeline:

- `test/Driver/load-tensor-ir.mlir` — the driver loads and round-trips tensor/linalg input unchanged.
- `test/Lowering/bufferization.mlir` — `one-shot-bufferize` turns tensor arguments into memrefs.
- `test/Lowering/elementwise-add-to-loops.mlir` — `linalg.add` lowers to a nested `scf.for` loop with scalar `arith.addf`.
- `test/Lowering/matmul-to-loops.mlir` — `linalg.fill` + `linalg.matmul` lower to a zero-fill loop nest plus a triply-nested `scf.for` loop.
- `test/Lowering/elementwise-add-to-llvm.mlir` — the full tensor-to-`llvm` pipeline, asserting no `tensor`/`linalg`/`scf` ops remain.

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

Verified on macOS (arm64) with Homebrew LLVM/MLIR 22.1.6, 2026-06-19:

```text
-- Testing: 5 tests, 5 workers --
PASS: tensor-pipeline :: Driver/load-tensor-ir.mlir
PASS: tensor-pipeline :: Lowering/bufferization.mlir
PASS: tensor-pipeline :: Lowering/elementwise-add-to-loops.mlir
PASS: tensor-pipeline :: Lowering/matmul-to-loops.mlir
PASS: tensor-pipeline :: Lowering/elementwise-add-to-llvm.mlir

Testing Time: 1.66s
Total Discovered Tests: 5
  Passed: 5 (100.00%)
```

JIT execution of both examples produces the mathematically correct result:

```text
$ scripts/run_jit_example.sh examples/elementwise_add_main.mlir
Unranked Memref base@ = 0x... rank = 2 offset = 0 sizes = [2, 2] strides = [2, 1] data =
[[6,    8],
 [10,   12]]

$ scripts/run_jit_example.sh examples/matmul_main.mlir
Unranked Memref base@ = 0x... rank = 2 offset = 0 sizes = [2, 2] strides = [2, 1] data =
[[19,   22],
 [43,   50]]
```

(`A = [[1,2],[3,4]]`, `B = [[5,6],[7,8]]`: `A+B = [[6,8],[10,12]]` and `A x B = [[19,22],[43,50]]`, both confirmed correct.)

## Roadmap

1. ~~Create the CMake project skeleton.~~
2. ~~Add a minimal compiler driver.~~
3. ~~Accept `.mlir` input files and run a configurable pass pipeline.~~
4. ~~Lower one elementwise tensor example end to end.~~
5. ~~Add matrix multiplication.~~
6. ~~Add pass tests with `lit` and `FileCheck`.~~
7. Implement tiling and fusion experiments.
8. Add vectorization for small static shapes.
9. Add documentation for each lowering stage.
10. Optionally add a small custom tensor dialect.
11. Add a path to a native executable (object file + linking), not just JIT execution.

## Design Principles

- Prefer clarity over cleverness.
- Keep each pass independently testable.
- Make intermediate IR easy to inspect.
- Start with static shapes before supporting dynamic shapes.
- Reuse standard MLIR dialects before inventing custom abstractions.
- Document surprising lowering behavior with small examples.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
