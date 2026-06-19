# Tensor Compiler Mini-Pipeline

A small MLIR-based compiler project for learning how high-level tensor programs become lower-level executable code. The goal is to build a focused, inspectable pipeline that starts with tensor operations and progressively lowers them through MLIR dialects such as `linalg`, `scf`, `affine`, `vector`, `memref`, and `llvm`.

This project is intentionally scoped as a learning compiler rather than a production framework. Each stage should make the IR easier to understand, test, and debug.

## Goals

- Parse or define a tiny tensor-oriented input language.
- Represent tensor computations in a high-level MLIR dialect.
- Lower tensor operations through standard MLIR dialects.
- Add simple optimization passes such as tiling, fusion, vectorization, and bufferization.
- Generate runnable code through the LLVM lowering path.
- Keep examples small enough that every pass can be inspected by hand.

## Planned Pipeline

The initial compiler pipeline will likely follow this shape:

```text
Toy Tensor Input
  -> custom tensor dialect or MLIR tensor/linalg input
  -> linalg on tensors
  -> bufferization
  -> linalg on memrefs
  -> scf / affine loops
  -> vector dialect
  -> memref + arith + cf
  -> llvm dialect
  -> native executable or JIT execution
```

The first useful milestone is a single operation such as elementwise add or matrix multiplication. Once that path works end to end, the project can grow into more interesting transformations.

## Example Programs

Planned examples:

- Elementwise tensor addition
- Matrix multiplication
- 2D convolution
- Reduction over one dimension
- Fused elementwise operation, such as `relu(add(A, B))`

Example input shape:

```text
func @matmul(%A: tensor<16x16xf32>, %B: tensor<16x16xf32>) -> tensor<16x16xf32>
```

Early versions may use plain `.mlir` files as input before adding a custom parser.

## Repository Layout

Planned structure:

```text
.
├── include/                 # Dialect, pass, and compiler declarations
├── lib/                     # Dialect definitions and pass implementations
├── tools/                   # Compiler driver and command-line tools
├── test/                    # FileCheck-based MLIR tests
├── examples/                # Small tensor programs and expected IR snapshots
├── docs/                    # Design notes and pipeline explanations
├── CMakeLists.txt           # Build configuration
└── README.md
```

## Development Setup

This project is expected to build against LLVM and MLIR from source. A typical setup will look like:

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

Once this repository has a `CMakeLists.txt`, the project can be configured with:

```sh
cmake -S . -B build \
  -G Ninja \
  -DMLIR_DIR=/path/to/llvm-build/lib/cmake/mlir
cmake --build build
```

## Running the Compiler

Planned command shape:

```sh
tensor-pipeline examples/matmul.mlir \
  --lower-to=llvm \
  --print-ir-after-all
```

Useful development modes:

- Print the IR after each pass.
- Run a single pass at a time.
- Verify IR after every transformation.
- Emit timing information for each pass.
- Save intermediate IR files for debugging.

## Testing

The test suite should use MLIR's `lit` and `FileCheck` style tests. Each test should focus on one transformation at a time.

Example test shape:

```mlir
// RUN: tensor-pipeline-opt %s --tile-matmul | FileCheck %s

// CHECK: scf.for
// CHECK: linalg.matmul
```

The first tests should cover:

- Parsing or loading input IR.
- Lowering high-level tensor operations to `linalg`.
- Bufferization.
- Loop generation.
- LLVM dialect lowering.

## Roadmap

1. Create the CMake project skeleton.
2. Add a minimal compiler driver.
3. Accept `.mlir` input files and run a configurable pass pipeline.
4. Lower one elementwise tensor example end to end.
5. Add matrix multiplication.
6. Add pass tests with `lit` and `FileCheck`.
7. Implement tiling and fusion experiments.
8. Add vectorization for small static shapes.
9. Add documentation for each lowering stage.
10. Optionally add a small custom tensor dialect.

## Design Principles

- Prefer clarity over cleverness.
- Keep each pass independently testable.
- Make intermediate IR easy to inspect.
- Start with static shapes before supporting dynamic shapes.
- Reuse standard MLIR dialects before inventing custom abstractions.
- Document surprising lowering behavior with small examples.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
