//===- Passes.h - TTensor dialect passes -----------------------*- C++ -*-===//
//===----------------------------------------------------------------------===//

#ifndef TTENSOR_PASSES_H
#define TTENSOR_PASSES_H

#include <memory>

namespace mlir {
class Pass;

namespace ttensor {

/// Creates a pass that lowers ttensor dialect ops (currently just
/// ttensor.relu_add) to the standard tensor/linalg ops they desugar into.
std::unique_ptr<Pass> createConvertTTensorToLinalgPass();

/// Registers -convert-ttensor-to-linalg with the global pass registry, so
/// it can be passed on the tensor-pipeline-opt command line like any other
/// pass.
void registerConvertTTensorToLinalgPass();

} // namespace ttensor
} // namespace mlir

#endif // TTENSOR_PASSES_H
