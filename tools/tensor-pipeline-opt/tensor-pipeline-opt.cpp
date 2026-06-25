//===- tensor-pipeline-opt.cpp - Tensor pipeline compiler driver --------===//
//
// A minimal mlir-opt-style driver for the tensor compiler mini-pipeline.
// It loads an MLIR input file, runs a user-specified --pass-pipeline (or
// a sequence of single pass flags), and prints the resulting IR. This is
// the entry point used to drive the tensor -> linalg -> bufferization ->
// loops -> llvm lowering path described in the project README.
//
//===----------------------------------------------------------------------===//

#include "mlir/InitAllDialects.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

#include "TTensor/Passes.h"
#include "TTensor/TTensorOps.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  mlir::registerAllExtensions(registry);
  mlir::registerAllPasses();

  registry.insert<mlir::ttensor::TTensorDialect>();
  mlir::ttensor::registerConvertTTensorToLinalgPass();

  return mlir::asMainReturnCode(mlir::MlirOptMain(
      argc, argv, "tensor-pipeline-opt: tensor compiler mini-pipeline driver\n",
      registry));
}
