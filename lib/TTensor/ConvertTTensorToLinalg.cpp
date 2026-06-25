//===- ConvertTTensorToLinalg.cpp - ttensor -> tensor/linalg ---*- C++ -*-===//
//===----------------------------------------------------------------------===//

#include "TTensor/Passes.h"
#include "TTensor/TTensorOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace mlir::ttensor;

namespace {

/// Expands ttensor.relu_add(%lhs, %rhs) into the same tensor.empty /
/// linalg.add / linalg.fill / linalg.max sequence used by
/// examples/fused_relu_add.mlir, so every later pipeline stage (bufferize,
/// linalg-to-loops, ..., llvm) only ever sees standard upstream dialects.
struct ReluAddOpLowering : public OpRewritePattern<ReluAddOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ReluAddOp op,
                                 PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto resultType = cast<RankedTensorType>(op.getType());
    ArrayRef<int64_t> shape = resultType.getShape();
    Type elementType = resultType.getElementType();

    Value sumInit = tensor::EmptyOp::create(rewriter, loc, shape, elementType);
    Value sum = linalg::AddOp::create(rewriter, loc, TypeRange{resultType},
                                       ValueRange{op.getLhs(), op.getRhs()},
                                       ValueRange{sumInit})
                    .getResult(0);

    Value zeroInit = tensor::EmptyOp::create(rewriter, loc, shape, elementType);
    Value zeroCst = arith::ConstantOp::create(rewriter, loc,
                                               rewriter.getZeroAttr(elementType));
    Value zeroed = linalg::FillOp::create(rewriter, loc, TypeRange{resultType},
                                           ValueRange{zeroCst},
                                           ValueRange{zeroInit})
                       .getResult(0);

    Value reluInit = tensor::EmptyOp::create(rewriter, loc, shape, elementType);
    Value relu = linalg::MaxOp::create(rewriter, loc, TypeRange{resultType},
                                        ValueRange{sum, zeroed},
                                        ValueRange{reluInit})
                     .getResult(0);

    rewriter.replaceOp(op, relu);
    return success();
  }
};

struct ConvertTTensorToLinalgPass
    : public PassWrapper<ConvertTTensorToLinalgPass,
                          OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertTTensorToLinalgPass)

  StringRef getArgument() const override {
    return "convert-ttensor-to-linalg";
  }
  StringRef getDescription() const override {
    return "Lower ttensor dialect ops to tensor/linalg ops";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, linalg::LinalgDialect,
                     tensor::TensorDialect>();
  }

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<ReluAddOpLowering>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> mlir::ttensor::createConvertTTensorToLinalgPass() {
  return std::make_unique<ConvertTTensorToLinalgPass>();
}

void mlir::ttensor::registerConvertTTensorToLinalgPass() {
  PassRegistration<ConvertTTensorToLinalgPass>();
}
