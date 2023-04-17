
#include "mlir/IR/Dominance.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;

namespace hello {



struct LowerCopy : public OpRewritePattern<memref::CopyOp> {
  LowerCopy(MLIRContext *context)
      : OpRewritePattern(context)  {}

  using OpRewritePattern<memref::CopyOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::CopyOp copy,
                                PatternRewriter &rewriter) const override {
    rewriter.setInsertionPoint(copy);
    auto loc = copy.getLoc();
    auto memrefType = copy.getSource().getType().cast<MemRefType>();

    // Create explicit memory copy using an affine loop nest.
    SmallVector<Value, 4> ivs;
    auto constantZero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    for (auto dimSize : memrefType.getShape()) {
      if (dimSize == 1) {
        ivs.push_back(constantZero);
        continue;
      }
      auto loop = rewriter.create<mlir::AffineForOp>(loc, 0, dimSize);

      rewriter.setInsertionPointToStart(loop.getBody());
      ivs.push_back(loop.getInductionVar());
    }

    // Create affine load/store operations.
    auto value =
        rewriter.create<mlir::AffineLoadOp>(loc, copy.getSource(), ivs);
    rewriter.create<mlir::AffineStoreOp>(loc, value, copy.getTarget(), ivs);

    rewriter.eraseOp(copy);
    return success();
  }
};

  class LowerCopyToAffine
      : public mlir::PassWrapper<LowerCopyToAffine, mlir::OperationPass<mlir::ModuleOp>>
  {
  public:
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerCopyToAffine)
    void getDependentDialects(mlir::DialectRegistry &registry) const override
    {
      registry.insert<mlir::AffineDialect, mlir::memref::MemRefDialect, mlir::arith::ArithDialect, mlir::scf::SCFDialect>();
    }

    void runOnOperation() final;
  };

  void LowerCopyToAffine::runOnOperation()
  {
    mlir::ConversionTarget target(getContext());

    // Not sure if I should also add mlir::memref::GlobalOp as illegal and remove that.
    target.addIllegalOp<mlir::memref::CopyOp>();
    target.addLegalOp<mlir::arith::ConstantOp, mlir::memref::AllocOp, mlir::AffineStoreOp, mlir::AffineLoadOp, mlir::AffineYieldOp, mlir::AffineForOp>();

    mlir::RewritePatternSet patterns(&getContext());
    patterns.add<LowerCopy>(&getContext());

    if (mlir::failed(mlir::applyPartialConversion(getOperation(), target, std::move(patterns))))
    {
      signalPassFailure();
    }
  }


std::unique_ptr<Pass>
createLowerCopyToAffinePass() {
  return std::make_unique<LowerCopyToAffine>();
}

}