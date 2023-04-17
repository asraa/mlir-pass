// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.
#include <iostream>

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
 #include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Sequence.h"


// Lowers arith::SelectOp's to conditional branches.
namespace hello
{
  class LowerSelectLowering : public mlir::ConversionPattern
  {
  public:
    explicit LowerSelectLowering(mlir::MLIRContext *context)
        : mlir::ConversionPattern(mlir::arith::SelectOp::getOperationName(), 1, context) {}

    mlir::LogicalResult matchAndRewrite(mlir::Operation *op,
                                        mlir::ArrayRef<mlir::Value> operands,
                                        mlir::ConversionPatternRewriter &rewriter) const override
    {
      auto loc = op->getLoc();
      auto selectOp = mlir::cast<mlir::arith::SelectOp>(op);
      auto result = rewriter.create<mlir::scf::IfOp>(
        loc, 
        selectOp.getCondition(),
        /*thenBuilder=*/
        [&](mlir::OpBuilder &b, mlir::Location loc) {
          b.create<mlir::scf::YieldOp>(loc, selectOp.getTrueValue());
        },
        /*elseBuilder=*/
        [&](mlir::OpBuilder &b, mlir::Location loc) {
          b.create<mlir::scf::YieldOp>(loc, selectOp.getFalseValue());
        });
      rewriter.replaceOp(op, result.getResults());
      return mlir::success();
    }
  };

  class LowerSelectLoweringPass
      : public mlir::PassWrapper<LowerSelectLoweringPass, mlir::OperationPass<mlir::ModuleOp>>
  {
  public:
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerSelectLoweringPass)
    void getDependentDialects(mlir::DialectRegistry &registry) const override
    {
      registry.insert<mlir::AffineDialect, mlir::memref::MemRefDialect, mlir::arith::ArithDialect, mlir::scf::SCFDialect>();
    }

    void runOnOperation() final;
  };

  void LowerSelectLoweringPass::runOnOperation()
  {
    mlir::ConversionTarget target(getContext());

    target.addIllegalOp<mlir::arith::SelectOp>();
    target.addLegalOp<mlir::scf::IfOp, mlir::scf::YieldOp>();

    mlir::RewritePatternSet patterns(&getContext());
    patterns.add<hello::LowerSelectLowering>(&getContext());

    if (mlir::failed(mlir::applyPartialConversion(getOperation(), target, std::move(patterns))))
    {
      signalPassFailure();
    }
  }

  std::unique_ptr<mlir::Pass> createLowerSelectPass()
  {
    return std::make_unique<LowerSelectLoweringPass>();
  }

}
