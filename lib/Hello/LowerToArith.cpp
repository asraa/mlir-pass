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

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/Sequence.h"

namespace hello {
class MemRefGlobalOpLowering : public mlir::ConversionPattern {
public:
  explicit MemRefGlobalOpLowering(mlir::MLIRContext *context)
    : mlir::ConversionPattern(mlir::memref::GetGlobalOp::getOperationName(), 1, context) {}

  mlir::LogicalResult matchAndRewrite(mlir::Operation *op,
                                      mlir::ArrayRef<mlir::Value> operands,
                                      mlir::ConversionPatternRewriter &rewriter) const override {
    // auto loc = op->getLoc();

    // Notify the rewriter that this operation has been removed.
    rewriter.eraseOp(op);
    return mlir::success();
  }
};

std::unique_ptr<mlir::Pass> createLowerToArithPass() {
  return std::make_unique<MemRefGlobalToArithLoweringPass>();
}


class MemRefGlobalToArithLoweringPass
        : public mlir::PassWrapper<MemRefGlobalToArithLoweringPass, mlir::OperationPass<mlir::ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MemRefGlobalToArithLoweringPass)
  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::AffineDialect, mlir::memref::MemRefDialect, mlir::arith::ArithDialect>();
  }

  void runOnOperation() final;
};


void MemRefGlobalToArithLoweringPass::runOnOperation() {
  mlir::ConversionTarget target(getContext());
  
  // Not sure if I should also add mlir::memref::GlobalOp as illegal and remove that.
  target.addIllegalOp<mlir::memref::GetGlobalOp>();
  target.addLegalOp<mlir::arith::ConstantOp, mlir::memref::AllocOp, mlir::AffineStoreOp>();


  mlir::RewritePatternSet patterns(&getContext());
  patterns.add<hello::MemRefGlobalOpLowering>(&getContext());

  if (mlir::failed(mlir::applyPartialConversion(getOperation(), target, std::move(patterns)))) {
    signalPassFailure();
  }
}

}
