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

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
 #include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Sequence.h"

/*

This pass replaces affine.load statements with the associated constant,
if available. (This assumes that memrefs are created with simple operations,
an allocation and a store).

This must be the provable last writer to the particular memref being loaded
by the load op, so that the store value can be forwarded to the load.

*/

namespace hello
{
  class AffineLoadOpRemoval : public mlir::ConversionPattern
  {
  public:
    // TODO: I could also flip this and erase all affine.stores with the subsequent loads
    // Don't do this until all memref.copies are lowered!
    explicit AffineLoadOpRemoval(mlir::MLIRContext *context)
        : mlir::ConversionPattern(mlir::AffineLoadOp::getOperationName(), 1, context) {}

    mlir::LogicalResult matchAndRewrite(mlir::Operation *op,
                                        mlir::ArrayRef<mlir::Value> operands,
                                        mlir::ConversionPatternRewriter &rewriter) const override
    {
      // TODO: Implicit location
      // ImplicitLocOpBuilder builder =
      // ImplicitLocOpBuilder::atBlockEnd(module->getLoc(), module->getBody());

      // location and type of the Affine::LoadOp
      auto loc = op->getLoc();
      auto affineLoad = mlir::cast<mlir::AffineLoadOp>(op);
      auto memRefType = affineLoad.getType().cast<mlir::MemRefType>();
      auto valueShape = memRefType.getShape();
      auto resultElementType = memRefType.getElementType();

      // Find an affine.store op referencing the value.
      // If there is none: mlir::failure();

      // Get the affine.store op's value: storedValue

      // Find the result of the affine load op: loadedValue

      // Replace loadedValue with storedValue

      // Erase the affineLoadOp

      // TODO: May be possible to replace the affine.store op

      rewriter.eraseOp(global);
      rewriter.replaceOp(op, {alloc});
      return mlir::success();
    }
  };

  class AffineLoadRemovalPass
      : public mlir::PassWrapper<AffineLoadRemovalPass, mlir::OperationPass<mlir::ModuleOp>>
  {
  public:
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AffineLoadRemovalPass)
    void getDependentDialects(mlir::DialectRegistry &registry) const override
    {
      registry.insert<mlir::AffineDialect, mlir::memref::MemRefDialect, mlir::arith::ArithDialect>();
    }

    void runOnOperation() final;
  };

  void AffineLoadRemovalPass::runOnOperation()
  {
    mlir::ConversionTarget target(getContext());

    target.addIllegalOp<mlir::AffineLoadOp>();
    target.addLegalOp<mlir::arith::ConstantOp>();

    mlir::RewritePatternSet patterns(&getContext());
    patterns.add<hello::AffineLoadOpRemoval>(&getContext());

    if (mlir::failed(mlir::applyPartialConversion(getOperation(), target, std::move(patterns))))
    {
      signalPassFailure();
    }
  }

  std::unique_ptr<mlir::Pass> replaceLoadsPass()
  {
    return std::make_unique<AffineLoadRemovalPass>();
  }

}
