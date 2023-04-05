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

namespace hello
{
  class MemRefGlobalOpLowering : public mlir::ConversionPattern
  {
  public:
    explicit MemRefGlobalOpLowering(mlir::MLIRContext *context)
        : mlir::ConversionPattern(mlir::memref::GetGlobalOp::getOperationName(), 1, context) {}

    mlir::LogicalResult matchAndRewrite(mlir::Operation *op,
                                        mlir::ArrayRef<mlir::Value> operands,
                                        mlir::ConversionPatternRewriter &rewriter) const override
    {
      // TODO: Implicit location
      // ImplicitLocOpBuilder builder =
      // ImplicitLocOpBuilder::atBlockEnd(module->getLoc(), module->getBody());

      // location and type of the get_global operation
      auto loc = op->getLoc();
      auto getGlobal = mlir::cast<mlir::memref::GetGlobalOp>(op);
      auto memRefType = getGlobal.getType().cast<mlir::MemRefType>();
      auto valueShape = memRefType.getShape();
      auto resultElementType = memRefType.getElementType();

      // First create an allocation with memref.alloc() : type
      mlir::Value alloc = rewriter.create<mlir::memref::AllocOp>(loc, memRefType);

      // Add arith.constant declarations for each index up to the largest dimension.
      mlir::SmallVector<mlir::Value, 8> constantIndices;
      if (!valueShape.empty())
      {
        for (auto i : llvm::seq<int64_t>(
                 0, *std::max_element(valueShape.begin(), valueShape.end())))
          constantIndices.push_back(rewriter.create<mlir::arith::ConstantIndexOp>(loc, i));
      }
      else
      {
        constantIndices.push_back(rewriter.create<mlir::arith::ConstantIndexOp>(loc, 0));
      }
      
      auto module = op->getParentOfType<mlir::ModuleOp>();
      auto global = mlir::cast<mlir::memref::GlobalOp>(
          module.lookupSymbol(getGlobal.getName()));


      if (!global.getConstant() || !global.getInitialValue())
      {
        return mlir::failure();
      }

      auto constVals = global.getInitialValue().value().cast<mlir::DenseElementsAttr>();
      // The splats are unused, but technically I can optimize and replace splats
      // with a constant of the splat value and result type.

      auto values = constVals.tryGetValues<mlir::Attribute>();
      auto valueIt = (*values).begin();
      mlir::SmallVector<mlir::Value, 2> indices;

      std::function<void(uint64_t)> storeElements = [&](uint64_t dimension)
      {
        // The last dimension is the base case of the recursion, at this point
        // we store the element at the given index.
        if (dimension == valueShape.size())
        {
          rewriter.create<mlir::AffineStoreOp>(
              loc, rewriter.create<mlir::arith::ConstantOp>(
                loc, resultElementType, *valueIt++), alloc,
              indices);
          return;
        }

        // Otherwise, iterate over the current dimension and add the indices to
        // the list.
        for (uint64_t i = 0, e = valueShape[dimension]; i != e; ++i)
        {
          indices.push_back(constantIndices[i]);
          storeElements(dimension + 1);
          indices.pop_back();
        }
      };

      storeElements(0);
      rewriter.eraseOp(global);
      rewriter.replaceOp(op, {alloc});
      return mlir::success();
    }
  };

  class MemRefGlobalToArithLoweringPass
      : public mlir::PassWrapper<MemRefGlobalToArithLoweringPass, mlir::OperationPass<mlir::ModuleOp>>
  {
  public:
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MemRefGlobalToArithLoweringPass)
    void getDependentDialects(mlir::DialectRegistry &registry) const override
    {
      registry.insert<mlir::AffineDialect, mlir::memref::MemRefDialect, mlir::arith::ArithDialect>();
    }

    void runOnOperation() final;
  };

  void MemRefGlobalToArithLoweringPass::runOnOperation()
  {
    mlir::ConversionTarget target(getContext());

    // Not sure if I should also add mlir::memref::GlobalOp as illegal and remove that.
    target.addIllegalOp<mlir::memref::GetGlobalOp>();
    target.addLegalOp<mlir::arith::ConstantOp, mlir::memref::AllocOp, mlir::AffineStoreOp>();

    mlir::RewritePatternSet patterns(&getContext());
    patterns.add<hello::MemRefGlobalOpLowering>(&getContext());

    if (mlir::failed(mlir::applyPartialConversion(getOperation(), target, std::move(patterns))))
    {
      signalPassFailure();
    }
  }

  std::unique_ptr<mlir::Pass> createLowerToArithPass()
  {
    return std::make_unique<MemRefGlobalToArithLoweringPass>();
  }

}
