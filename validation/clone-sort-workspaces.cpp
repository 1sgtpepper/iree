// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <utility>

#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/Flow/Transforms/ConvertRegionToWorkgroups.h"
#include "iree/compiler/Dialect/Flow/Transforms/FormDispatchRegions.h"
#include "iree/compiler/Dialect/Flow/Transforms/RegionOpUtils.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "iree/compiler/DispatchCreation/Passes.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"

#define DEBUG_TYPE                                                             \
  "iree-dispatch-creation-convert-dispatch-regions-to-workgroups"

namespace mlir::iree_compiler::DispatchCreation {

#define GEN_PASS_DEF_CONVERTDISPATCHREGIONSTOWORKGROUPSPASS
#include "iree/compiler/DispatchCreation/Passes.h.inc"

namespace {
struct ConvertDispatchRegionsToWorkgroupsPass
    : impl::ConvertDispatchRegionsToWorkgroupsPassBase<
          ConvertDispatchRegionsToWorkgroupsPass> {
  using Base::Base;
  void runOnOperation() override;
};
} // namespace

// Sort swaps retained comparator lanes even when their results do not escape.
// Experiment: clone captured working state before outlining Sort.
static FailureOr<IREE::Flow::DispatchWorkgroupsOp>
convertDispatchRegionWithSortWorkspaces(IREE::Flow::DispatchRegionOp &regionOp,
                                        RewriterBase &rewriter) {
  if (!llvm::hasSingleElement(regionOp.getBody())) {
    return rewriteFlowDispatchRegionToFlowDispatchWorkgroups(regionOp,
                                                             rewriter);
  }
  auto returnOp =
      cast<IREE::Flow::ReturnOp>(regionOp.getBody().front().getTerminator());
  SmallVector<Value> workspaceResults;
  SmallVector<SmallVector<Value>> workspaceDims;
  OpBuilder::InsertionGuard guard(rewriter);

  for (auto sortOp :
       regionOp.getBody().front().getOps<IREE::LinalgExt::SortOp>()) {
    if (!sortOp.hasPureTensorSemantics()) {
      continue;
    }
    bool hasLiveResult = llvm::any_of(sortOp.getResults(), [&](Value result) {
      return llvm::any_of(result.getUses(), [&](OpOperand &use) {
        if (isa<tensor::DimOp>(use.getOwner())) {
          return false;
        }
        return use.getOwner() != returnOp ||
               !regionOp->getResult(use.getOperandNumber()).use_empty();
      });
    });
    if (!hasLiveResult) {
      continue;
    }
    auto comparatorArgs = sortOp.getRegion().front().getArguments();
    for (auto result : sortOp.getResults()) {
      unsigned index = result.getResultNumber();
      if (!result.use_empty() || (comparatorArgs[2 * index].use_empty() &&
                                  comparatorArgs[2 * index + 1].use_empty())) {
        continue;
      }
      OpOperand &init = *sortOp.getDpsInitOperand(index);
      Value source = init.get();
      if (regionOp.getBody().isAncestor(source.getParentRegion())) {
        continue;
      }
      rewriter.setInsertionPoint(regionOp);
      SmallVector<Value> dims;
      if (failed(IREE::Flow::getOptimizedDynamicResultDims(rewriter, source,
                                                           dims))) {
        continue;
      }
      auto cloneOp = IREE::Flow::TensorCloneOp::create(
          rewriter, sortOp.getLoc(), source);
      rewriter.modifyOpInPlace(sortOp, [&]() { init.set(cloneOp); });
      workspaceResults.push_back(result);
      workspaceDims.push_back(std::move(dims));
    }
  }

  if (!workspaceResults.empty()) {
    auto dialectAttrs = llvm::to_vector(regionOp->getDialectAttrs());
    Region workgroupCount;
    workgroupCount.takeBody(regionOp.getWorkgroupCount());
    auto newRegionOp = IREE::Flow::appendDispatchRegionResults(
        rewriter, regionOp, workspaceResults, workspaceDims);
    if (failed(newRegionOp)) {
      return failure();
    }
    (*newRegionOp)->setDialectAttrs(dialectAttrs);
    newRegionOp->getWorkgroupCount().takeBody(workgroupCount);
    regionOp = *newRegionOp;
  }
  return rewriteFlowDispatchRegionToFlowDispatchWorkgroups(regionOp,
                                                           rewriter);
}

// Creates a DispatchWorkgroupsOp for every DispatchRegionOp.
void ConvertDispatchRegionsToWorkgroupsPass::runOnOperation() {
  FunctionOpInterface funcOp = getOperation();
  TensorDimTrackingRewriter rewriter(funcOp);

  SmallVector<IREE::Flow::DispatchRegionOp> regionOps;
  funcOp.walk(
      [&](IREE::Flow::DispatchRegionOp op) { regionOps.push_back(op); });

  numDispatches += regionOps.size();

  // Clone additional producers and rewrite to DispatchWorkgroupsOp.
  for (auto regionOp : regionOps) {
    auto maybeWorkgroupOp =
        convertDispatchRegionWithSortWorkspaces(regionOp, rewriter);
    if (failed(maybeWorkgroupOp)) {
      regionOp.emitError(
          "failed to convert dispatch.region op to dispatch.workgroup op");
      return signalPassFailure();
    }
  }
}
} // namespace mlir::iree_compiler::DispatchCreation
