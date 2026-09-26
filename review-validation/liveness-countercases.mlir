// REVIEW CANDIDATES: these fixtures have not been executed.
// Run on both the PR head and its merge base through fork CI.
// RUN: iree-opt %s --pass-pipeline="builtin.module(util.func(iree-dispatch-creation-convert-dispatch-regions-to-workgroups, iree-flow-canonicalize, cse))" -split-input-file | FileCheck %s

// CHECK-LABEL: util.func public @sort_with_dead_internal_user
// CHECK-NOT: iree_linalg_ext.sort
util.func public @sort_with_dead_internal_user(
    %keys: tensor<4xi64>, %indices: tensor<4xi64>) -> tensor<4xi64> {
  %c0 = arith.constant 0 : index
  %result = flow.dispatch.region -> (tensor<4xi64>) {
    %sorted:2 = iree_linalg_ext.sort dimension(0)
        outs(%keys, %indices : tensor<4xi64>, tensor<4xi64>) {
    ^bb0(%lhs_key: i64, %rhs_key: i64, %lhs_index: i64, %rhs_index: i64):
      %take_lhs = arith.cmpi sle, %lhs_key, %rhs_key : i64
      iree_linalg_ext.yield %take_lhs : i1
    } -> tensor<4xi64>, tensor<4xi64>
    %unused = tensor.extract %sorted#0[%c0] : tensor<4xi64>
    flow.return %indices : tensor<4xi64>
  }
  // CHECK: util.return
  util.return %result : tensor<4xi64>
}

// -----

// CHECK-LABEL: util.func public @sort_with_only_shape_use
// CHECK-NOT: iree_linalg_ext.sort
util.func public @sort_with_only_shape_use(
    %keys: tensor<4xi64>, %indices: tensor<4xi64>) -> tensor<4xi64> {
  %c0 = arith.constant 0 : index
  %result = flow.dispatch.region -> (tensor<4xi64>) {
    %sorted:2 = iree_linalg_ext.sort dimension(0)
        outs(%keys, %indices : tensor<4xi64>, tensor<4xi64>) {
    ^bb0(%lhs_key: i64, %rhs_key: i64, %lhs_index: i64, %rhs_index: i64):
      %take_lhs = arith.cmpi sle, %lhs_key, %rhs_key : i64
      iree_linalg_ext.yield %take_lhs : i1
    } -> tensor<4xi64>, tensor<4xi64>
    %dim = tensor.dim %sorted#0, %c0 : tensor<4xi64>
    %value = arith.index_cast %dim : index to i64
    %filled = linalg.fill ins(%value : i64) outs(%indices : tensor<4xi64>)
        -> tensor<4xi64>
    flow.return %filled : tensor<4xi64>
  }
  // CHECK: util.return
  util.return %result : tensor<4xi64>
}
