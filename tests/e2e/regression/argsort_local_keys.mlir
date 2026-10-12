// RUN: iree-run-mlir --Xcompiler,iree-hal-target-backends=llvm-cpu --Xcompiler,iree-llvmcpu-target-cpu=generic %s --function=argsort_local_shared_keys --input="3xi64=[7 -3 9]" | FileCheck %s --check-prefix=SHARED
// RUN: iree-run-mlir --Xcompiler,iree-hal-target-backends=llvm-cpu --Xcompiler,iree-llvmcpu-target-cpu=generic %s --function=argsort_local_keys --input="3xi64=[7 -3 9]" | FileCheck %s
// CHECK-LABEL: EXEC @argsort_local_keys
// CHECK: 3xi64=9 -3 7
#identity = affine_map<(d0) -> (d0)>
func.func @argsort_local_keys(%labels: tensor<?xi64>) -> tensor<?xi64> {
  %c0 = arith.constant 0 : index
  %n = tensor.dim %labels, %c0 : tensor<?xi64>
  %result = flow.dispatch.region -> (tensor<?xi64>{%n}) {
    %empty = tensor.empty(%n) : tensor<?xi64>
    %keys = linalg.generic {indexing_maps = [#identity], iterator_types = ["parallel"]} outs(%empty : tensor<?xi64>) {
    ^bb0(%out: i64):
      %zero = arith.constant 0 : i64
      %i = linalg.index 0 : index
      %value = arith.index_cast %i : index to i64
      %key = arith.subi %zero, %value : i64
      linalg.yield %key : i64
    } -> tensor<?xi64>
    %sorted:2 = iree_linalg_ext.sort dimension(0) outs(%keys, %labels : tensor<?xi64>, tensor<?xi64>) {
    ^bb0(%lhs: i64, %rhs: i64, %lhs_label: i64, %rhs_label: i64):
      %cmp = arith.cmpi sle, %lhs, %rhs : i64
      iree_linalg_ext.yield %cmp : i1
    } -> tensor<?xi64>, tensor<?xi64>
    flow.return %sorted#1 : tensor<?xi64>
  }
  return %result : tensor<?xi64>
}

func.func @argsort_local_shared_keys(%labels: tensor<?xi64>) -> (tensor<?xi64>, tensor<?xi64>) {
  %c0 = arith.constant 0 : index
  %n = tensor.dim %labels, %c0 : tensor<?xi64>
  %result:2 = flow.dispatch.region -> (tensor<?xi64>{%n}, tensor<?xi64>{%n}) {
    %empty = tensor.empty(%n) : tensor<?xi64>
    %keys = linalg.generic {indexing_maps = [#identity], iterator_types = ["parallel"]} outs(%empty : tensor<?xi64>) {
    ^bb0(%out: i64):
      %zero = arith.constant 0 : i64
      %i = linalg.index 0 : index
      %value = arith.index_cast %i : index to i64
      %key = arith.subi %zero, %value : i64
      linalg.yield %key : i64
    } -> tensor<?xi64>
    %sorted:2 = iree_linalg_ext.sort dimension(0) outs(%keys, %labels : tensor<?xi64>, tensor<?xi64>) {
    ^bb0(%lhs: i64, %rhs: i64, %lhs_label: i64, %rhs_label: i64):
      %cmp = arith.cmpi sle, %lhs, %rhs : i64
      iree_linalg_ext.yield %cmp : i1
    } -> tensor<?xi64>, tensor<?xi64>
    flow.return %sorted#1, %keys : tensor<?xi64>, tensor<?xi64>
  }
  return %result#0, %result#1 : tensor<?xi64>, tensor<?xi64>
}
// SHARED-LABEL: EXEC @argsort_local_shared_keys
// SHARED: 3xi64=9 -3 7
// SHARED: 3xi64=0 -1 -2
