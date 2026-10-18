// The earlier key SSA version is returned after a dependent Sort updates it.
func.func @dependent_observed_dynamic(%keys: tensor<?xi64>, %labels: tensor<?xi64>)
    -> (tensor<?xi64>, tensor<?xi64>, tensor<?xi64>, tensor<?xi64>) {
  %c0 = arith.constant 0 : index
  %n = tensor.dim %keys, %c0 : tensor<?xi64>
  %results:2 = flow.dispatch.region -> (tensor<?xi64>{%n}, tensor<?xi64>{%n}) {
    %ascending:2 = iree_linalg_ext.sort dimension(0)
        outs(%keys, %labels : tensor<?xi64>, tensor<?xi64>) {
    ^bb0(%lhs: i64, %rhs: i64, %lhs_label: i64, %rhs_label: i64):
      %cmp = arith.cmpi sle, %lhs, %rhs : i64
      iree_linalg_ext.yield %cmp : i1
    } -> tensor<?xi64>, tensor<?xi64>
    %descending:2 = iree_linalg_ext.sort dimension(0)
        outs(%ascending#0, %ascending#1 : tensor<?xi64>, tensor<?xi64>) {
    ^bb0(%lhs: i64, %rhs: i64, %lhs_label: i64, %rhs_label: i64):
      %cmp = arith.cmpi sge, %lhs, %rhs : i64
      iree_linalg_ext.yield %cmp : i1
    } -> tensor<?xi64>, tensor<?xi64>
    flow.return %descending#1, %ascending#0 : tensor<?xi64>, tensor<?xi64>
  }
  return %results#0, %results#1, %keys, %labels
      : tensor<?xi64>, tensor<?xi64>, tensor<?xi64>, tensor<?xi64>
}
