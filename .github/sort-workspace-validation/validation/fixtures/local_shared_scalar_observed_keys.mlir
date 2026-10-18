// A local key initializer shares its scalar with a live auxiliary fill.
// The original key version remains observable after Sort.
#identity = affine_map<(d0) -> (d0)>
func.func @local_shared_scalar_observed_keys(%labels: tensor<?xi64>)
    -> (tensor<?xi64>, tensor<?xi64>, tensor<?xi64>, tensor<?xi64>) {
  %c0 = arith.constant 0 : index
  %n = tensor.dim %labels, %c0 : tensor<?xi64>
  %results:3 = flow.dispatch.region ->
      (tensor<?xi64>{%n}, tensor<?xi64>{%n}, tensor<?xi64>{%n}) {
    %seed = arith.index_cast %n : index to i64
    %key_empty = tensor.empty(%n) : tensor<?xi64>
    %filled = linalg.fill ins(%seed : i64) outs(%key_empty : tensor<?xi64>) -> tensor<?xi64>
    %keys = linalg.generic {indexing_maps = [#identity], iterator_types = ["parallel"]}
        outs(%filled : tensor<?xi64>) {
    ^bb0(%base: i64):
      %i = linalg.index 0 : index
      %iv = arith.index_cast %i : index to i64
      %key = arith.subi %base, %iv : i64
      linalg.yield %key : i64
    } -> tensor<?xi64>
    %aux_empty = tensor.empty(%n) : tensor<?xi64>
    %aux = linalg.fill ins(%seed : i64) outs(%aux_empty : tensor<?xi64>) -> tensor<?xi64>
    %sorted:2 = iree_linalg_ext.sort dimension(0)
        outs(%keys, %labels : tensor<?xi64>, tensor<?xi64>) {
    ^bb0(%lhs: i64, %rhs: i64, %lhs_label: i64, %rhs_label: i64):
      %cmp = arith.cmpi sle, %lhs, %rhs : i64
      iree_linalg_ext.yield %cmp : i1
    } -> tensor<?xi64>, tensor<?xi64>
    flow.return %sorted#1, %keys, %aux : tensor<?xi64>, tensor<?xi64>, tensor<?xi64>
  }
  return %results#0, %results#1, %results#2, %labels
      : tensor<?xi64>, tensor<?xi64>, tensor<?xi64>, tensor<?xi64>
}
