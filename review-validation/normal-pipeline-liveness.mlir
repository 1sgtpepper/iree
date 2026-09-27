// Fork-only scope check: normal dispatch creation must remove unused sorting.
// A live argsort is the positive control.

// CHECK-LABEL: util.func public @dead_sort_user(
// CHECK-NOT: iree_linalg_ext.sort
// CHECK: util.return
util.func public @dead_sort_user(
    %keys: tensor<4xi64>, %indices: tensor<4xi64>) -> tensor<4xi64> {
  %c0 = arith.constant 0 : index
  %sorted:2 = iree_linalg_ext.sort dimension(0)
      outs(%keys, %indices : tensor<4xi64>, tensor<4xi64>) {
  ^bb0(%lhs: i64, %rhs: i64, %lhs_index: i64, %rhs_index: i64):
    %take_lhs = arith.cmpi sle, %lhs, %rhs : i64
    iree_linalg_ext.yield %take_lhs : i1
  } -> tensor<4xi64>, tensor<4xi64>
  %unused = tensor.extract %sorted#0[%c0] : tensor<4xi64>
  util.return %indices : tensor<4xi64>
}

// -----

// CHECK-LABEL: util.func public @static_shape_only(
// CHECK-NOT: iree_linalg_ext.sort
// CHECK: util.return
util.func public @static_shape_only(
    %keys: tensor<4xi64>, %indices: tensor<4xi64>) -> tensor<4xi64> {
  %c0 = arith.constant 0 : index
  %sorted:2 = iree_linalg_ext.sort dimension(0)
      outs(%keys, %indices : tensor<4xi64>, tensor<4xi64>) {
  ^bb0(%lhs: i64, %rhs: i64, %lhs_index: i64, %rhs_index: i64):
    %take_lhs = arith.cmpi sle, %lhs, %rhs : i64
    iree_linalg_ext.yield %take_lhs : i1
  } -> tensor<4xi64>, tensor<4xi64>
  %dim = tensor.dim %sorted#0, %c0 : tensor<4xi64>
  %value = arith.index_cast %dim : index to i64
  %filled = linalg.fill ins(%value : i64) outs(%indices : tensor<4xi64>)
      -> tensor<4xi64>
  util.return %filled : tensor<4xi64>
}

// -----

// CHECK-LABEL: util.func public @dynamic_shape_only(
// CHECK-NOT: iree_linalg_ext.sort
// CHECK: util.return
util.func public @dynamic_shape_only(
    %keys: tensor<?xi64>, %indices: tensor<?xi64>) -> tensor<?xi64> {
  %c0 = arith.constant 0 : index
  %sorted:2 = iree_linalg_ext.sort dimension(0)
      outs(%keys, %indices : tensor<?xi64>, tensor<?xi64>) {
  ^bb0(%lhs: i64, %rhs: i64, %lhs_index: i64, %rhs_index: i64):
    %take_lhs = arith.cmpi sle, %lhs, %rhs : i64
    iree_linalg_ext.yield %take_lhs : i1
  } -> tensor<?xi64>, tensor<?xi64>
  %dim = tensor.dim %sorted#0, %c0 : tensor<?xi64>
  %value = arith.index_cast %dim : index to i64
  %filled = linalg.fill ins(%value : i64) outs(%indices : tensor<?xi64>)
      -> tensor<?xi64>
  util.return %filled : tensor<?xi64>
}

// -----

// CHECK-LABEL: util.func public @live_argsort(
// CHECK: iree_linalg_ext.sort
// CHECK: util.return
util.func public @live_argsort(
    %keys: tensor<?xi64>, %indices: tensor<?xi64>) -> tensor<?xi64> {
  %sorted:2 = iree_linalg_ext.sort dimension(0)
      outs(%keys, %indices : tensor<?xi64>, tensor<?xi64>) {
  ^bb0(%lhs: i64, %rhs: i64, %lhs_index: i64, %rhs_index: i64):
    %take_lhs = arith.cmpi sle, %lhs, %rhs : i64
    iree_linalg_ext.yield %take_lhs : i1
  } -> tensor<?xi64>, tensor<?xi64>
  util.return %sorted#1 : tensor<?xi64>
}
