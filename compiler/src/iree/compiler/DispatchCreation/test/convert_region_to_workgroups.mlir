// RUN: iree-opt %s --pass-pipeline="builtin.module(util.func(iree-dispatch-creation-convert-dispatch-regions-to-workgroups, iree-flow-canonicalize, cse))" -split-input-file | FileCheck %s

util.global private @device : !hal.device

// CHECK-LABEL: util.func public @foo(
//       CHECK:   %[[argA:.*]]: tensor<?x?xf32>, %[[argB:.*]]: tensor<5x10xf32>, %[[argC:.*]]: tensor<10x11xf32>
util.func public @foo(%argA: tensor<?x?xf32>, %argB: tensor<5x10xf32>, %argC: tensor<10x11xf32>) -> (tensor<?x?xf32>, tensor<5x11xf32>) {
  //  CHECK-DAG: %[[c0:.*]] = arith.constant 0 : index
  //  CHECK-DAG: %[[c1:.*]] = arith.constant 1 : index
  //  CHECK-DAG: %[[dim_argA_0:.*]] = tensor.dim %[[argA]], %[[c0]]
  //  CHECK-DAG: %[[dim_argA_1:.*]] = tensor.dim %[[argA]], %[[c1]]
  //      CHECK: %[[r0:.*]] = flow.dispatch.workgroups(%[[argA]], %[[dim_argA_0]], %[[dim_argA_1]]) : (tensor<?x?xf32>{%[[dim_argA_0]], %[[dim_argA_1]]}, index, index) -> %[[argA]]{%[[dim_argA_0]], %[[dim_argA_1]]} =
  // CHECK-NEXT: (%[[arg1:.*]]: !iree_tensor_ext.dispatch.tensor<readwrite:tensor<?x?xf32>>, %[[arg2:.*]]: index, %[[arg3:.*]]: index) {
  //      CHECK:   %[[load:.*]] = iree_tensor_ext.dispatch.tensor.load %[[arg1]], offsets = [0, 0], sizes = [%[[arg2]], %[[arg3]]], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readwrite:tensor<?x?xf32>>{%[[arg2]], %[[arg3]]} -> tensor<?x?xf32>
  //      CHECK:   iree_tensor_ext.dispatch.tensor.store %[[load]], %[[arg1]], offsets = [0, 0], sizes = [%[[arg2]], %[[arg3]]], strides = [1, 1] : tensor<?x?xf32> -> !iree_tensor_ext.dispatch.tensor<readwrite:tensor<?x?xf32>>{%[[arg2]], %[[arg3]]}
  //      CHECK:   flow.return
  //      CHECK: }
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %dimA0 = tensor.dim %argA, %c0 : tensor<?x?xf32>
  %dimA1 = tensor.dim %argA, %c1 : tensor<?x?xf32>
  %r0 = flow.dispatch.region -> (tensor<?x?xf32>{%dimA0, %dimA1}) {
    flow.return %argA : tensor<?x?xf32>
  }
  //      CHECK: %[[r1:.*]] = flow.dispatch.workgroups(%[[argB]], %[[argC]]) : (tensor<5x10xf32>, tensor<10x11xf32>) -> tensor<5x11xf32>
  // CHECK-SAME:   stream.affinity = #hal.device.affinity<@device>
  // CHECK-NEXT: (%[[arg3:.*]]: !iree_tensor_ext.dispatch.tensor<readonly:tensor<5x10xf32>>, %[[arg4:.*]]: !iree_tensor_ext.dispatch.tensor<readonly:tensor<10x11xf32>>, %[[arg5:.*]]: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<5x11xf32>>)
  //  CHECK-DAG:   %[[loadB:.*]] = iree_tensor_ext.dispatch.tensor.load %[[arg3]], offsets = [0, 0], sizes = [5, 10], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<5x10xf32>> -> tensor<5x10xf32>
  //  CHECK-DAG:   %[[loadC:.*]] = iree_tensor_ext.dispatch.tensor.load %[[arg4]], offsets = [0, 0], sizes = [10, 11], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<10x11xf32>> -> tensor<10x11xf32>
  //      CHECK:   %[[empty:.*]] = tensor.empty() : tensor<5x11xf32>
  //      CHECK:   %[[fill:.*]] = linalg.fill ins(%{{.*}} : f32) outs(%[[empty]] : tensor<5x11xf32>) -> tensor<5x11xf32>
  //      CHECK:   %[[matmul:.*]] = linalg.matmul ins(%[[loadB]], %[[loadC]] : tensor<5x10xf32>, tensor<10x11xf32>) outs(%[[fill]] : tensor<5x11xf32>) -> tensor<5x11xf32>
  //      CHECK:   iree_tensor_ext.dispatch.tensor.store %[[matmul]], %[[arg5]], offsets = [0, 0], sizes = [5, 11], strides = [1, 1] : tensor<5x11xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<5x11xf32>>
  //      CHECK:   flow.return
  //      CHECK: }
  %r1 = flow.dispatch.region -> (tensor<5x11xf32>) attributes {
    stream.affinity = #hal.device.affinity<@device>
  } {
    %zero = arith.constant 0.0 : f32
    %0 = tensor.empty() : tensor<5x11xf32>
    %1 = linalg.fill ins(%zero : f32) outs(%0 : tensor<5x11xf32>) -> tensor<5x11xf32>
    %2 = linalg.matmul ins(%argB, %argC : tensor<5x10xf32>, tensor<10x11xf32>)
        outs(%1 : tensor<5x11xf32>) -> tensor<5x11xf32>
    flow.return %2 : tensor<5x11xf32>
  }

  //      CHECK: util.return %[[r0]], %[[r1]]
  util.return %r0, %r1 : tensor<?x?xf32>, tensor<5x11xf32>
}

// -----

util.global private @device : !hal.device

// CHECK-LABEL: util.func public @sort_unused_comparator_workspace
// CHECK: %[[DISPATCH:.+]]:2 = flow.dispatch.workgroups
// CHECK-SAME: stream.affinity = #hal.device.affinity<@device>
// CHECK: %[[KEY_BINDING:[a-zA-Z0-9_]+]]: !iree_tensor_ext.dispatch.tensor<readonly:tensor<?xi64>>
// CHECK-SAME: %[[INDEX_BINDING:[a-zA-Z0-9_]+]]: !iree_tensor_ext.dispatch.tensor<readwrite:tensor<?xi64>>
// CHECK-SAME: %[[WORKSPACE:[a-zA-Z0-9_]+]]: !iree_tensor_ext.dispatch.tensor<readwrite:tensor<?xi64>>
// CHECK-DAG: %[[KEYS:.+]] = iree_tensor_ext.dispatch.tensor.load %[[KEY_BINDING]]
// CHECK-DAG: %[[INDICES:.+]] = iree_tensor_ext.dispatch.tensor.load %[[INDEX_BINDING]]
// CHECK: %[[DEST:.+]] = iree_tensor_ext.dispatch.tensor.load %[[WORKSPACE]]
// CHECK: %[[COPY:.+]] = linalg.copy ins(%[[KEYS]] : tensor<?xi64>) outs(%[[DEST]] : tensor<?xi64>)
// CHECK: %[[SORT:.+]]:2 = iree_linalg_ext.sort dimension(0) outs(%[[COPY]], %[[INDICES]]
// CHECK: iree_tensor_ext.dispatch.tensor.store %[[SORT]]#1, %[[INDEX_BINDING]]
// CHECK: iree_tensor_ext.dispatch.tensor.store %[[SORT]]#0, %[[WORKSPACE]]
// CHECK: count(%[[COUNT_ARG:.+]]: index)
// CHECK: flow.return %[[COUNT_ARG]], %{{.+}}, %{{.+}} : index, index, index
// CHECK: util.return %[[DISPATCH]]#0
util.func public @sort_unused_comparator_workspace(
    %keys: tensor<?xi64>, %indices: tensor<?xi64>) -> tensor<?xi64> {
  %c0 = arith.constant 0 : index
  %n = tensor.dim %keys, %c0 : tensor<?xi64>
  %result = flow.dispatch.region[%n] -> (tensor<?xi64>{%n}) attributes {
    stream.affinity = #hal.device.affinity<@device>
  } {
    %sorted:2 = iree_linalg_ext.sort dimension(0)
        outs(%keys, %indices : tensor<?xi64>, tensor<?xi64>) {
    ^bb0(%lhs: i64, %rhs: i64, %lhs_index: i64, %rhs_index: i64):
      %cmp = arith.cmpi sle, %lhs, %rhs : i64
      iree_linalg_ext.yield %cmp : i1
    } -> tensor<?xi64>, tensor<?xi64>
    flow.return %sorted#1 : tensor<?xi64>
  } count(%workload: index) -> (index, index, index) {
    %c1 = arith.constant 1 : index
    flow.return %workload, %c1, %c1 : index, index, index
  }
  util.return %result : tensor<?xi64>
}

// -----

// CHECK-LABEL: util.func public @sort_preserves_internal_key_view
// CHECK: %[[DISPATCH:.+]]:2 = flow.dispatch.workgroups
// CHECK: %[[KEY_BINDING:[a-zA-Z0-9_]+]]: !iree_tensor_ext.dispatch.tensor<readonly:tensor<4xi64>>
// CHECK-SAME: %[[INDEX_BINDING:[a-zA-Z0-9_]+]]: !iree_tensor_ext.dispatch.tensor<readwrite:tensor<4xi64>>
// CHECK-SAME: %[[WORKSPACE:[a-zA-Z0-9_]+]]: !iree_tensor_ext.dispatch.tensor<readwrite:tensor<4xi64>>
// CHECK: %[[KEYS:.+]] = iree_tensor_ext.dispatch.tensor.load %[[KEY_BINDING]]
// CHECK: %[[DEST:.+]] = iree_tensor_ext.dispatch.tensor.load %[[WORKSPACE]]
// CHECK: %[[COPY:.+]] = linalg.copy ins(%[[KEYS]] : tensor<4xi64>) outs(%[[DEST]] : tensor<4xi64>)
// CHECK: %[[SORT:.+]]:2 = iree_linalg_ext.sort dimension(0) outs(%[[COPY]],
// CHECK: %[[VIEW:.+]] = tensor.expand_shape %[[KEYS]]
// CHECK: %[[COMBINED:.+]] = linalg.generic
// CHECK-SAME: ins(%[[VIEW]] : tensor<1x4xi64>) outs(%[[SORT]]#1 : tensor<4xi64>)
// CHECK: iree_tensor_ext.dispatch.tensor.store %[[COMBINED]], %[[INDEX_BINDING]]
// CHECK: iree_tensor_ext.dispatch.tensor.store %[[SORT]]#0, %[[WORKSPACE]]
util.func public @sort_preserves_internal_key_view(
    %keys: tensor<4xi64>, %indices: tensor<4xi64>) -> tensor<4xi64> {
  %result = flow.dispatch.region -> (tensor<4xi64>) {
    %sorted:2 = iree_linalg_ext.sort dimension(0)
        outs(%keys, %indices : tensor<4xi64>, tensor<4xi64>) {
    ^bb0(%lhs: i64, %rhs: i64, %lhs_index: i64, %rhs_index: i64):
      %cmp = arith.cmpi sle, %lhs, %rhs : i64
      iree_linalg_ext.yield %cmp : i1
    } -> tensor<4xi64>, tensor<4xi64>
    %view = tensor.expand_shape %keys [[0, 1]] output_shape [1, 4]
        : tensor<4xi64> into tensor<1x4xi64>
    %combined = linalg.generic {
        indexing_maps = [affine_map<(d0) -> (0, d0)>,
                         affine_map<(d0) -> (d0)>],
        iterator_types = ["parallel"]}
        ins(%view : tensor<1x4xi64>) outs(%sorted#1 : tensor<4xi64>) {
    ^bb0(%key: i64, %index: i64):
      %sum = arith.addi %key, %index : i64
      linalg.yield %sum : i64
    } -> tensor<4xi64>
    flow.return %combined : tensor<4xi64>
  }
  util.return %result : tensor<4xi64>
}

// -----

// Source equality alone does not establish legal reuse for separate init roles.
// CHECK-LABEL: util.func public @sort_shared_comparator_sources
// CHECK: %[[DISPATCH:.+]]:3 = flow.dispatch.workgroups
// CHECK: %[[KEY_BINDING:[a-zA-Z0-9_]+]]: !iree_tensor_ext.dispatch.tensor<readonly:tensor<4xi64>>
// CHECK-SAME: %[[INDEX_BINDING:[a-zA-Z0-9_]+]]: !iree_tensor_ext.dispatch.tensor<readwrite:tensor<4xi64>>
// CHECK-SAME: %[[WORKSPACE0:[a-zA-Z0-9_]+]]: !iree_tensor_ext.dispatch.tensor<readwrite:tensor<4xi64>>
// CHECK-SAME: %[[WORKSPACE1:[a-zA-Z0-9_]+]]: !iree_tensor_ext.dispatch.tensor<readwrite:tensor<4xi64>>
// CHECK: %[[KEYS:.+]] = iree_tensor_ext.dispatch.tensor.load %[[KEY_BINDING]]
// CHECK: %[[DEST0:.+]] = iree_tensor_ext.dispatch.tensor.load %[[WORKSPACE0]]
// CHECK: %[[COPY0:.+]] = linalg.copy ins(%[[KEYS]] : tensor<4xi64>) outs(%[[DEST0]] : tensor<4xi64>)
// CHECK: %[[DEST1:.+]] = iree_tensor_ext.dispatch.tensor.load %[[WORKSPACE1]]
// CHECK: %[[COPY1:.+]] = linalg.copy ins(%[[KEYS]] : tensor<4xi64>) outs(%[[DEST1]] : tensor<4xi64>)
// CHECK: %[[SORT:.+]]:3 = iree_linalg_ext.sort dimension(0) outs(%[[COPY0]], %[[COPY1]],
// CHECK: iree_tensor_ext.dispatch.tensor.store %[[SORT]]#1, %[[WORKSPACE1]]
util.func public @sort_shared_comparator_sources(
    %keys: tensor<4xi64>, %indices: tensor<4xi64>) -> tensor<4xi64> {
  %result = flow.dispatch.region -> (tensor<4xi64>) {
    %sorted:3 = iree_linalg_ext.sort dimension(0)
        outs(%keys, %keys, %indices
             : tensor<4xi64>, tensor<4xi64>, tensor<4xi64>) {
    ^bb0(%lhs0: i64, %rhs0: i64, %lhs1: i64, %rhs1: i64,
         %lhs_index: i64, %rhs_index: i64):
      %cmp0 = arith.cmpi sle, %lhs0, %rhs0 : i64
      %cmp1 = arith.cmpi sle, %lhs1, %rhs1 : i64
      %cmp = arith.andi %cmp0, %cmp1 : i1
      iree_linalg_ext.yield %cmp : i1
    } -> tensor<4xi64>, tensor<4xi64>, tensor<4xi64>
    flow.return %sorted#2 : tensor<4xi64>
  }
  util.return %result : tensor<4xi64>
}

// -----

// CHECK-LABEL: util.func public @sort_trailing_rhs_comparator
// CHECK: %[[DISPATCH:.+]]:2 = flow.dispatch.workgroups
// CHECK: %[[KEY_BINDING:[a-zA-Z0-9_]+]]: !iree_tensor_ext.dispatch.tensor<readonly:tensor<?xf32>>
// CHECK-SAME: %[[WORKSPACE:[a-zA-Z0-9_]+]]: !iree_tensor_ext.dispatch.tensor<readwrite:tensor<?xf32>>
// CHECK: %[[KEYS:.+]] = iree_tensor_ext.dispatch.tensor.load %[[KEY_BINDING]]
// CHECK: %[[DEST:.+]] = iree_tensor_ext.dispatch.tensor.load %[[WORKSPACE]]
// CHECK: %[[COPY:.+]] = linalg.copy ins(%[[KEYS]] : tensor<?xf32>) outs(%[[DEST]] : tensor<?xf32>)
// CHECK: %[[SORT:.+]]:2 = iree_linalg_ext.sort dimension(0) outs(%{{.+}}, %[[COPY]] : tensor<?xi64>, tensor<?xf32>)
// CHECK: iree_tensor_ext.dispatch.tensor.store %[[SORT]]#1, %[[WORKSPACE]]
util.func public @sort_trailing_rhs_comparator(
    %indices: tensor<?xi64>, %keys: tensor<?xf32>) -> tensor<?xi64> {
  %c0 = arith.constant 0 : index
  %n = tensor.dim %keys, %c0 : tensor<?xf32>
  %result = flow.dispatch.region -> (tensor<?xi64>{%n}) {
    %sorted:2 = iree_linalg_ext.sort dimension(0)
        outs(%indices, %keys : tensor<?xi64>, tensor<?xf32>) {
    ^bb0(%lhs_index: i64, %rhs_index: i64, %lhs: f32, %rhs: f32):
      %zero = arith.constant 0.0 : f32
      %cmp = arith.cmpf oge, %rhs, %zero : f32
      iree_linalg_ext.yield %cmp : i1
    } -> tensor<?xi64>, tensor<?xf32>
    flow.return %sorted#0 : tensor<?xi64>
  }
  util.return %result : tensor<?xi64>
}

// -----

// Already returned destinations retain baseline ties.
// CHECK-LABEL: util.func public @sort_returned_keys
// CHECK: %[[DISPATCH:.+]]:2 = flow.dispatch.workgroups
// CHECK-NOT: linalg.copy
// CHECK: iree_linalg_ext.sort dimension(0)
// CHECK-NOT: linalg.copy
// CHECK: util.return %[[DISPATCH]]#0, %[[DISPATCH]]#1
util.func public @sort_returned_keys(%keys: tensor<4xi64>, %indices: tensor<4xi64>)
    -> (tensor<4xi64>, tensor<4xi64>) {
  %result:2 = flow.dispatch.region -> (tensor<4xi64>, tensor<4xi64>) {
    %sorted:2 = iree_linalg_ext.sort dimension(0)
        outs(%keys, %indices : tensor<4xi64>, tensor<4xi64>) {
    ^bb0(%lhs: i64, %rhs: i64, %lhs_index: i64, %rhs_index: i64):
      %cmp = arith.cmpi sle, %lhs, %rhs : i64
      iree_linalg_ext.yield %cmp : i1
    } -> tensor<4xi64>, tensor<4xi64>
    flow.return %sorted#0, %sorted#1 : tensor<4xi64>, tensor<4xi64>
  }
  util.return %result#0, %result#1 : tensor<4xi64>, tensor<4xi64>
}

// -----

// Unused noncomparator lanes remain eligible for existing Sort pruning.
// CHECK-LABEL: util.func public @sort_unused_noncomparator_lane
// CHECK-NOT: linalg.copy
// CHECK: iree_linalg_ext.sort dimension(0) outs(%{{.+}} : tensor<4xi64>)
// CHECK-NOT: linalg.copy
util.func public @sort_unused_noncomparator_lane(
    %keys: tensor<4xi64>, %indices: tensor<4xi64>) -> tensor<4xi64> {
  %result = flow.dispatch.region -> (tensor<4xi64>) {
    %sorted:2 = iree_linalg_ext.sort dimension(0)
        outs(%keys, %indices : tensor<4xi64>, tensor<4xi64>) {
    ^bb0(%lhs: i64, %rhs: i64, %lhs_index: i64, %rhs_index: i64):
      %cmp = arith.cmpi sle, %lhs, %rhs : i64
      iree_linalg_ext.yield %cmp : i1
    } -> tensor<4xi64>, tensor<4xi64>
    flow.return %sorted#0 : tensor<4xi64>
  }
  util.return %result : tensor<4xi64>
}

// -----

// A local key producer is outside this captured-workspace lowering.
// CHECK-LABEL: util.func public @sort_local_keys_unchanged
// CHECK: flow.dispatch.workgroups
// CHECK-NOT: linalg.copy
// CHECK: %[[EMPTY:.+]] = tensor.empty
// CHECK: %[[KEYS:.+]] = linalg.generic
// CHECK-SAME: outs(%[[EMPTY]] : tensor<?xi64>)
// CHECK: iree_linalg_ext.sort dimension(0) outs(%[[KEYS]],
// CHECK-NOT: linalg.copy
util.func public @sort_local_keys_unchanged(%indices: tensor<?xi64>)
    -> tensor<?xi64> {
  %c0 = arith.constant 0 : index
  %n = tensor.dim %indices, %c0 : tensor<?xi64>
  %result = flow.dispatch.region -> (tensor<?xi64>{%n}) {
    %empty = tensor.empty(%n) : tensor<?xi64>
    %keys = linalg.generic {
        indexing_maps = [affine_map<(d0) -> (d0)>],
        iterator_types = ["parallel"]} outs(%empty : tensor<?xi64>) {
    ^bb0(%out: i64):
      %index = linalg.index 0 : index
      %value = arith.index_cast %index : index to i64
      linalg.yield %value : i64
    } -> tensor<?xi64>
    %sorted:2 = iree_linalg_ext.sort dimension(0)
        outs(%keys, %indices : tensor<?xi64>, tensor<?xi64>) {
    ^bb0(%lhs: i64, %rhs: i64, %lhs_index: i64, %rhs_index: i64):
      %cmp = arith.cmpi sle, %lhs, %rhs : i64
      iree_linalg_ext.yield %cmp : i1
    } -> tensor<?xi64>, tensor<?xi64>
    flow.return %sorted#1 : tensor<?xi64>
  }
  util.return %result : tensor<?xi64>
}

// -----

// Tensor-result liveness does not account for mixed buffer side effects.
// CHECK-LABEL: util.func public @sort_mixed_semantics_unchanged
// CHECK: flow.dispatch.workgroups
// CHECK-NOT: linalg.copy
// CHECK: iree_linalg_ext.sort dimension(0) outs(%{{.+}}, %{{.+}}, %{{.+}} : tensor<4xi64>, tensor<4xi64>, memref<4xi64>)
// CHECK-NOT: linalg.copy
util.func public @sort_mixed_semantics_unchanged(
    %keys: tensor<4xi64>, %indices: tensor<4xi64>, %buffer: memref<4xi64>)
    -> tensor<4xi64> {
  %result = flow.dispatch.region -> (tensor<4xi64>) {
    %sorted:2 = iree_linalg_ext.sort dimension(0)
        outs(%keys, %indices, %buffer
             : tensor<4xi64>, tensor<4xi64>, memref<4xi64>) {
    ^bb0(%lhs: i64, %rhs: i64, %lhs_index: i64, %rhs_index: i64,
         %lhs_buffer: i64, %rhs_buffer: i64):
      %cmp = arith.cmpi sle, %lhs, %rhs : i64
      iree_linalg_ext.yield %cmp : i1
    } -> tensor<4xi64>, tensor<4xi64>
    flow.return %sorted#1 : tensor<4xi64>
  }
  util.return %result : tensor<4xi64>
}

// -----

// CHECK-LABEL: util.func public @dead_sort
// CHECK-NOT: linalg.copy
// CHECK-NOT: iree_linalg_ext.sort
// CHECK: util.return
util.func public @dead_sort(%keys: tensor<4xi64>, %indices: tensor<4xi64>) {
  flow.dispatch.region {
    %sorted:2 = iree_linalg_ext.sort dimension(0)
        outs(%keys, %indices : tensor<4xi64>, tensor<4xi64>) {
    ^bb0(%lhs: i64, %rhs: i64, %lhs_index: i64, %rhs_index: i64):
      %cmp = arith.cmpi sle, %lhs, %rhs : i64
      iree_linalg_ext.yield %cmp : i1
    } -> tensor<4xi64>, tensor<4xi64>
    flow.return
  }
  util.return
}

// -----

// CHECK-LABEL: util.func public @shape_only_sort
// CHECK-NOT: linalg.copy
// CHECK-NOT: iree_linalg_ext.sort
// CHECK: util.return
util.func public @shape_only_sort(%keys: tensor<4xi64>, %indices: tensor<4xi64>)
    -> tensor<4xi64> {
  %result = flow.dispatch.region -> (tensor<4xi64>) {
    %sorted:2 = iree_linalg_ext.sort dimension(0)
        outs(%keys, %indices : tensor<4xi64>, tensor<4xi64>) {
    ^bb0(%lhs: i64, %rhs: i64, %lhs_index: i64, %rhs_index: i64):
      %cmp = arith.cmpi sle, %lhs, %rhs : i64
      iree_linalg_ext.yield %cmp : i1
    } -> tensor<4xi64>, tensor<4xi64>
    %c0 = arith.constant 0 : index
    %n = tensor.dim %sorted#1, %c0 : tensor<4xi64>
    %empty = tensor.empty(%n) : tensor<?xi64>
    %one = arith.constant 1 : i64
    %filled = linalg.fill ins(%one : i64) outs(%empty : tensor<?xi64>)
        -> tensor<?xi64>
    %cast = tensor.cast %filled : tensor<?xi64> to tensor<4xi64>
    flow.return %cast : tensor<4xi64>
  }
  util.return %result : tensor<4xi64>
}
