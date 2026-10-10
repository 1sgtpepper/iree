// Source probe: two DPS roles initialize from one writable tensor definition.
#pipeline_layout = #hal.pipeline.layout<constants = 1, bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
func.func @sort_shared_destination() {
  %n = hal.interface.constant.load layout(#pipeline_layout) ordinal(0) : index
  %keys_binding = hal.interface.binding.subspan layout(#pipeline_layout) binding(0)
      : !iree_tensor_ext.dispatch.tensor<readwrite:tensor<?xi64>>{%n}
  %indices_binding = hal.interface.binding.subspan layout(#pipeline_layout) binding(1)
      : !iree_tensor_ext.dispatch.tensor<readwrite:tensor<?xi64>>{%n}
  %extra_binding = hal.interface.binding.subspan layout(#pipeline_layout) binding(2)
      : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?xi64>>{%n}
  %keys = iree_tensor_ext.dispatch.tensor.load %keys_binding,
      offsets = [0], sizes = [%n], strides = [1]
      : !iree_tensor_ext.dispatch.tensor<readwrite:tensor<?xi64>>{%n} -> tensor<?xi64>
  %indices = iree_tensor_ext.dispatch.tensor.load %indices_binding,
      offsets = [0], sizes = [%n], strides = [1]
      : !iree_tensor_ext.dispatch.tensor<readwrite:tensor<?xi64>>{%n} -> tensor<?xi64>
  %sorted:3 = iree_linalg_ext.sort dimension(0)
      outs(%keys, %keys, %indices : tensor<?xi64>, tensor<?xi64>, tensor<?xi64>) {
  ^bb0(%lhs0: i64, %rhs0: i64, %lhs1: i64, %rhs1: i64,
       %lhs_index: i64, %rhs_index: i64):
    %cmp0 = arith.cmpi sle, %lhs0, %rhs0 : i64
    %cmp1 = arith.cmpi sle, %lhs1, %rhs1 : i64
    %cmp = arith.andi %cmp0, %cmp1 : i1
    iree_linalg_ext.yield %cmp : i1
  } -> tensor<?xi64>, tensor<?xi64>, tensor<?xi64>
  iree_tensor_ext.dispatch.tensor.store %sorted#2, %indices_binding,
      offsets = [0], sizes = [%n], strides = [1]
      : tensor<?xi64> -> !iree_tensor_ext.dispatch.tensor<readwrite:tensor<?xi64>>{%n}
  iree_tensor_ext.dispatch.tensor.store %sorted#0, %keys_binding,
      offsets = [0], sizes = [%n], strides = [1]
      : tensor<?xi64> -> !iree_tensor_ext.dispatch.tensor<readwrite:tensor<?xi64>>{%n}
  iree_tensor_ext.dispatch.tensor.store %sorted#1, %extra_binding,
      offsets = [0], sizes = [%n], strides = [1]
      : tensor<?xi64> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?xi64>>{%n}
  return
}

// Two operations read the same original tensor version. A same-operation
// conflict exemption cannot make the earlier Sort overwrite that version.
#two_sort_layout = #hal.pipeline.layout<constants = 1, bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
func.func @two_sorts_shared_destination() {
  %n = hal.interface.constant.load layout(#two_sort_layout) ordinal(0) : index
  %keys_binding = hal.interface.binding.subspan layout(#two_sort_layout) binding(0)
      : !iree_tensor_ext.dispatch.tensor<readwrite:tensor<?xi64>>{%n}
  %indices0_binding = hal.interface.binding.subspan layout(#two_sort_layout) binding(1)
      : !iree_tensor_ext.dispatch.tensor<readwrite:tensor<?xi64>>{%n}
  %indices1_binding = hal.interface.binding.subspan layout(#two_sort_layout) binding(2)
      : !iree_tensor_ext.dispatch.tensor<readwrite:tensor<?xi64>>{%n}
  %extra_binding = hal.interface.binding.subspan layout(#two_sort_layout) binding(3)
      : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?xi64>>{%n}
  %keys = iree_tensor_ext.dispatch.tensor.load %keys_binding,
      offsets = [0], sizes = [%n], strides = [1]
      : !iree_tensor_ext.dispatch.tensor<readwrite:tensor<?xi64>>{%n} -> tensor<?xi64>
  %indices0 = iree_tensor_ext.dispatch.tensor.load %indices0_binding,
      offsets = [0], sizes = [%n], strides = [1]
      : !iree_tensor_ext.dispatch.tensor<readwrite:tensor<?xi64>>{%n} -> tensor<?xi64>
  %indices1 = iree_tensor_ext.dispatch.tensor.load %indices1_binding,
      offsets = [0], sizes = [%n], strides = [1]
      : !iree_tensor_ext.dispatch.tensor<readwrite:tensor<?xi64>>{%n} -> tensor<?xi64>
  %ascending:2 = iree_linalg_ext.sort dimension(0)
      outs(%keys, %indices0 : tensor<?xi64>, tensor<?xi64>) {
  ^bb0(%lhs: i64, %rhs: i64, %lhs_index: i64, %rhs_index: i64):
    %cmp = arith.cmpi sle, %lhs, %rhs : i64
    iree_linalg_ext.yield %cmp : i1
  } -> tensor<?xi64>, tensor<?xi64>
  %descending:2 = iree_linalg_ext.sort dimension(0)
      outs(%keys, %indices1 : tensor<?xi64>, tensor<?xi64>) {
  ^bb0(%lhs: i64, %rhs: i64, %lhs_index: i64, %rhs_index: i64):
    %cmp = arith.cmpi sge, %lhs, %rhs : i64
    iree_linalg_ext.yield %cmp : i1
  } -> tensor<?xi64>, tensor<?xi64>
  iree_tensor_ext.dispatch.tensor.store %ascending#1, %indices0_binding,
      offsets = [0], sizes = [%n], strides = [1]
      : tensor<?xi64> -> !iree_tensor_ext.dispatch.tensor<readwrite:tensor<?xi64>>{%n}
  iree_tensor_ext.dispatch.tensor.store %descending#1, %indices1_binding,
      offsets = [0], sizes = [%n], strides = [1]
      : tensor<?xi64> -> !iree_tensor_ext.dispatch.tensor<readwrite:tensor<?xi64>>{%n}
  iree_tensor_ext.dispatch.tensor.store %ascending#0, %keys_binding,
      offsets = [0], sizes = [%n], strides = [1]
      : tensor<?xi64> -> !iree_tensor_ext.dispatch.tensor<readwrite:tensor<?xi64>>{%n}
  iree_tensor_ext.dispatch.tensor.store %descending#0, %extra_binding,
      offsets = [0], sizes = [%n], strides = [1]
      : tensor<?xi64> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?xi64>>{%n}
  return
}
