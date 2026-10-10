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
