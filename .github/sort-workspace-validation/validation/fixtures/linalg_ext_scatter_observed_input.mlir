// A separate LinalgExt DPS op tests whether an observable init version is
// preserved by the full dispatch and bufferization pipeline.
func.func @scatter_observed(%original: tensor<8xi32>, %updates: tensor<3xi32>,
                            %indices: tensor<3x1xi32>)
    -> (tensor<8xi32>, tensor<8xi32>) {
  %result = iree_linalg_ext.scatter dimension_map = [0] unique_indices(true)
      ins(%updates, %indices : tensor<3xi32>, tensor<3x1xi32>)
      outs(%original : tensor<8xi32>) {
  ^bb0(%update: i32, %old: i32):
    iree_linalg_ext.yield %update : i32
  } -> tensor<8xi32>
  return %result, %original : tensor<8xi32>, tensor<8xi32>
}
