#id = affine_map<(d0, d1) -> (d0, d1)>

// Keep both data versions of an ordinary DPS update visible across fusion.
func.func @dependent_generic_observed(%input: tensor<?x?xf32>)
    -> (tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %one = arith.constant 1.0 : f32
  %two = arith.constant 2.0 : f32
  %rows = tensor.dim %input, %c0 : tensor<?x?xf32>
  %cols = tensor.dim %input, %c1 : tensor<?x?xf32>
  %results:2 = flow.dispatch.region ->
      (tensor<?x?xf32>{%rows, %cols}, tensor<?x?xf32>{%rows, %cols}) {
    %first = linalg.generic {indexing_maps = [#id],
        iterator_types = ["parallel", "parallel"]} outs(%input : tensor<?x?xf32>) {
    ^bb0(%old: f32):
      %next = arith.addf %old, %one : f32
      linalg.yield %next : f32
    } -> tensor<?x?xf32>
    %held = iree_codegen.fusion_barrier %first : tensor<?x?xf32>
    %second = linalg.generic {indexing_maps = [#id],
        iterator_types = ["parallel", "parallel"]} outs(%held : tensor<?x?xf32>) {
    ^bb0(%old: f32):
      %next = arith.mulf %old, %two : f32
      linalg.yield %next : f32
    } -> tensor<?x?xf32>
    flow.return %second, %first : tensor<?x?xf32>, tensor<?x?xf32>
  }
  return %results#0, %results#1, %input
      : tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>
}
