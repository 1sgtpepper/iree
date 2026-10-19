// The second matrix product updates the first product's still observable result.
func.func @dependent_matmul_observed(%input: tensor<?x?xf32>)
    -> (tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %zero = arith.constant 0.0 : f32
  %rows = tensor.dim %input, %c0 : tensor<?x?xf32>
  %cols = tensor.dim %input, %c1 : tensor<?x?xf32>
  %results:2 = flow.dispatch.region ->
      (tensor<?x?xf32>{%rows, %cols}, tensor<?x?xf32>{%rows, %cols}) {
    %empty = tensor.empty(%rows, %cols) : tensor<?x?xf32>
    %initial = linalg.fill ins(%zero : f32) outs(%empty : tensor<?x?xf32>) -> tensor<?x?xf32>
    %first = linalg.matmul ins(%input, %input : tensor<?x?xf32>, tensor<?x?xf32>)
        outs(%initial : tensor<?x?xf32>) -> tensor<?x?xf32>
    %second = linalg.matmul ins(%input, %input : tensor<?x?xf32>, tensor<?x?xf32>)
        outs(%first : tensor<?x?xf32>) -> tensor<?x?xf32>
    flow.return %second, %first : tensor<?x?xf32>, tensor<?x?xf32>
  }
  return %results#0, %results#1, %input
      : tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>
}
