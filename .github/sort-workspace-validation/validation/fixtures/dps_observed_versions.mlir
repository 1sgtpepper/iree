#id = affine_map<(d0) -> (d0)>

// A DPS update of an observed input is the ordinary control for Sort.
func.func @single_observed(%input: tensor<?xi64>)
    -> (tensor<?xi64>, tensor<?xi64>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : i64
  %n = tensor.dim %input, %c0 : tensor<?xi64>
  %result = flow.dispatch.region -> (tensor<?xi64>{%n}) {
    %updated = linalg.generic {indexing_maps = [#id],
        iterator_types = ["parallel"]} outs(%input : tensor<?xi64>) {
    ^bb0(%old: i64):
      %next = arith.addi %old, %c1 : i64
      linalg.yield %next : i64
    } -> tensor<?xi64>
    flow.return %updated : tensor<?xi64>
  }
  return %result, %input : tensor<?xi64>, tensor<?xi64>
}

// The earlier data version is also read after a dependent DPS update.
func.func @dependent_observed(%input: tensor<?xi64>)
    -> (tensor<?xi64>, tensor<?xi64>, tensor<?xi64>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : i64
  %n = tensor.dim %input, %c0 : tensor<?xi64>
  %results:2 = flow.dispatch.region ->
      (tensor<?xi64>{%n}, tensor<?xi64>{%n}) {
    %first = linalg.generic {indexing_maps = [#id],
        iterator_types = ["parallel"]} outs(%input : tensor<?xi64>) {
    ^bb0(%old: i64):
      %next = arith.addi %old, %c1 : i64
      linalg.yield %next : i64
    } -> tensor<?xi64>
    %second = linalg.generic {indexing_maps = [#id],
        iterator_types = ["parallel"]} outs(%first : tensor<?xi64>) {
    ^bb0(%old: i64):
      %next = arith.addi %old, %c1 : i64
      linalg.yield %next : i64
    } -> tensor<?xi64>
    flow.return %second, %first : tensor<?xi64>, tensor<?xi64>
  }
  return %results#0, %results#1, %input
      : tensor<?xi64>, tensor<?xi64>, tensor<?xi64>
}
