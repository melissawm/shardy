// RUN: sdy_opt %s -sdy-insert-explicit-reshards | FileCheck %s

sdy.mesh @mesh = <["x"=4, "y"=2]>

// CHECK-LABEL: func @dot_compatible_ik
func.func @dot_compatible_ik(%arg0: tensor<8x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>}, %arg1: tensor<32x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {}]>}) -> (tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}) {
  // CHECK: stablehlo.dot %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>
  // CHECK-NOT: sdy.reshard
  %0 = stablehlo.dot %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>, sdy.sharding_rule = #sdy.op_sharding_rule<([i, k],[k, j])->([i, j]) {i=8, j=16, k=32}>} : (tensor<8x32xf32>, tensor<32x16xf32>) -> tensor<8x16xf32>
  return %0 : tensor<8x16xf32>
}

// CHECK-LABEL: func @dot_funcop_result_sharding_does_not_match
func.func @dot_funcop_result_sharding_does_not_match(%arg0: tensor<8x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>}, %arg1: tensor<32x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {}]>}) -> tensor<8x16xf32> {
  // TODO(enver): This should actually reshard for the result.
  %0 = stablehlo.dot %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>, sdy.sharding_rule = #sdy.op_sharding_rule<([i, k],[k, j])->([i, j]) {i=8, j=16, k=32}>} : (tensor<8x32xf32>, tensor<32x16xf32>) -> tensor<8x16xf32>
  return %0 : tensor<8x16xf32>
}

// CHECK-LABEL: func @dot_funcop_result_sharding_does_not_match_and_reshard_result_twice
// CHECK-NEXT: %[[DOT:.*]]  = stablehlo.dot %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}]>]>, sdy.sharding_rule = #sdy.op_sharding_rule<([i, k], [k, j])->([i, j]) {i=8, j=16, k=32}>} : (tensor<8x32xf32>, tensor<32x16xf32>) -> tensor<8x16xf32>
// CHECK-NEXT: %[[RESHARD:.*]] = sdy.reshard %[[DOT]] <@mesh, [{"y"}, {}]> : tensor<8x16xf32>
// CHECK-NEXT: return %[[RESHARD]] : tensor<8x16xf32>
func.func @dot_funcop_result_sharding_does_not_match_and_reshard_result_twice(%arg0: tensor<8x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>}, %arg1: tensor<32x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {}]>}) -> (tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}) {
  // TODO(enver): This should actually reshard for the result, twice.
  %0 = stablehlo.dot %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}, {}]>]>, sdy.sharding_rule = #sdy.op_sharding_rule<([i, k],[k, j])->([i, j]) {i=8, j=16, k=32}>} : (tensor<8x32xf32>, tensor<32x16xf32>) -> tensor<8x16xf32>
  return %0 : tensor<8x16xf32>
}

// CHECK-LABEL: func @dot_incompatible_subaxis_no_overlap
// CHECK-NEXT: %[[RESHARD1:.*]] = sdy.reshard %arg0 <@mesh, [{}, {"y"}]> : tensor<8x32xf32>
// CHECK-NEXT: %[[DOT:.*]] = stablehlo.dot %[[RESHARD1]], %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}]>]>, sdy.sharding_rule = #sdy.op_sharding_rule<([i, k], [k, j])->([i, j]) {i=8, j=16, k=32}>} : (tensor<8x32xf32>, tensor<32x16xf32>) -> tensor<8x16xf32>
// CHECK-NEXT: %[[RESHARD2:.*]] = sdy.reshard %[[DOT]] <@mesh, [{}, {"x":(2)2}]> : tensor<8x16xf32>
// CHECK-NEXT: return %[[RESHARD2]] : tensor<8x16xf32>
func.func @dot_incompatible_subaxis_no_overlap(%arg0: tensor<8x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x":(1)2}, {"y"}]>}, %arg1: tensor<32x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {}]>}) -> (tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"x":(2)2}]>}) {
  %0 = stablehlo.dot %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"x":(2)2}]>]>, sdy.sharding_rule = #sdy.op_sharding_rule<([i, k],[k, j])->([i, j]) {i=8, j=16, k=32}>} : (tensor<8x32xf32>, tensor<32x16xf32>) -> tensor<8x16xf32>
  return %0 : tensor<8x16xf32>
}

// CHECK-LABEL: func @dot_compatible_jk
func.func @dot_compatible_jk(%arg0: tensor<8x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>}, %arg1: tensor<32x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {"x"}]>}) -> (tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"x"}]>}) {
  // CHECK: stablehlo.dot %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"x"}]>]>
  // CHECK-NOT: sdy.reshard
  %0 = stablehlo.dot %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"x"}]>]>, sdy.sharding_rule = #sdy.op_sharding_rule<([i, k],[k, j])->([i, j]) {i=8, j=16, k=32}>} : (tensor<8x32xf32>, tensor<32x16xf32>) -> tensor<8x16xf32>
  return %0 : tensor<8x16xf32>
}

// CHECK-LABEL: func @dot_compatible_k
func.func @dot_compatible_k(%arg0: tensor<8x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>}, %arg1: tensor<32x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {}]>}) -> tensor<8x16xf32> {
  // CHECK: stablehlo.dot %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}]>]>
  // CHECK-NOT: sdy.reshard
  %0 = stablehlo.dot %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}]>]>, sdy.sharding_rule = #sdy.op_sharding_rule<([i, k],[k, j])->([i, j]) {i=8, j=16, k=32}>} : (tensor<8x32xf32>, tensor<32x16xf32>) -> tensor<8x16xf32>
  return %0 : tensor<8x16xf32>
}

// CHECK-LABEL: func @dot_compatible_empty
func.func @dot_compatible_empty(%arg0: tensor<8x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}, %arg1: tensor<32x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}) -> tensor<8x16xf32> {
  // CHECK: stablehlo.dot %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}]>]>
  // CHECK-NOT: sdy.reshard
  %0 = stablehlo.dot %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}]>]>, sdy.sharding_rule = #sdy.op_sharding_rule<([i, k],[k, j])->([i, j]) {i=8, j=16, k=32}>} : (tensor<8x32xf32>, tensor<32x16xf32>) -> tensor<8x16xf32>
  return %0 : tensor<8x16xf32>
}

// CHECK-LABEL: func @dot_compatible_contracting_dim_empty
func.func @dot_compatible_contracting_dim_empty(%arg0: tensor<8x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}, %arg1: tensor<32x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>}) -> (tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>}) {
  // CHECK: stablehlo.dot %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {"y"}]>]>
  // CHECK-NOT: sdy.reshard
  %0 = stablehlo.dot %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {"y"}]>]>, sdy.sharding_rule = #sdy.op_sharding_rule<([i, k],[k, j])->([i, j]) {i=8, j=16, k=32}>} : (tensor<8x32xf32>, tensor<32x16xf32>) -> tensor<8x16xf32>
  return %0 : tensor<8x16xf32>
}

// CHECK-LABEL: func @dot_incompatible_same_non_contracting_dims_out_empty
// CHECK-NEXT: %[[RESHARD1:.*]] = sdy.reshard %arg0 <@mesh, [{}, {"y"}]> : tensor<8x32xf32>
// CHECK-NEXT: %[[RESHARD2:.*]] = sdy.reshard %arg1 <@mesh, [{"y"}, {}]> : tensor<32x16xf32>
// CHECK-NEXT: %[[DOT:.*]] = stablehlo.dot %[[RESHARD1]], %[[RESHARD2]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}]>]>, sdy.sharding_rule = #sdy.op_sharding_rule<([i, k], [k, j])->([i, j]) {i=8, j=16, k=32}>} : (tensor<8x32xf32>, tensor<32x16xf32>) -> tensor<8x16xf32>
// CHECK-NEXT: return %[[DOT]] : tensor<8x16xf32>
func.func @dot_incompatible_same_non_contracting_dims_out_empty(%arg0: tensor<8x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>}, %arg1: tensor<32x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {"x"}]>}) -> tensor<8x16xf32> {
  // TODO(enver): Another solution would be to reshard (fully resplicate) the result while keeping the sharding of LHS (or RHS) on the dot.
  %0 = stablehlo.dot %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}]>]>, sdy.sharding_rule = #sdy.op_sharding_rule<([i, k],[k, j])->([i, j]) {i=8, j=16, k=32}>} : (tensor<8x32xf32>, tensor<32x16xf32>) -> tensor<8x16xf32>
  return %0 : tensor<8x16xf32>
}

// CHECK-LABEL: func @dot_incompatible_same_non_contracting_dims_out_i
// CHECK-NEXT: %[[RESHARD:.*]] = sdy.reshard %arg1 <@mesh, [{"y"}, {}]> : tensor<32x16xf32>
// CHECK-NEXT: %[[DOT:.*]] = stablehlo.dot %arg0, %[[RESHARD]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>, sdy.sharding_rule = #sdy.op_sharding_rule<([i, k], [k, j])->([i, j]) {i=8, j=16, k=32}>} : (tensor<8x32xf32>, tensor<32x16xf32>) -> tensor<8x16xf32>
// CHECK-NEXT: return %[[DOT]] : tensor<8x16xf32>
func.func @dot_incompatible_same_non_contracting_dims_out_i(%arg0: tensor<8x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>}, %arg1: tensor<32x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {"x"}]>}) -> (tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}) {
  %0 = stablehlo.dot %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>, sdy.sharding_rule = #sdy.op_sharding_rule<([i, k],[k, j])->([i, j]) {i=8, j=16, k=32}>} : (tensor<8x32xf32>, tensor<32x16xf32>) -> tensor<8x16xf32>
  return %0 : tensor<8x16xf32>
}

// CHECK-LABEL: func @dot_incompatible_same_non_contracting_dims_out_j
// CHECK-NEXT: %[[RESHARD:.*]] = sdy.reshard %arg0 <@mesh, [{}, {"y"}]> : tensor<8x32xf32>
// CHECK-NEXT: %[[DOT:.*]] = stablehlo.dot %[[RESHARD]], %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"x"}]>]>, sdy.sharding_rule = #sdy.op_sharding_rule<([i, k], [k, j])->([i, j]) {i=8, j=16, k=32}>} : (tensor<8x32xf32>, tensor<32x16xf32>) -> tensor<8x16xf32>
// CHECK-NEXT: return %[[DOT]] : tensor<8x16xf32>
func.func @dot_incompatible_same_non_contracting_dims_out_j(%arg0: tensor<8x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>}, %arg1: tensor<32x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {"x"}]>}) -> (tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"x"}]>}) {
  %0 = stablehlo.dot %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"x"}]>]>, sdy.sharding_rule = #sdy.op_sharding_rule<([i, k],[k, j])->([i, j]) {i=8, j=16, k=32}>} : (tensor<8x32xf32>, tensor<32x16xf32>) -> tensor<8x16xf32>
  return %0 : tensor<8x16xf32>
}

// CHECK-LABEL: func @dot_incompatible_out_empty
// CHECK-NEXT: %[[RESHARD:.*]] = sdy.reshard %arg0 <@mesh, [{}, {"y"}]> : tensor<8x32xf32>
// CHECK-NEXT: %[[DOT:.*]] = stablehlo.dot %[[RESHARD]], %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}]>]>, sdy.sharding_rule = #sdy.op_sharding_rule<([i, k], [k, j])->([i, j]) {i=8, j=16, k=32}>} : (tensor<8x32xf32>, tensor<32x16xf32>) -> tensor<8x16xf32>
// CHECK-NEXT: return %[[DOT]] : tensor<8x16xf32>
func.func @dot_incompatible_out_empty(%arg0: tensor<8x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>}, %arg1: tensor<32x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {}]>}) -> tensor<8x16xf32> {
  %0 = stablehlo.dot %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}]>]>, sdy.sharding_rule = #sdy.op_sharding_rule<([i, k],[k, j])->([i, j]) {i=8, j=16, k=32}>} : (tensor<8x32xf32>, tensor<32x16xf32>) -> tensor<8x16xf32>
  return %0 : tensor<8x16xf32>
}

// CHECK-LABEL: func @dot_incompatible_i_empty
// CHECK-NEXT: %[[DOT:.*]] = stablehlo.dot %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}]>]>, sdy.sharding_rule = #sdy.op_sharding_rule<([i, k], [k, j])->([i, j]) {i=8, j=16, k=32}>} : (tensor<8x32xf32>, tensor<32x16xf32>) -> tensor<8x16xf32>
// CHECK-NEXT: %[[RESHARD:.*]] = sdy.reshard %[[DOT]] <@mesh, [{"x"}, {}]> : tensor<8x16xf32>
// CHECK-NEXT: return %[[RESHARD]] : tensor<8x16xf32>
func.func @dot_incompatible_i_empty(%arg0: tensor<8x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>}, %arg1: tensor<32x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {}]>}) -> (tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}) {
  %0 = stablehlo.dot %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>, sdy.sharding_rule = #sdy.op_sharding_rule<([i, k],[k, j])->([i, j]) {i=8, j=16, k=32}>} : (tensor<8x32xf32>, tensor<32x16xf32>) -> tensor<8x16xf32>
  return %0 : tensor<8x16xf32>
}

// CHECK-LABEL: func @dot_incompatible_same_factor_for_contracting_dim_and_output_i
// CHECK-NEXT: %[[DOT:.*]] = stablehlo.dot %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}]>]>, sdy.sharding_rule = #sdy.op_sharding_rule<([i, k], [k, j])->([i, j]) {i=8, j=16, k=32}>} : (tensor<8x32xf32>, tensor<32x16xf32>) -> tensor<8x16xf32>
// CHECK-NEXT: %[[RESHARD:.*]] = sdy.reshard %[[DOT]] <@mesh, [{"y"}, {}]> : tensor<8x16xf32>
// CHECK-NEXT: return %[[RESHARD]] : tensor<8x16xf32>
func.func @dot_incompatible_same_factor_for_contracting_dim_and_output_i(%arg0: tensor<8x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>}, %arg1: tensor<32x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {}]>}) -> (tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {}]>}) {
  %0 = stablehlo.dot %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}, {}]>]>, sdy.sharding_rule = #sdy.op_sharding_rule<([i, k],[k, j])->([i, j]) {i=8, j=16, k=32}>} : (tensor<8x32xf32>, tensor<32x16xf32>) -> tensor<8x16xf32>
  return %0 : tensor<8x16xf32>
}

// CHECK-LABEL: func @dot_incompatible_same_factor_for_contracting_dim_and_output_j
// CHECK-NEXT: %[[DOT:.*]] = stablehlo.dot %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}]>]>, sdy.sharding_rule = #sdy.op_sharding_rule<([i, k], [k, j])->([i, j]) {i=8, j=16, k=32}>} : (tensor<8x32xf32>, tensor<32x16xf32>) -> tensor<8x16xf32>
// CHECK-NEXT: %[[RESHARD:.*]] = sdy.reshard %[[DOT]] <@mesh, [{}, {"y"}]> : tensor<8x16xf32>
// CHECK-NEXT: return %[[RESHARD]] : tensor<8x16xf32>
func.func @dot_incompatible_same_factor_for_contracting_dim_and_output_j(%arg0: tensor<8x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>}, %arg1: tensor<32x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {}]>}) -> (tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>}) {
  %0 = stablehlo.dot %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"y"}]>]>, sdy.sharding_rule = #sdy.op_sharding_rule<([i, k],[k, j])->([i, j]) {i=8, j=16, k=32}>} : (tensor<8x32xf32>, tensor<32x16xf32>) -> tensor<8x16xf32>
  return %0 : tensor<8x16xf32>
}

// CHECK-LABEL: func @dot_incompatible_i_mismatch
// CHECK-NEXT: %[[RESHARD1:.*]] = sdy.reshard %arg0 <@mesh, [{}, {"y"}]> : tensor<8x32xf32>
// CHECK-NEXT: %[[DOT:.*]] = stablehlo.dot %[[RESHARD1]], %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}]>]>, sdy.sharding_rule = #sdy.op_sharding_rule<([i, k], [k, j])->([i, j]) {i=8, j=16, k=32}>} : (tensor<8x32xf32>, tensor<32x16xf32>) -> tensor<8x16xf32>
// CHECK-NEXT: %[[RESHARD2:.*]] = sdy.reshard %[[DOT]] <@mesh, [{"y"}, {}]> : tensor<8x16xf32>
// CHECK-NEXT: return %[[RESHARD2]] : tensor<8x16xf32>
func.func @dot_incompatible_i_mismatch(%arg0: tensor<8x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>}, %arg1: tensor<32x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {}]>}) -> (tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {}]>}) {
  %0 = stablehlo.dot %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}, {}]>]>, sdy.sharding_rule = #sdy.op_sharding_rule<([i, k],[k, j])->([i, j]) {i=8, j=16, k=32}>} : (tensor<8x32xf32>, tensor<32x16xf32>) -> tensor<8x16xf32>
  return %0 : tensor<8x16xf32>
}

// CHECK-LABEL: func @dot_incompatible_in_out_mismatch_same_axis_on_different_factors
// CHECK-NEXT: %[[RESHARD1:.*]] = sdy.reshard %arg0 <@mesh, [{}, {"y"}]> : tensor<8x32xf32>
// CHECK-NEXT: %[[DOT:.*]] = stablehlo.dot %[[RESHARD1]], %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}]>]>, sdy.sharding_rule = #sdy.op_sharding_rule<([i, k], [k, j])->([i, j]) {i=8, j=16, k=32}>} : (tensor<8x32xf32>, tensor<32x16xf32>) -> tensor<8x16xf32>
// CHECK-NEXT: %[[RESHARD2:.*]] = sdy.reshard %[[DOT]] <@mesh, [{}, {"x"}]> : tensor<8x16xf32>
// CHECK-NEXT: return %[[RESHARD2]] : tensor<8x16xf32>
func.func @dot_incompatible_in_out_mismatch_same_axis_on_different_factors(%arg0: tensor<8x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>}, %arg1: tensor<32x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {}]>}) -> (tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"x"}]>}) {
  // TODO(enver): A better solution would be dot and then reshard only on the result.
  %0 = stablehlo.dot %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"x"}]>]>, sdy.sharding_rule = #sdy.op_sharding_rule<([i, k],[k, j])->([i, j]) {i=8, j=16, k=32}>} : (tensor<8x32xf32>, tensor<32x16xf32>) -> tensor<8x16xf32>
  return %0 : tensor<8x16xf32>
}

// CHECK-LABEL: func @dot_incompatible_in_out_mismatch_i_j_swapped
// CHECK-NEXT: %[[RESHARD1:.*]] = sdy.reshard %arg0 <@mesh, [{}, {}]> : tensor<8x32xf32>
// CHECK-NEXT: %[[RESHARD2:.*]] = sdy.reshard %arg1 <@mesh, [{}, {}]> : tensor<32x16xf32>
// CHECK-NEXT: %[[DOT:.*]] = stablehlo.dot %[[RESHARD1]], %[[RESHARD2]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}]>]>, sdy.sharding_rule = #sdy.op_sharding_rule<([i, k], [k, j])->([i, j]) {i=8, j=16, k=32}>} : (tensor<8x32xf32>, tensor<32x16xf32>) -> tensor<8x16xf32>
// CHECK-NEXT: %[[RESHARD3:.*]] = sdy.reshard %[[DOT]] <@mesh, [{"y"}, {"x"}]> : tensor<8x16xf32>
// CHECK-NEXT: return %[[RESHARD3]] : tensor<8x16xf32>
func.func @dot_incompatible_in_out_mismatch_i_j_swapped(%arg0: tensor<8x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}, %arg1: tensor<32x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>}) -> (tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {"x"}]>}) {
  %0 = stablehlo.dot %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}, {"x"}]>]>, sdy.sharding_rule = #sdy.op_sharding_rule<([i, k],[k, j])->([i, j]) {i=8, j=16, k=32}>} : (tensor<8x32xf32>, tensor<32x16xf32>) -> tensor<8x16xf32>
  return %0 : tensor<8x16xf32>
}

// CHECK-LABEL: func @dot_incompatible_sub_axis_overlaps
// CHECK-NEXT: %[[RESHARD1:.*]] = sdy.reshard %arg0 <@mesh, [{}, {"y"}]> : tensor<8x32xf32>
// CHECK-NEXT: %[[DOT:.*]] = stablehlo.dot %[[RESHARD1]], %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}]>]>, sdy.sharding_rule = #sdy.op_sharding_rule<([i, k], [k, j])->([i, j]) {i=8, j=16, k=32}>} : (tensor<8x32xf32>, tensor<32x16xf32>) -> tensor<8x16xf32>
// CHECK-NEXT: %[[RESHARD2:.*]] = sdy.reshard %[[DOT]] <@mesh, [{}, {"x"}]> : tensor<8x16xf32>
// CHECK-NEXT: return %[[RESHARD2]] : tensor<8x16xf32>
func.func @dot_incompatible_sub_axis_overlaps(%arg0: tensor<8x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x":(2)2}, {"y"}]>}, %arg1: tensor<32x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {}]>}) -> (tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"x"}]>}) {
  %0 = stablehlo.dot %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"x"}]>]>, sdy.sharding_rule = #sdy.op_sharding_rule<([i, k],[k, j])->([i, j]) {i=8, j=16, k=32}>} : (tensor<8x32xf32>, tensor<32x16xf32>) -> tensor<8x16xf32>
  return %0 : tensor<8x16xf32>
}

// CHECK-LABEL: func @dot_reshard_is_local
func.func @dot_reshard_is_local(%arg0: tensor<8x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>}, %arg1: tensor<32x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {"x"}]>}) -> (tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}) {
  %0 = stablehlo.negate %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}, {"x"}]>]>} : tensor<32x16xf32>
  // CHECK: %[[RESHARD1:.*]] = sdy.reshard %0 <@mesh, [{"y"}, {}]> : tensor<32x16xf32>
  // CHECK-NEXT: %[[DOT:.*]] = stablehlo.dot %arg0, %[[RESHARD1]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}]>]>, sdy.sharding_rule = #sdy.op_sharding_rule<([i, k], [k, j])->([i, j]) {i=8, j=16, k=32}>} : (tensor<8x32xf32>, tensor<32x16xf32>) -> tensor<8x16xf32>
  // CHECK-NEXT: %[[RESHARD2:.*]] = sdy.reshard %[[DOT]] <@mesh, [{"x"}, {}]> : tensor<8x16xf32>
  // CHECK-NEXT: %[[NEGATE:.*]] = stablehlo.negate %[[RESHARD2]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>} : tensor<8x16xf32>
  // CHECK-NEXT: return %[[NEGATE]] : tensor<8x16xf32>
  %1 = stablehlo.dot %arg0, %0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>, sdy.sharding_rule = #sdy.op_sharding_rule<([i, k],[k, j])->([i, j]) {i=8, j=16, k=32}>} : (tensor<8x32xf32>, tensor<32x16xf32>) -> tensor<8x16xf32>
  %2 = stablehlo.negate %1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>} : tensor<8x16xf32>
  return %2 : tensor<8x16xf32>
}

// CHECK-LABEL: func @dot_reshard_does_not_change_input_sharding
// CHECK-NEXT: %[[RESHARD1:.*]] = sdy.reshard %arg1 <@mesh, [{"y"}, {}]> : tensor<32x16xf32>
// CHECK-NEXT: %[[DOT:.*]] = stablehlo.dot %arg0, %[[RESHARD1]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}]>]>, sdy.sharding_rule = #sdy.op_sharding_rule<([i, k], [k, j])->([i, j]) {i=8, j=16, k=32}>} : (tensor<8x32xf32>, tensor<32x16xf32>) -> tensor<8x16xf32>
// CHECK-NEXT: %[[RESHARD2:.*]] = sdy.reshard %[[DOT]] <@mesh, [{"x"}, {}]> : tensor<8x16xf32>
// CHECK-NEXT: %[[NEGATE:.*]] = stablehlo.negate %[[RESHARD2]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>} : tensor<8x16xf32>
// CHECK-NEXT: return %[[NEGATE]] : tensor<8x16xf32>
func.func @dot_reshard_does_not_change_input_sharding(%arg0: tensor<8x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>}, %arg1: tensor<32x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {"x"}]>}) -> (tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}) {
  %0 = stablehlo.dot %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>, sdy.sharding_rule = #sdy.op_sharding_rule<([i, k],[k, j])->([i, j]) {i=8, j=16, k=32}>} : (tensor<8x32xf32>, tensor<32x16xf32>) -> tensor<8x16xf32>
  %1 = stablehlo.negate %0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>} : tensor<8x16xf32>
  return %1 : tensor<8x16xf32>
}

// CHECK-LABEL: func @dot_without_sharding_rule
// CHECK-NEXT: %[[RESHARD1:.*]] = sdy.reshard %arg1 <@mesh, [{"y"}, {}]> : tensor<32x16xf32>
// CHECK-NEXT: %[[DOT:.*]] = stablehlo.dot %arg0, %[[RESHARD1]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}]>]>} : (tensor<8x32xf32>, tensor<32x16xf32>) -> tensor<8x16xf32>
// CHECK-NEXT: %[[RESHARD2:.*]] = sdy.reshard %[[DOT]] <@mesh, [{"x"}, {}]> : tensor<8x16xf32>
// CHECK-NEXT: return %[[RESHARD2]] : tensor<8x16xf32>
func.func @dot_without_sharding_rule(%arg0: tensor<8x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>}, %arg1: tensor<32x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {"x"}]>}) -> (tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}) {
  %0 = stablehlo.dot %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>} : (tensor<8x32xf32>, tensor<32x16xf32>) -> tensor<8x16xf32>
  return %0 : tensor<8x16xf32>
}
