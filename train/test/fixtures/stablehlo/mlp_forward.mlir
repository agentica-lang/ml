module {
  func.func public @main(%arg0: tensor<4x3xf32>, %arg1: tensor<3x5xf32>, %arg2: tensor<5xf32>, %arg3: tensor<5x2xf32>, %arg4: tensor<2xf32>) -> (tensor<4x2xf32>) {
    %0 = "stablehlo.dot_general"(%arg0, %arg1) {dot_dimension_numbers = #stablehlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>} : (tensor<4x3xf32>, tensor<3x5xf32>) -> (tensor<4x5xf32>)
    %1 = "stablehlo.broadcast_in_dim"(%arg2) {broadcast_dimensions = array<i64: 1>} : (tensor<5xf32>) -> (tensor<4x5xf32>)
    %2 = "stablehlo.add"(%0, %1) : (tensor<4x5xf32>, tensor<4x5xf32>) -> (tensor<4x5xf32>)
    %3 = "stablehlo.multiply"(%2, %2) : (tensor<4x5xf32>, tensor<4x5xf32>) -> (tensor<4x5xf32>)
    %4 = "stablehlo.multiply"(%3, %2) : (tensor<4x5xf32>, tensor<4x5xf32>) -> (tensor<4x5xf32>)
    %5 = "stablehlo.constant"() {value = dense<0.044715> : tensor<4x5xf32>} : () -> (tensor<4x5xf32>)
    %6 = "stablehlo.multiply"(%5, %4) : (tensor<4x5xf32>, tensor<4x5xf32>) -> (tensor<4x5xf32>)
    %7 = "stablehlo.add"(%2, %6) : (tensor<4x5xf32>, tensor<4x5xf32>) -> (tensor<4x5xf32>)
    %8 = "stablehlo.constant"() {value = dense<0.7978845608028654> : tensor<4x5xf32>} : () -> (tensor<4x5xf32>)
    %9 = "stablehlo.multiply"(%8, %7) : (tensor<4x5xf32>, tensor<4x5xf32>) -> (tensor<4x5xf32>)
    %10 = "stablehlo.tanh"(%9) : (tensor<4x5xf32>) -> (tensor<4x5xf32>)
    %11 = "stablehlo.constant"() {value = dense<1.0> : tensor<4x5xf32>} : () -> (tensor<4x5xf32>)
    %12 = "stablehlo.add"(%11, %10) : (tensor<4x5xf32>, tensor<4x5xf32>) -> (tensor<4x5xf32>)
    %13 = "stablehlo.constant"() {value = dense<0.5> : tensor<4x5xf32>} : () -> (tensor<4x5xf32>)
    %14 = "stablehlo.multiply"(%13, %2) : (tensor<4x5xf32>, tensor<4x5xf32>) -> (tensor<4x5xf32>)
    %15 = "stablehlo.multiply"(%14, %12) : (tensor<4x5xf32>, tensor<4x5xf32>) -> (tensor<4x5xf32>)
    %16 = "stablehlo.dot_general"(%15, %arg3) {dot_dimension_numbers = #stablehlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>} : (tensor<4x5xf32>, tensor<5x2xf32>) -> (tensor<4x2xf32>)
    %17 = "stablehlo.broadcast_in_dim"(%arg4) {broadcast_dimensions = array<i64: 1>} : (tensor<2xf32>) -> (tensor<4x2xf32>)
    %18 = "stablehlo.add"(%16, %17) : (tensor<4x2xf32>, tensor<4x2xf32>) -> (tensor<4x2xf32>)
    func.return %18 : tensor<4x2xf32>
  }
}
