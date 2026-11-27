module {
  func.func public @main(%arg0: tensor<3x5xf32>, %arg1: tensor<5xf32>, %arg2: tensor<5x3xf32>, %arg3: tensor<3xf32>, %arg4: tensor<3x5xf32>, %arg5: tensor<5xf32>, %arg6: tensor<5x3xf32>, %arg7: tensor<3xf32>, %arg8: tensor<3x5xf32>, %arg9: tensor<5xf32>, %arg10: tensor<5x3xf32>, %arg11: tensor<3xf32>, %arg12: tensor<7xf32>, %arg13: tensor<4x3xf32>, %arg14: tensor<4xi32>) -> (tensor<f32>, tensor<3x5xf32>, tensor<5xf32>, tensor<5x3xf32>, tensor<3xf32>, tensor<3x5xf32>, tensor<5xf32>, tensor<5x3xf32>, tensor<3xf32>, tensor<3x5xf32>, tensor<5xf32>, tensor<5x3xf32>, tensor<3xf32>) {
    %0 = "stablehlo.dot_general"(%arg13, %arg0) {dot_dimension_numbers = #stablehlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>} : (tensor<4x3xf32>, tensor<3x5xf32>) -> (tensor<4x5xf32>)
    %1 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<5xf32>) -> tensor<4x5xf32>
    %2 = stablehlo.add %0, %1 : tensor<4x5xf32>
    %3 = "stablehlo.constant"() {value = dense<0.0> : tensor<4x5xf32>} : () -> (tensor<4x5xf32>)
    %4 = stablehlo.maximum %2, %3 : tensor<4x5xf32>
    %5 = "stablehlo.dot_general"(%4, %arg2) {dot_dimension_numbers = #stablehlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>} : (tensor<4x5xf32>, tensor<5x3xf32>) -> (tensor<4x3xf32>)
    %6 = stablehlo.broadcast_in_dim %arg3, dims = [1] : (tensor<3xf32>) -> tensor<4x3xf32>
    %7 = stablehlo.add %5, %6 : tensor<4x3xf32>
    %8 = "stablehlo.constant"() {value = dense<0xFF800000> : tensor<f32>} : () -> (tensor<f32>)
    %9 = "stablehlo.reduce"(%7, %8) ({
    ^bb0(%ir9_0r0_a0: tensor<f32>, %ir9_0r0_a1: tensor<f32>):
      %ir9_0r0_0_0 = "stablehlo.maximum"(%ir9_0r0_a0, %ir9_0r0_a1) : (tensor<f32>, tensor<f32>) -> (tensor<f32>)
      "stablehlo.return"(%ir9_0r0_0_0) : (tensor<f32>) -> ()
    }) {dimensions = array<i64: 1>} : (tensor<4x3xf32>, tensor<f32>) -> (tensor<4xf32>)
    %10 = stablehlo.reshape %9 : (tensor<4xf32>) -> tensor<4x1xf32>
    %11 = stablehlo.broadcast_in_dim %10, dims = [0, 1] : (tensor<4x1xf32>) -> tensor<4x3xf32>
    %12 = stablehlo.subtract %7, %11 : tensor<4x3xf32>
    %13 = stablehlo.exponential %12 : tensor<4x3xf32>
    %14 = "stablehlo.constant"() {value = dense<0.0> : tensor<f32>} : () -> (tensor<f32>)
    %15 = "stablehlo.reduce"(%13, %14) ({
    ^bb0(%ir15_0r0_a0: tensor<f32>, %ir15_0r0_a1: tensor<f32>):
      %ir15_0r0_0_0 = "stablehlo.add"(%ir15_0r0_a0, %ir15_0r0_a1) : (tensor<f32>, tensor<f32>) -> (tensor<f32>)
      "stablehlo.return"(%ir15_0r0_0_0) : (tensor<f32>) -> ()
    }) {dimensions = array<i64: 1>} : (tensor<4x3xf32>, tensor<f32>) -> (tensor<4xf32>)
    %16 = stablehlo.reshape %15 : (tensor<4xf32>) -> tensor<4x1xf32>
    %17 = stablehlo.log %16 : tensor<4x1xf32>
    %18 = stablehlo.broadcast_in_dim %17, dims = [0, 1] : (tensor<4x1xf32>) -> tensor<4x3xf32>
    %19 = stablehlo.subtract %12, %18 : tensor<4x3xf32>
    %20 = stablehlo.iota dim = 0 : tensor<3xi32>
    %21 = stablehlo.reshape %20 : (tensor<3xi32>) -> tensor<1x3xi32>
    %22 = stablehlo.broadcast_in_dim %21, dims = [0, 1] : (tensor<1x3xi32>) -> tensor<4x3xi32>
    %23 = stablehlo.reshape %arg14 : (tensor<4xi32>) -> tensor<4x1xi32>
    %24 = stablehlo.broadcast_in_dim %23, dims = [0, 1] : (tensor<4x1xi32>) -> tensor<4x3xi32>
    %25 = stablehlo.compare EQ, %22, %24, SIGNED : (tensor<4x3xi32>, tensor<4x3xi32>) -> tensor<4x3xi1>
    %26 = "stablehlo.constant"() {value = dense<0.0> : tensor<4x3xf32>} : () -> (tensor<4x3xf32>)
    %27 = stablehlo.select %25, %19, %26 : tensor<4x3xi1>, tensor<4x3xf32>
    %28 = "stablehlo.constant"() {value = dense<0.0> : tensor<f32>} : () -> (tensor<f32>)
    %29 = "stablehlo.reduce"(%27, %28) ({
    ^bb0(%ir29_0r0_a0: tensor<f32>, %ir29_0r0_a1: tensor<f32>):
      %ir29_0r0_0_0 = "stablehlo.add"(%ir29_0r0_a0, %ir29_0r0_a1) : (tensor<f32>, tensor<f32>) -> (tensor<f32>)
      "stablehlo.return"(%ir29_0r0_0_0) : (tensor<f32>) -> ()
    }) {dimensions = array<i64: 1>} : (tensor<4x3xf32>, tensor<f32>) -> (tensor<4xf32>)
    %30 = "stablehlo.constant"() {value = dense<0.0> : tensor<f32>} : () -> (tensor<f32>)
    %31 = "stablehlo.reduce"(%29, %30) ({
    ^bb0(%ir31_0r0_a0: tensor<f32>, %ir31_0r0_a1: tensor<f32>):
      %ir31_0r0_0_0 = "stablehlo.add"(%ir31_0r0_a0, %ir31_0r0_a1) : (tensor<f32>, tensor<f32>) -> (tensor<f32>)
      "stablehlo.return"(%ir31_0r0_0_0) : (tensor<f32>) -> ()
    }) {dimensions = array<i64: 0>} : (tensor<4xf32>, tensor<f32>) -> (tensor<f32>)
    %32 = "stablehlo.constant"() {value = dense<4.0> : tensor<f32>} : () -> (tensor<f32>)
    %33 = stablehlo.divide %31, %32 : tensor<f32>
    %34 = stablehlo.negate %33 : tensor<f32>
    %35 = "stablehlo.constant"() {value = dense<1.0> : tensor<f32>} : () -> (tensor<f32>)
    %36 = stablehlo.negate %35 : tensor<f32>
    %37 = stablehlo.reshape %36 : (tensor<f32>) -> tensor<1xf32>
    %38 = stablehlo.broadcast_in_dim %37, dims = [0] : (tensor<1xf32>) -> tensor<4xf32>
    %39 = "stablehlo.constant"() {value = dense<4.0> : tensor<4xf32>} : () -> (tensor<4xf32>)
    %40 = stablehlo.divide %38, %39 : tensor<4xf32>
    %41 = stablehlo.reshape %40 : (tensor<4xf32>) -> tensor<4x1xf32>
    %42 = stablehlo.broadcast_in_dim %41, dims = [0, 1] : (tensor<4x1xf32>) -> tensor<4x3xf32>
    %43 = "stablehlo.constant"() {value = dense<0.0> : tensor<4x3xf32>} : () -> (tensor<4x3xf32>)
    %44 = stablehlo.select %25, %42, %43 : tensor<4x3xi1>, tensor<4x3xf32>
    %45 = stablehlo.negate %44 : tensor<4x3xf32>
    %46 = "stablehlo.constant"() {value = dense<0.0> : tensor<f32>} : () -> (tensor<f32>)
    %47 = "stablehlo.reduce"(%45, %46) ({
    ^bb0(%ir47_0r0_a0: tensor<f32>, %ir47_0r0_a1: tensor<f32>):
      %ir47_0r0_0_0 = "stablehlo.add"(%ir47_0r0_a0, %ir47_0r0_a1) : (tensor<f32>, tensor<f32>) -> (tensor<f32>)
      "stablehlo.return"(%ir47_0r0_0_0) : (tensor<f32>) -> ()
    }) {dimensions = array<i64: 1>} : (tensor<4x3xf32>, tensor<f32>) -> (tensor<4xf32>)
    %48 = stablehlo.reshape %47 : (tensor<4xf32>) -> tensor<4x1xf32>
    %49 = stablehlo.reshape %48 : (tensor<4x1xf32>) -> tensor<4x1xf32>
    %50 = stablehlo.divide %49, %16 : tensor<4x1xf32>
    %51 = stablehlo.broadcast_in_dim %50, dims = [0, 1] : (tensor<4x1xf32>) -> tensor<4x3xf32>
    %52 = stablehlo.multiply %51, %13 : tensor<4x3xf32>
    %53 = stablehlo.add %44, %52 : tensor<4x3xf32>
    %54 = "stablehlo.constant"() {value = dense<0.0> : tensor<f32>} : () -> (tensor<f32>)
    %55 = "stablehlo.reduce"(%53, %54) ({
    ^bb0(%ir55_0r0_a0: tensor<f32>, %ir55_0r0_a1: tensor<f32>):
      %ir55_0r0_0_0 = "stablehlo.add"(%ir55_0r0_a0, %ir55_0r0_a1) : (tensor<f32>, tensor<f32>) -> (tensor<f32>)
      "stablehlo.return"(%ir55_0r0_0_0) : (tensor<f32>) -> ()
    }) {dimensions = array<i64: 0>} : (tensor<4x3xf32>, tensor<f32>) -> (tensor<3xf32>)
    %56 = stablehlo.reshape %55 : (tensor<3xf32>) -> tensor<1x3xf32>
    %57 = stablehlo.reshape %56 : (tensor<1x3xf32>) -> tensor<3xf32>
    %58 = stablehlo.transpose %arg2, dims = [1, 0] : (tensor<5x3xf32>) -> tensor<3x5xf32>
    %59 = "stablehlo.dot_general"(%53, %58) {dot_dimension_numbers = #stablehlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>} : (tensor<4x3xf32>, tensor<3x5xf32>) -> (tensor<4x5xf32>)
    %60 = stablehlo.transpose %4, dims = [1, 0] : (tensor<4x5xf32>) -> tensor<5x4xf32>
    %61 = "stablehlo.dot_general"(%60, %53) {dot_dimension_numbers = #stablehlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>} : (tensor<5x4xf32>, tensor<4x3xf32>) -> (tensor<5x3xf32>)
    %62 = "stablehlo.constant"() {value = dense<0.0> : tensor<4x5xf32>} : () -> (tensor<4x5xf32>)
    %63 = stablehlo.compare LT, %62, %2, FLOAT : (tensor<4x5xf32>, tensor<4x5xf32>) -> tensor<4x5xi1>
    %64 = stablehlo.select %63, %59, %62 : tensor<4x5xi1>, tensor<4x5xf32>
    %65 = "stablehlo.constant"() {value = dense<0.0> : tensor<f32>} : () -> (tensor<f32>)
    %66 = "stablehlo.reduce"(%64, %65) ({
    ^bb0(%ir66_0r0_a0: tensor<f32>, %ir66_0r0_a1: tensor<f32>):
      %ir66_0r0_0_0 = "stablehlo.add"(%ir66_0r0_a0, %ir66_0r0_a1) : (tensor<f32>, tensor<f32>) -> (tensor<f32>)
      "stablehlo.return"(%ir66_0r0_0_0) : (tensor<f32>) -> ()
    }) {dimensions = array<i64: 0>} : (tensor<4x5xf32>, tensor<f32>) -> (tensor<5xf32>)
    %67 = stablehlo.reshape %66 : (tensor<5xf32>) -> tensor<1x5xf32>
    %68 = stablehlo.reshape %67 : (tensor<1x5xf32>) -> tensor<5xf32>
    %69 = stablehlo.transpose %arg13, dims = [1, 0] : (tensor<4x3xf32>) -> tensor<3x4xf32>
    %70 = "stablehlo.dot_general"(%69, %64) {dot_dimension_numbers = #stablehlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>} : (tensor<3x4xf32>, tensor<4x5xf32>) -> (tensor<3x5xf32>)
    %71 = stablehlo.slice %arg12 [0:1] : (tensor<7xf32>) -> tensor<1xf32>
    %72 = stablehlo.reshape %71 : (tensor<1xf32>) -> tensor<f32>
    %73 = stablehlo.broadcast_in_dim %72, dims = [] : (tensor<f32>) -> tensor<3x5xf32>
    %74 = stablehlo.slice %arg12 [1:2] : (tensor<7xf32>) -> tensor<1xf32>
    %75 = stablehlo.reshape %74 : (tensor<1xf32>) -> tensor<f32>
    %76 = stablehlo.broadcast_in_dim %75, dims = [] : (tensor<f32>) -> tensor<3x5xf32>
    %77 = stablehlo.slice %arg12 [2:3] : (tensor<7xf32>) -> tensor<1xf32>
    %78 = stablehlo.reshape %77 : (tensor<1xf32>) -> tensor<f32>
    %79 = stablehlo.broadcast_in_dim %78, dims = [] : (tensor<f32>) -> tensor<3x5xf32>
    %80 = stablehlo.slice %arg12 [3:4] : (tensor<7xf32>) -> tensor<1xf32>
    %81 = stablehlo.reshape %80 : (tensor<1xf32>) -> tensor<f32>
    %82 = stablehlo.broadcast_in_dim %81, dims = [] : (tensor<f32>) -> tensor<3x5xf32>
    %83 = stablehlo.slice %arg12 [4:5] : (tensor<7xf32>) -> tensor<1xf32>
    %84 = stablehlo.reshape %83 : (tensor<1xf32>) -> tensor<f32>
    %85 = stablehlo.broadcast_in_dim %84, dims = [] : (tensor<f32>) -> tensor<3x5xf32>
    %86 = stablehlo.slice %arg12 [5:6] : (tensor<7xf32>) -> tensor<1xf32>
    %87 = stablehlo.reshape %86 : (tensor<1xf32>) -> tensor<f32>
    %88 = stablehlo.broadcast_in_dim %87, dims = [] : (tensor<f32>) -> tensor<3x5xf32>
    %89 = stablehlo.slice %arg12 [6:7] : (tensor<7xf32>) -> tensor<1xf32>
    %90 = stablehlo.reshape %89 : (tensor<1xf32>) -> tensor<f32>
    %91 = stablehlo.broadcast_in_dim %90, dims = [] : (tensor<f32>) -> tensor<3x5xf32>
    %92 = "stablehlo.constant"() {value = dense<1.0> : tensor<3x5xf32>} : () -> (tensor<3x5xf32>)
    %93 = stablehlo.subtract %92, %76 : tensor<3x5xf32>
    %94 = stablehlo.subtract %92, %79 : tensor<3x5xf32>
    %95 = stablehlo.multiply %73, %85 : tensor<3x5xf32>
    %96 = stablehlo.subtract %92, %95 : tensor<3x5xf32>
    %97 = stablehlo.multiply %arg4, %76 : tensor<3x5xf32>
    %98 = stablehlo.multiply %70, %93 : tensor<3x5xf32>
    %99 = stablehlo.add %97, %98 : tensor<3x5xf32>
    %100 = stablehlo.multiply %arg8, %79 : tensor<3x5xf32>
    %101 = stablehlo.multiply %70, %70 : tensor<3x5xf32>
    %102 = stablehlo.multiply %101, %94 : tensor<3x5xf32>
    %103 = stablehlo.add %100, %102 : tensor<3x5xf32>
    %104 = stablehlo.multiply %103, %91 : tensor<3x5xf32>
    %105 = stablehlo.sqrt %104 : tensor<3x5xf32>
    %106 = stablehlo.add %105, %82 : tensor<3x5xf32>
    %107 = stablehlo.multiply %99, %88 : tensor<3x5xf32>
    %108 = stablehlo.divide %107, %106 : tensor<3x5xf32>
    %109 = stablehlo.multiply %73, %108 : tensor<3x5xf32>
    %110 = stablehlo.multiply %arg0, %96 : tensor<3x5xf32>
    %111 = stablehlo.subtract %110, %109 : tensor<3x5xf32>
    %112 = stablehlo.slice %arg12 [0:1] : (tensor<7xf32>) -> tensor<1xf32>
    %113 = stablehlo.reshape %112 : (tensor<1xf32>) -> tensor<f32>
    %114 = stablehlo.broadcast_in_dim %113, dims = [] : (tensor<f32>) -> tensor<5xf32>
    %115 = stablehlo.slice %arg12 [1:2] : (tensor<7xf32>) -> tensor<1xf32>
    %116 = stablehlo.reshape %115 : (tensor<1xf32>) -> tensor<f32>
    %117 = stablehlo.broadcast_in_dim %116, dims = [] : (tensor<f32>) -> tensor<5xf32>
    %118 = stablehlo.slice %arg12 [2:3] : (tensor<7xf32>) -> tensor<1xf32>
    %119 = stablehlo.reshape %118 : (tensor<1xf32>) -> tensor<f32>
    %120 = stablehlo.broadcast_in_dim %119, dims = [] : (tensor<f32>) -> tensor<5xf32>
    %121 = stablehlo.slice %arg12 [3:4] : (tensor<7xf32>) -> tensor<1xf32>
    %122 = stablehlo.reshape %121 : (tensor<1xf32>) -> tensor<f32>
    %123 = stablehlo.broadcast_in_dim %122, dims = [] : (tensor<f32>) -> tensor<5xf32>
    %124 = stablehlo.slice %arg12 [4:5] : (tensor<7xf32>) -> tensor<1xf32>
    %125 = stablehlo.reshape %124 : (tensor<1xf32>) -> tensor<f32>
    %126 = stablehlo.broadcast_in_dim %125, dims = [] : (tensor<f32>) -> tensor<5xf32>
    %127 = stablehlo.slice %arg12 [5:6] : (tensor<7xf32>) -> tensor<1xf32>
    %128 = stablehlo.reshape %127 : (tensor<1xf32>) -> tensor<f32>
    %129 = stablehlo.broadcast_in_dim %128, dims = [] : (tensor<f32>) -> tensor<5xf32>
    %130 = stablehlo.slice %arg12 [6:7] : (tensor<7xf32>) -> tensor<1xf32>
    %131 = stablehlo.reshape %130 : (tensor<1xf32>) -> tensor<f32>
    %132 = stablehlo.broadcast_in_dim %131, dims = [] : (tensor<f32>) -> tensor<5xf32>
    %133 = "stablehlo.constant"() {value = dense<1.0> : tensor<5xf32>} : () -> (tensor<5xf32>)
    %134 = stablehlo.subtract %133, %117 : tensor<5xf32>
    %135 = stablehlo.subtract %133, %120 : tensor<5xf32>
    %136 = stablehlo.multiply %114, %126 : tensor<5xf32>
    %137 = stablehlo.subtract %133, %136 : tensor<5xf32>
    %138 = stablehlo.multiply %arg5, %117 : tensor<5xf32>
    %139 = stablehlo.multiply %68, %134 : tensor<5xf32>
    %140 = stablehlo.add %138, %139 : tensor<5xf32>
    %141 = stablehlo.multiply %arg9, %120 : tensor<5xf32>
    %142 = stablehlo.multiply %68, %68 : tensor<5xf32>
    %143 = stablehlo.multiply %142, %135 : tensor<5xf32>
    %144 = stablehlo.add %141, %143 : tensor<5xf32>
    %145 = stablehlo.multiply %144, %132 : tensor<5xf32>
    %146 = stablehlo.sqrt %145 : tensor<5xf32>
    %147 = stablehlo.add %146, %123 : tensor<5xf32>
    %148 = stablehlo.multiply %140, %129 : tensor<5xf32>
    %149 = stablehlo.divide %148, %147 : tensor<5xf32>
    %150 = stablehlo.multiply %114, %149 : tensor<5xf32>
    %151 = stablehlo.multiply %arg1, %137 : tensor<5xf32>
    %152 = stablehlo.subtract %151, %150 : tensor<5xf32>
    %153 = stablehlo.slice %arg12 [0:1] : (tensor<7xf32>) -> tensor<1xf32>
    %154 = stablehlo.reshape %153 : (tensor<1xf32>) -> tensor<f32>
    %155 = stablehlo.broadcast_in_dim %154, dims = [] : (tensor<f32>) -> tensor<5x3xf32>
    %156 = stablehlo.slice %arg12 [1:2] : (tensor<7xf32>) -> tensor<1xf32>
    %157 = stablehlo.reshape %156 : (tensor<1xf32>) -> tensor<f32>
    %158 = stablehlo.broadcast_in_dim %157, dims = [] : (tensor<f32>) -> tensor<5x3xf32>
    %159 = stablehlo.slice %arg12 [2:3] : (tensor<7xf32>) -> tensor<1xf32>
    %160 = stablehlo.reshape %159 : (tensor<1xf32>) -> tensor<f32>
    %161 = stablehlo.broadcast_in_dim %160, dims = [] : (tensor<f32>) -> tensor<5x3xf32>
    %162 = stablehlo.slice %arg12 [3:4] : (tensor<7xf32>) -> tensor<1xf32>
    %163 = stablehlo.reshape %162 : (tensor<1xf32>) -> tensor<f32>
    %164 = stablehlo.broadcast_in_dim %163, dims = [] : (tensor<f32>) -> tensor<5x3xf32>
    %165 = stablehlo.slice %arg12 [4:5] : (tensor<7xf32>) -> tensor<1xf32>
    %166 = stablehlo.reshape %165 : (tensor<1xf32>) -> tensor<f32>
    %167 = stablehlo.broadcast_in_dim %166, dims = [] : (tensor<f32>) -> tensor<5x3xf32>
    %168 = stablehlo.slice %arg12 [5:6] : (tensor<7xf32>) -> tensor<1xf32>
    %169 = stablehlo.reshape %168 : (tensor<1xf32>) -> tensor<f32>
    %170 = stablehlo.broadcast_in_dim %169, dims = [] : (tensor<f32>) -> tensor<5x3xf32>
    %171 = stablehlo.slice %arg12 [6:7] : (tensor<7xf32>) -> tensor<1xf32>
    %172 = stablehlo.reshape %171 : (tensor<1xf32>) -> tensor<f32>
    %173 = stablehlo.broadcast_in_dim %172, dims = [] : (tensor<f32>) -> tensor<5x3xf32>
    %174 = "stablehlo.constant"() {value = dense<1.0> : tensor<5x3xf32>} : () -> (tensor<5x3xf32>)
    %175 = stablehlo.subtract %174, %158 : tensor<5x3xf32>
    %176 = stablehlo.subtract %174, %161 : tensor<5x3xf32>
    %177 = stablehlo.multiply %155, %167 : tensor<5x3xf32>
    %178 = stablehlo.subtract %174, %177 : tensor<5x3xf32>
    %179 = stablehlo.multiply %arg6, %158 : tensor<5x3xf32>
    %180 = stablehlo.multiply %61, %175 : tensor<5x3xf32>
    %181 = stablehlo.add %179, %180 : tensor<5x3xf32>
    %182 = stablehlo.multiply %arg10, %161 : tensor<5x3xf32>
    %183 = stablehlo.multiply %61, %61 : tensor<5x3xf32>
    %184 = stablehlo.multiply %183, %176 : tensor<5x3xf32>
    %185 = stablehlo.add %182, %184 : tensor<5x3xf32>
    %186 = stablehlo.multiply %185, %173 : tensor<5x3xf32>
    %187 = stablehlo.sqrt %186 : tensor<5x3xf32>
    %188 = stablehlo.add %187, %164 : tensor<5x3xf32>
    %189 = stablehlo.multiply %181, %170 : tensor<5x3xf32>
    %190 = stablehlo.divide %189, %188 : tensor<5x3xf32>
    %191 = stablehlo.multiply %155, %190 : tensor<5x3xf32>
    %192 = stablehlo.multiply %arg2, %178 : tensor<5x3xf32>
    %193 = stablehlo.subtract %192, %191 : tensor<5x3xf32>
    %194 = stablehlo.slice %arg12 [0:1] : (tensor<7xf32>) -> tensor<1xf32>
    %195 = stablehlo.reshape %194 : (tensor<1xf32>) -> tensor<f32>
    %196 = stablehlo.broadcast_in_dim %195, dims = [] : (tensor<f32>) -> tensor<3xf32>
    %197 = stablehlo.slice %arg12 [1:2] : (tensor<7xf32>) -> tensor<1xf32>
    %198 = stablehlo.reshape %197 : (tensor<1xf32>) -> tensor<f32>
    %199 = stablehlo.broadcast_in_dim %198, dims = [] : (tensor<f32>) -> tensor<3xf32>
    %200 = stablehlo.slice %arg12 [2:3] : (tensor<7xf32>) -> tensor<1xf32>
    %201 = stablehlo.reshape %200 : (tensor<1xf32>) -> tensor<f32>
    %202 = stablehlo.broadcast_in_dim %201, dims = [] : (tensor<f32>) -> tensor<3xf32>
    %203 = stablehlo.slice %arg12 [3:4] : (tensor<7xf32>) -> tensor<1xf32>
    %204 = stablehlo.reshape %203 : (tensor<1xf32>) -> tensor<f32>
    %205 = stablehlo.broadcast_in_dim %204, dims = [] : (tensor<f32>) -> tensor<3xf32>
    %206 = stablehlo.slice %arg12 [4:5] : (tensor<7xf32>) -> tensor<1xf32>
    %207 = stablehlo.reshape %206 : (tensor<1xf32>) -> tensor<f32>
    %208 = stablehlo.broadcast_in_dim %207, dims = [] : (tensor<f32>) -> tensor<3xf32>
    %209 = stablehlo.slice %arg12 [5:6] : (tensor<7xf32>) -> tensor<1xf32>
    %210 = stablehlo.reshape %209 : (tensor<1xf32>) -> tensor<f32>
    %211 = stablehlo.broadcast_in_dim %210, dims = [] : (tensor<f32>) -> tensor<3xf32>
    %212 = stablehlo.slice %arg12 [6:7] : (tensor<7xf32>) -> tensor<1xf32>
    %213 = stablehlo.reshape %212 : (tensor<1xf32>) -> tensor<f32>
    %214 = stablehlo.broadcast_in_dim %213, dims = [] : (tensor<f32>) -> tensor<3xf32>
    %215 = "stablehlo.constant"() {value = dense<1.0> : tensor<3xf32>} : () -> (tensor<3xf32>)
    %216 = stablehlo.subtract %215, %199 : tensor<3xf32>
    %217 = stablehlo.subtract %215, %202 : tensor<3xf32>
    %218 = stablehlo.multiply %196, %208 : tensor<3xf32>
    %219 = stablehlo.subtract %215, %218 : tensor<3xf32>
    %220 = stablehlo.multiply %arg7, %199 : tensor<3xf32>
    %221 = stablehlo.multiply %57, %216 : tensor<3xf32>
    %222 = stablehlo.add %220, %221 : tensor<3xf32>
    %223 = stablehlo.multiply %arg11, %202 : tensor<3xf32>
    %224 = stablehlo.multiply %57, %57 : tensor<3xf32>
    %225 = stablehlo.multiply %224, %217 : tensor<3xf32>
    %226 = stablehlo.add %223, %225 : tensor<3xf32>
    %227 = stablehlo.multiply %226, %214 : tensor<3xf32>
    %228 = stablehlo.sqrt %227 : tensor<3xf32>
    %229 = stablehlo.add %228, %205 : tensor<3xf32>
    %230 = stablehlo.multiply %222, %211 : tensor<3xf32>
    %231 = stablehlo.divide %230, %229 : tensor<3xf32>
    %232 = stablehlo.multiply %196, %231 : tensor<3xf32>
    %233 = stablehlo.multiply %arg3, %219 : tensor<3xf32>
    %234 = stablehlo.subtract %233, %232 : tensor<3xf32>
    func.return %34, %111, %152, %193, %234, %99, %140, %181, %222, %103, %144, %185, %226 : tensor<f32>, tensor<3x5xf32>, tensor<5xf32>, tensor<5x3xf32>, tensor<3xf32>, tensor<3x5xf32>, tensor<5xf32>, tensor<5x3xf32>, tensor<3xf32>, tensor<3x5xf32>, tensor<5xf32>, tensor<5x3xf32>, tensor<3xf32>
  }
}
