module {
  func.func public @main(%arg0: tensor<3x5xf32>, %arg1: tensor<5xf32>, %arg2: tensor<5x3xf32>, %arg3: tensor<3xf32>, %arg4: tensor<3x5xf32>, %arg5: tensor<5xf32>, %arg6: tensor<5x3xf32>, %arg7: tensor<3xf32>, %arg8: tensor<3x5xf32>, %arg9: tensor<5xf32>, %arg10: tensor<5x3xf32>, %arg11: tensor<3xf32>, %arg12: tensor<f32>, %arg13: tensor<f32>, %arg14: tensor<4x3xf32>, %arg15: tensor<4xi32>) -> (tensor<f32>, tensor<3x5xf32>, tensor<5xf32>, tensor<5x3xf32>, tensor<3xf32>, tensor<3x5xf32>, tensor<5xf32>, tensor<5x3xf32>, tensor<3xf32>, tensor<3x5xf32>, tensor<5xf32>, tensor<5x3xf32>, tensor<3xf32>) {
    %0 = stablehlo.dot_general %arg14, %arg0, contracting_dims = [1] x [0] : (tensor<4x3xf32>, tensor<3x5xf32>) -> tensor<4x5xf32>
    %1 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<5xf32>) -> tensor<4x5xf32>
    %2 = stablehlo.add %0, %1 : tensor<4x5xf32>
    %3 = stablehlo.constant dense<0.0> : tensor<4x5xf32>
    %4 = stablehlo.maximum %2, %3 : tensor<4x5xf32>
    %5 = stablehlo.dot_general %4, %arg2, contracting_dims = [1] x [0] : (tensor<4x5xf32>, tensor<5x3xf32>) -> tensor<4x3xf32>
    %6 = stablehlo.broadcast_in_dim %arg3, dims = [1] : (tensor<3xf32>) -> tensor<4x3xf32>
    %7 = stablehlo.add %5, %6 : tensor<4x3xf32>
    %8 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %9 = stablehlo.reduce(%7 init: %8) applies stablehlo.maximum across dimensions = [1] : (tensor<4x3xf32>, tensor<f32>) -> tensor<4xf32>
    %10 = stablehlo.reshape %9 : (tensor<4xf32>) -> tensor<4x1xf32>
    %11 = stablehlo.broadcast_in_dim %10, dims = [0, 1] : (tensor<4x1xf32>) -> tensor<4x3xf32>
    %12 = stablehlo.subtract %7, %11 : tensor<4x3xf32>
    %13 = stablehlo.exponential %12 : tensor<4x3xf32>
    %14 = stablehlo.constant dense<0.0> : tensor<f32>
    %15 = stablehlo.reduce(%13 init: %14) applies stablehlo.add across dimensions = [1] : (tensor<4x3xf32>, tensor<f32>) -> tensor<4xf32>
    %16 = stablehlo.reshape %15 : (tensor<4xf32>) -> tensor<4x1xf32>
    %17 = stablehlo.log %16 : tensor<4x1xf32>
    %18 = stablehlo.broadcast_in_dim %17, dims = [0, 1] : (tensor<4x1xf32>) -> tensor<4x3xf32>
    %19 = stablehlo.subtract %12, %18 : tensor<4x3xf32>
    %20 = stablehlo.iota dim = 0 : tensor<3xi32>
    %21 = stablehlo.reshape %20 : (tensor<3xi32>) -> tensor<1x3xi32>
    %22 = stablehlo.broadcast_in_dim %21, dims = [0, 1] : (tensor<1x3xi32>) -> tensor<4x3xi32>
    %23 = stablehlo.reshape %arg15 : (tensor<4xi32>) -> tensor<4x1xi32>
    %24 = stablehlo.broadcast_in_dim %23, dims = [0, 1] : (tensor<4x1xi32>) -> tensor<4x3xi32>
    %25 = stablehlo.compare EQ, %22, %24, SIGNED : (tensor<4x3xi32>, tensor<4x3xi32>) -> tensor<4x3xi1>
    %26 = stablehlo.constant dense<0.0> : tensor<4x3xf32>
    %27 = stablehlo.select %25, %19, %26 : tensor<4x3xi1>, tensor<4x3xf32>
    %28 = stablehlo.constant dense<0.0> : tensor<f32>
    %29 = stablehlo.reduce(%27 init: %28) applies stablehlo.add across dimensions = [1] : (tensor<4x3xf32>, tensor<f32>) -> tensor<4xf32>
    %30 = stablehlo.constant dense<0.0> : tensor<f32>
    %31 = stablehlo.reduce(%29 init: %30) applies stablehlo.add across dimensions = [0] : (tensor<4xf32>, tensor<f32>) -> tensor<f32>
    %32 = stablehlo.constant dense<4.0> : tensor<f32>
    %33 = stablehlo.divide %31, %32 : tensor<f32>
    %34 = stablehlo.negate %33 : tensor<f32>
    %35 = stablehlo.constant dense<1.0> : tensor<f32>
    %36 = stablehlo.negate %35 : tensor<f32>
    %37 = stablehlo.reshape %36 : (tensor<f32>) -> tensor<1xf32>
    %38 = stablehlo.broadcast_in_dim %37, dims = [0] : (tensor<1xf32>) -> tensor<4xf32>
    %39 = stablehlo.constant dense<4.0> : tensor<4xf32>
    %40 = stablehlo.divide %38, %39 : tensor<4xf32>
    %41 = stablehlo.reshape %40 : (tensor<4xf32>) -> tensor<4x1xf32>
    %42 = stablehlo.broadcast_in_dim %41, dims = [0, 1] : (tensor<4x1xf32>) -> tensor<4x3xf32>
    %43 = stablehlo.constant dense<0.0> : tensor<4x3xf32>
    %44 = stablehlo.select %25, %42, %43 : tensor<4x3xi1>, tensor<4x3xf32>
    %45 = stablehlo.negate %44 : tensor<4x3xf32>
    %46 = stablehlo.constant dense<0.0> : tensor<f32>
    %47 = stablehlo.reduce(%45 init: %46) applies stablehlo.add across dimensions = [1] : (tensor<4x3xf32>, tensor<f32>) -> tensor<4xf32>
    %48 = stablehlo.reshape %47 : (tensor<4xf32>) -> tensor<4x1xf32>
    %49 = stablehlo.reshape %48 : (tensor<4x1xf32>) -> tensor<4x1xf32>
    %50 = stablehlo.divide %49, %16 : tensor<4x1xf32>
    %51 = stablehlo.broadcast_in_dim %50, dims = [0, 1] : (tensor<4x1xf32>) -> tensor<4x3xf32>
    %52 = stablehlo.multiply %51, %13 : tensor<4x3xf32>
    %53 = stablehlo.add %44, %52 : tensor<4x3xf32>
    %54 = stablehlo.constant dense<0.0> : tensor<f32>
    %55 = stablehlo.reduce(%53 init: %54) applies stablehlo.add across dimensions = [0] : (tensor<4x3xf32>, tensor<f32>) -> tensor<3xf32>
    %56 = stablehlo.reshape %55 : (tensor<3xf32>) -> tensor<1x3xf32>
    %57 = stablehlo.reshape %56 : (tensor<1x3xf32>) -> tensor<3xf32>
    %58 = stablehlo.transpose %arg2, dims = [1, 0] : (tensor<5x3xf32>) -> tensor<3x5xf32>
    %59 = stablehlo.dot_general %53, %58, contracting_dims = [1] x [0] : (tensor<4x3xf32>, tensor<3x5xf32>) -> tensor<4x5xf32>
    %60 = stablehlo.transpose %4, dims = [1, 0] : (tensor<4x5xf32>) -> tensor<5x4xf32>
    %61 = stablehlo.dot_general %60, %53, contracting_dims = [1] x [0] : (tensor<5x4xf32>, tensor<4x3xf32>) -> tensor<5x3xf32>
    %62 = stablehlo.constant dense<0.0> : tensor<4x5xf32>
    %63 = stablehlo.compare LT, %62, %2, FLOAT : (tensor<4x5xf32>, tensor<4x5xf32>) -> tensor<4x5xi1>
    %64 = stablehlo.select %63, %59, %62 : tensor<4x5xi1>, tensor<4x5xf32>
    %65 = stablehlo.constant dense<0.0> : tensor<f32>
    %66 = stablehlo.reduce(%64 init: %65) applies stablehlo.add across dimensions = [0] : (tensor<4x5xf32>, tensor<f32>) -> tensor<5xf32>
    %67 = stablehlo.reshape %66 : (tensor<5xf32>) -> tensor<1x5xf32>
    %68 = stablehlo.reshape %67 : (tensor<1x5xf32>) -> tensor<5xf32>
    %69 = stablehlo.transpose %arg14, dims = [1, 0] : (tensor<4x3xf32>) -> tensor<3x4xf32>
    %70 = stablehlo.dot_general %69, %64, contracting_dims = [1] x [0] : (tensor<3x4xf32>, tensor<4x5xf32>) -> tensor<3x5xf32>
    %71 = stablehlo.constant dense<0.9> : tensor<3x5xf32>
    %72 = stablehlo.multiply %arg4, %71 : tensor<3x5xf32>
    %73 = stablehlo.constant dense<0.09999999999999998> : tensor<3x5xf32>
    %74 = stablehlo.multiply %70, %73 : tensor<3x5xf32>
    %75 = stablehlo.add %72, %74 : tensor<3x5xf32>
    %76 = stablehlo.constant dense<0.999> : tensor<3x5xf32>
    %77 = stablehlo.multiply %arg8, %76 : tensor<3x5xf32>
    %78 = stablehlo.multiply %70, %70 : tensor<3x5xf32>
    %79 = stablehlo.constant dense<0.0010000000000000009> : tensor<3x5xf32>
    %80 = stablehlo.multiply %78, %79 : tensor<3x5xf32>
    %81 = stablehlo.add %77, %80 : tensor<3x5xf32>
    %82 = stablehlo.broadcast_in_dim %arg13, dims = [] : (tensor<f32>) -> tensor<3x5xf32>
    %83 = stablehlo.multiply %81, %82 : tensor<3x5xf32>
    %84 = stablehlo.sqrt %83 : tensor<3x5xf32>
    %85 = stablehlo.constant dense<1.0e-8> : tensor<3x5xf32>
    %86 = stablehlo.add %84, %85 : tensor<3x5xf32>
    %87 = stablehlo.broadcast_in_dim %arg12, dims = [] : (tensor<f32>) -> tensor<3x5xf32>
    %88 = stablehlo.multiply %75, %87 : tensor<3x5xf32>
    %89 = stablehlo.divide %88, %86 : tensor<3x5xf32>
    %90 = stablehlo.subtract %arg0, %89 : tensor<3x5xf32>
    %91 = stablehlo.constant dense<0.9> : tensor<5xf32>
    %92 = stablehlo.multiply %arg5, %91 : tensor<5xf32>
    %93 = stablehlo.constant dense<0.09999999999999998> : tensor<5xf32>
    %94 = stablehlo.multiply %68, %93 : tensor<5xf32>
    %95 = stablehlo.add %92, %94 : tensor<5xf32>
    %96 = stablehlo.constant dense<0.999> : tensor<5xf32>
    %97 = stablehlo.multiply %arg9, %96 : tensor<5xf32>
    %98 = stablehlo.multiply %68, %68 : tensor<5xf32>
    %99 = stablehlo.constant dense<0.0010000000000000009> : tensor<5xf32>
    %100 = stablehlo.multiply %98, %99 : tensor<5xf32>
    %101 = stablehlo.add %97, %100 : tensor<5xf32>
    %102 = stablehlo.broadcast_in_dim %arg13, dims = [] : (tensor<f32>) -> tensor<5xf32>
    %103 = stablehlo.multiply %101, %102 : tensor<5xf32>
    %104 = stablehlo.sqrt %103 : tensor<5xf32>
    %105 = stablehlo.constant dense<1.0e-8> : tensor<5xf32>
    %106 = stablehlo.add %104, %105 : tensor<5xf32>
    %107 = stablehlo.broadcast_in_dim %arg12, dims = [] : (tensor<f32>) -> tensor<5xf32>
    %108 = stablehlo.multiply %95, %107 : tensor<5xf32>
    %109 = stablehlo.divide %108, %106 : tensor<5xf32>
    %110 = stablehlo.subtract %arg1, %109 : tensor<5xf32>
    %111 = stablehlo.constant dense<0.9> : tensor<5x3xf32>
    %112 = stablehlo.multiply %arg6, %111 : tensor<5x3xf32>
    %113 = stablehlo.constant dense<0.09999999999999998> : tensor<5x3xf32>
    %114 = stablehlo.multiply %61, %113 : tensor<5x3xf32>
    %115 = stablehlo.add %112, %114 : tensor<5x3xf32>
    %116 = stablehlo.constant dense<0.999> : tensor<5x3xf32>
    %117 = stablehlo.multiply %arg10, %116 : tensor<5x3xf32>
    %118 = stablehlo.multiply %61, %61 : tensor<5x3xf32>
    %119 = stablehlo.constant dense<0.0010000000000000009> : tensor<5x3xf32>
    %120 = stablehlo.multiply %118, %119 : tensor<5x3xf32>
    %121 = stablehlo.add %117, %120 : tensor<5x3xf32>
    %122 = stablehlo.broadcast_in_dim %arg13, dims = [] : (tensor<f32>) -> tensor<5x3xf32>
    %123 = stablehlo.multiply %121, %122 : tensor<5x3xf32>
    %124 = stablehlo.sqrt %123 : tensor<5x3xf32>
    %125 = stablehlo.constant dense<1.0e-8> : tensor<5x3xf32>
    %126 = stablehlo.add %124, %125 : tensor<5x3xf32>
    %127 = stablehlo.broadcast_in_dim %arg12, dims = [] : (tensor<f32>) -> tensor<5x3xf32>
    %128 = stablehlo.multiply %115, %127 : tensor<5x3xf32>
    %129 = stablehlo.divide %128, %126 : tensor<5x3xf32>
    %130 = stablehlo.subtract %arg2, %129 : tensor<5x3xf32>
    %131 = stablehlo.constant dense<0.9> : tensor<3xf32>
    %132 = stablehlo.multiply %arg7, %131 : tensor<3xf32>
    %133 = stablehlo.constant dense<0.09999999999999998> : tensor<3xf32>
    %134 = stablehlo.multiply %57, %133 : tensor<3xf32>
    %135 = stablehlo.add %132, %134 : tensor<3xf32>
    %136 = stablehlo.constant dense<0.999> : tensor<3xf32>
    %137 = stablehlo.multiply %arg11, %136 : tensor<3xf32>
    %138 = stablehlo.multiply %57, %57 : tensor<3xf32>
    %139 = stablehlo.constant dense<0.0010000000000000009> : tensor<3xf32>
    %140 = stablehlo.multiply %138, %139 : tensor<3xf32>
    %141 = stablehlo.add %137, %140 : tensor<3xf32>
    %142 = stablehlo.broadcast_in_dim %arg13, dims = [] : (tensor<f32>) -> tensor<3xf32>
    %143 = stablehlo.multiply %141, %142 : tensor<3xf32>
    %144 = stablehlo.sqrt %143 : tensor<3xf32>
    %145 = stablehlo.constant dense<1.0e-8> : tensor<3xf32>
    %146 = stablehlo.add %144, %145 : tensor<3xf32>
    %147 = stablehlo.broadcast_in_dim %arg12, dims = [] : (tensor<f32>) -> tensor<3xf32>
    %148 = stablehlo.multiply %135, %147 : tensor<3xf32>
    %149 = stablehlo.divide %148, %146 : tensor<3xf32>
    %150 = stablehlo.subtract %arg3, %149 : tensor<3xf32>
    func.return %34, %90, %110, %130, %150, %75, %95, %115, %135, %81, %101, %121, %141 : tensor<f32>, tensor<3x5xf32>, tensor<5xf32>, tensor<5x3xf32>, tensor<3xf32>, tensor<3x5xf32>, tensor<5xf32>, tensor<5x3xf32>, tensor<3xf32>, tensor<3x5xf32>, tensor<5xf32>, tensor<5x3xf32>, tensor<3xf32>
  }
}
