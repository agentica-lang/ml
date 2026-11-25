module @col attributes {mhlo.num_partitions = 2 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%x: tensor<8x16xf32> {mhlo.sharding = "{replicated}"}, %w: tensor<16x32xf32> {mhlo.sharding = "{devices=[1,2]0,1}"}) -> (tensor<8x32xf32> {mhlo.sharding = "{devices=[1,2]0,1}"}) {
    %0 = stablehlo.dot_general %x, %w, contracting_dims = [1] x [0] : (tensor<8x16xf32>, tensor<16x32xf32>) -> tensor<8x32xf32>
    return %0 : tensor<8x32xf32>
  }
}
