module {
  func.func @main(%arg0: tensor<1x128x1xf32>, %arg1: tensor<1x128x1xf32>) -> tensor<1x128x1xf32> {
    %0 = tfl.pow %arg0, %arg1 : tensor<1x128x1xf32>
    return %0 : tensor<1x128x1xf32>
  }
}
