module {
func.func @main(%arg0: tensor<128x2048xf32>, %arg1: tensor<2304x2048xf32>) -> tensor<128x2304xf32> {
  %cst = "tfl.no_value"() {value = unit} : () -> none
  %0 = "tfl.fully_connected"(%arg0, %arg1, %cst) <{asymmetric_quantize_inputs = false, fused_activation_function = "NONE", keep_num_dims = true, weights_format = "DEFAULT"}> : (tensor<128x2048xf32>, tensor<2304x2048xf32>, none) -> tensor<128x2304xf32>
  return %0 : tensor<128x2304xf32>
}
}