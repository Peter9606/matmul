func.func @matmul(%lhs : tensor<128x256xf16>, %rhs : tensor<256x128xf16>,
                  %init : tensor<128x128xf32>) -> tensor<128x128xf32> {
  %0 = linalg.matmul
         ins(%lhs, %rhs : tensor<128x256xf16>, tensor<256x128xf16>)
         outs(%init : tensor<128x128xf32>) -> tensor<128x128xf32>
  return %0 : tensor<128x128xf32>
}
