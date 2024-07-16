func.func @matmul(%lhs : tensor<16x16xf16>, %rhs : tensor<16x16xf16>,
                  %init : tensor<16x16xf32>) -> tensor<16x16xf32> {
  %0 = linalg.matmul
         ins(%lhs, %rhs : tensor<16x16xf16>, tensor<16x16xf16>)
         outs(%init : tensor<16x16xf32>) -> tensor<16x16xf32>
  return %0 : tensor<16x16xf32>
}
