// func.func @matmul_1(%lhs : tensor<512x512xf16>, %rhs : tensor<512x512xf16>, %empty : tensor<512x512xf32>) -> tensor<512x512xf32> {
//  %0      = linalg.matmul ins(%lhs, %rhs : tensor<512x512xf16>, tensor<512x512xf16>) outs(%empty: tensor<512x512xf32>) -> tensor<512x512xf32>
//  return %0 : tensor<512x512xf32>
// }
 
// func.func @matmul_2(%lhs : tensor<512x512xf16>, %rhs : tensor<512x512xf16>) -> tensor<512x512xf32> {
//  %c0     = arith.constant 0.0 : f32
//  %empty  = tensor.empty() : tensor<512x512xf32>
//  %c      = linalg.fill ins(%c0 : f32) outs(%empty: tensor<512x512xf32>) -> tensor<512x512xf32>
//  %0      = linalg.matmul ins(%lhs, %rhs : tensor<512x512xf16>, tensor<512x512xf16>) outs(%c: tensor<512x512xf32>) -> tensor<512x512xf32>
// 
//  return %0 : tensor<512x512xf32>
// }
 
func.func @matmul_3(%lhs : tensor<512x512xf16>, %rhs : tensor<512x512xf16>) -> tensor<512x512xf16> {
%c0     = arith.constant 0.0 : f32
%empty  = tensor.empty() : tensor<512x512xf32>
%c      = linalg.fill ins(%c0 : f32) outs(%empty: tensor<512x512xf32>) -> tensor<512x512xf32>
%0      = linalg.matmul ins(%lhs, %rhs : tensor<512x512xf16>, tensor<512x512xf16>) outs(%c: tensor<512x512xf32>) -> tensor<512x512xf32>

// if output f16, need cast tensor core result
%d      = tosa.cast %0 : (tensor<512x512xf32>) -> tensor<512x512xf16>
return %d : tensor<512x512xf16>
}

// TODO: Error while create async group, LOW PRIORITY
//
// // extra shared memory for intermeidate D, and assert in create async cp intrin
// func.func @matmul_4(%lhs : tensor<512x512xf16>, %rhs : tensor<512x512xf16>, %init : tensor<512x512xf32>) -> tensor<512x512xf16> {
//  %0      = linalg.matmul ins(%lhs, %rhs : tensor<512x512xf16>, tensor<512x512xf16>) outs(%init: tensor<512x512xf32>) -> tensor<512x512xf32>
// 
//  // if output f16, need cast tensor core result
//  %d      = tosa.cast %0 : (tensor<512x512xf32>) -> tensor<512x512xf16>
//  return %d : tensor<512x512xf16>
// }



//func.func @matmul_5(%lhs : tensor<512x512xf16>, %rhs : tensor<512x512xf16>) -> tensor<512x512xf16> {
//  %c0     = arith.constant 0.0 : f16
//  %empty  = tensor.empty() : tensor<512x512xf16>
//  %c      = linalg.fill ins(%c0: f16) outs(%empty: tensor<512x512xf16>) -> tensor<512x512xf16>
//  %0      = linalg.matmul ins(%lhs, %rhs : tensor<512x512xf16>, tensor<512x512xf16>) outs(%empty: tensor<512x512xf16>) -> tensor<512x512xf16>
//  return %0 : tensor<512x512xf16>
//}



//func.func @matmul_6(%lhs : tensor<512x512xf16>, %rhs : tensor<512x512xf16>, %empty : tensor<512x512xf16>) -> tensor<512x512xf16> {
//  %0      = linalg.matmul ins(%lhs, %rhs : tensor<512x512xf16>, tensor<512x512xf16>) outs(%empty: tensor<512x512xf16>) -> tensor<512x512xf16>
//  return %0 : tensor<512x512xf16>
//}


