//func.func @matmul_1(%lhs : tensor<2048x2048xf16>, %rhs : tensor<2048x2048xf16>, %empty : tensor<2048x2048xf32>) -> tensor<2048x2048xf32> {
// %0      = linalg.matmul ins(%lhs, %rhs : tensor<2048x2048xf16>, tensor<2048x2048xf16>) outs(%empty: tensor<2048x2048xf32>) -> tensor<2048x2048xf32>
// return %0 : tensor<2048x2048xf32>
//}


func.func @matmul_2(%lhs : tensor<2048x2048xf16>, %rhs : tensor<2048x2048xf16>) -> tensor<2048x2048xf32> {
 %c0     = arith.constant 0.0 : f32
 %empty  = tensor.empty() : tensor<2048x2048xf32>
 %c      = linalg.fill ins(%c0 : f32) outs(%empty: tensor<2048x2048xf32>) -> tensor<2048x2048xf32>
 %0      = linalg.matmul ins(%lhs, %rhs : tensor<2048x2048xf16>, tensor<2048x2048xf16>) outs(%c: tensor<2048x2048xf32>) -> tensor<2048x2048xf32>

 return %0 : tensor<2048x2048xf32>
}


// func.func @matmul_3(%lhs : tensor<2048x2048xf16>, %rhs : tensor<2048x2048xf16>) -> tensor<2048x2048xf16> {
// %c0     = arith.constant 0.0 : f32
// %empty  = tensor.empty() : tensor<2048x2048xf32>
// %c      = linalg.fill ins(%c0 : f32) outs(%empty: tensor<2048x2048xf32>) -> tensor<2048x2048xf32>
// %0      = linalg.matmul ins(%lhs, %rhs : tensor<2048x2048xf16>, tensor<2048x2048xf16>) outs(%c: tensor<2048x2048xf32>) -> tensor<2048x2048xf32>

// // if output f16, need cast tensor core result
// %d      = tosa.cast %0 : (tensor<2048x2048xf32>) -> tensor<2048x2048xf16>
// return %d : tensor<2048x2048xf16>
// }


// func.func @matmul_4(%lhs : tensor<2048x2048xf16>, %rhs : tensor<2048x2048xf16>, %init : tensor<2048x2048xf32>) -> tensor<2048x2048xf16> {
//  %0      = linalg.matmul ins(%lhs, %rhs : tensor<2048x2048xf16>, tensor<2048x2048xf16>) outs(%init: tensor<2048x2048xf32>) -> tensor<2048x2048xf32>

//  // if output f16, need cast tensor core result
//  %d      = tosa.cast %0 : (tensor<2048x2048xf32>) -> tensor<2048x2048xf16>
//  return %d : tensor<2048x2048xf16>
// }




// #translation = #iree_codegen.translation_info<LLVMGPUMatmulTensorCoreMmaSync>
// func.func @matmul_5(%lhs : tensor<1x2048x2048xf32>, %rhs : tensor<1x2048x2048xf32>) -> tensor<1x2048x2048xf32>
//   attributes {translation_info = #translation} {
//   %0 = "tosa.matmul"(%lhs, %rhs) : (tensor<1x2048x2048xf32>, tensor<1x2048x2048xf32>) -> tensor<1x2048x2048xf32>
//   return %0 : tensor<1x2048x2048xf32>
// }

