module {
  func.func @main(%arg0: !torch.vtensor<[8192,8192],f16>, %arg1: !torch.vtensor<[4096,8192],f16>) -> !torch.vtensor<[4096,8192],f16> {
    %0 = torch.aten.mm %arg1, %arg0 : !torch.vtensor<[4096,8192],f16>, !torch.vtensor<[8192,8192],f16> -> !torch.vtensor<[4096,8192],f16>
    return %0 : !torch.vtensor<[4096,8192],f16>
  }
}
