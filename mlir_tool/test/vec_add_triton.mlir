// RUN: triton-to-pto-mlir %s -o - | FileCheck %s

module {
  tt.func @vec_add(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: !tt.ptr<f32>) {
    %0 = tt.load %arg0 : !tt.ptr<f32> -> tensor<16x16xf32>
    %1 = tt.load %arg1 : !tt.ptr<f32> -> tensor<16x16xf32>
    %2 = tt.add %0, %1 : tensor<16x16xf32>
    tt.store %arg2, %2
    tt.return
  }
}

// CHECK-LABEL: tt.func @vec_add
// CHECK: pto.tload %arg0
// CHECK: pto.tload %arg1
// CHECK: pto.tadd %0, %1
// CHECK: pto.tstore %arg2, %2

