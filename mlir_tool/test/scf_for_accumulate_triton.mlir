// RUN: %triton_to_pto_mlir %s -o - | %filecheck %s
//
// Kernel with scf.for loop carrying a tensor accumulator (iter_args). Body
// uses arith.addf on the accumulator tile. Tests conversion of scf.for with
// tensor iter_args to tile_buf, conversion of body ops, and scf.yield.
//
module {
  tt.func public @accumulate_kernel(
    %input_ptr: !tt.ptr<f32>,
    %n_elements: i32
  ) attributes {noinline = false} {
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    %init = arith.constant dense<0.000000e+00> : tensor<1024xf32>
    %result = scf.for %k = %c0 to %c4 step %c1 iter_args(%acc = %init) -> (tensor<1024xf32>) {
      %acc_next = arith.addf %acc, %acc : tensor<1024xf32>
      scf.yield %acc_next : tensor<1024xf32>
    }
    tt.return
  }
}

// CHECK-LABEL: func.func @accumulate_kernel
// CHECK-SAME: %arg0: !pto.ptr<f32>
// CHECK-SAME: %arg1: i32)
// CHECK: pto.constant_tile
// CHECK: scf.for {{.*}} iter_args({{.*}} = {{.*}}) -> (!pto.tile_buf<
// CHECK:   pto.alloc_tile
// CHECK:   pto.tadd ins(
// CHECK:   scf.yield
// CHECK: return
// CHECK-NOT: tt.
