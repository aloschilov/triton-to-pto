// RUN: %triton_to_pto_mlir %s -o - | %filecheck %s
//
// Kernel with scf.for loop that loads a tile from memory inside the body
// and accumulates into an iter_arg. Tests cross-region references to Triton
// ops (tt.addptr result defined outside the loop used inside).
//
module {
  tt.func public @load_accumulate_kernel(
    %input_ptr: !tt.ptr<f32>,
    %n_elements: i32
  ) attributes {noinline = false} {
    %c1024_i32 = arith.constant 1024 : i32
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    %pid = tt.get_program_id x : i32
    %block_start = arith.muli %pid, %c1024_i32 : i32
    %offsets = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32>
    %0 = tt.splat %block_start : i32 -> tensor<1024xi32>
    %1 = arith.addi %0, %offsets : tensor<1024xi32>
    %2 = tt.splat %n_elements : i32 -> tensor<1024xi32>
    %mask = arith.cmpi slt, %1, %2 : tensor<1024xi32>
    %3 = tt.splat %input_ptr : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
    %4 = tt.addptr %3, %1 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>
    %init = arith.constant dense<0.000000e+00> : tensor<1024xf32>
    %result = scf.for %k = %c0 to %c4 step %c1 iter_args(%acc = %init) -> (tensor<1024xf32>) {
      %loaded = tt.load %4, %mask : tensor<1024x!tt.ptr<f32>>
      %acc_next = arith.addf %acc, %loaded : tensor<1024xf32>
      scf.yield %acc_next : tensor<1024xf32>
    }
    tt.return
  }
}

// CHECK-LABEL: func.func @load_accumulate_kernel
// CHECK-SAME: %arg0: !pto.ptr<f32>
// CHECK-SAME: %arg1: i32)
// CHECK: pto.constant_tile
// CHECK: scf.for {{.*}} iter_args({{.*}} = {{.*}}) -> (!pto.tile_buf<
// CHECK:   pto.make_tensor_view
// CHECK:   pto.tload
// CHECK:   pto.tadd
// CHECK:   scf.yield
// CHECK: return
// CHECK-NOT: tt.
