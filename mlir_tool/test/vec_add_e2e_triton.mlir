// E2E test asset: vec_add with 1024 elements, single block (no get_program_id),
// 3 args. Produces PTO compatible with PTOAS VectorAddition (32x32 tiles).
//
// RUN: %triton_to_pto_mlir %s -o - | %filecheck %s
//
module {
  tt.func public @vec_add_kernel_2d(
    %x_ptr: !tt.ptr<f32>,
    %y_ptr: !tt.ptr<f32>,
    %output_ptr: !tt.ptr<f32>
  ) attributes {noinline = false} {
    %c1024_i32 = arith.constant 1024 : i32
    %block_start = arith.constant 0 : i32
    %offsets = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32>
    %0 = tt.splat %block_start : i32 -> tensor<1024xi32>
    %1 = arith.addi %0, %offsets : tensor<1024xi32>
    %2 = tt.splat %c1024_i32 : i32 -> tensor<1024xi32>
    %mask = arith.cmpi slt, %1, %2 : tensor<1024xi32>
    %3 = tt.splat %x_ptr : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
    %4 = tt.addptr %3, %1 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>
    %x = tt.load %4, %mask : tensor<1024x!tt.ptr<f32>>
    %5 = tt.splat %y_ptr : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
    %6 = tt.addptr %5, %1 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>
    %y = tt.load %6, %mask : tensor<1024x!tt.ptr<f32>>
    %output = arith.addf %x, %y : tensor<1024xf32>
    %7 = tt.splat %output_ptr : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
    %8 = tt.addptr %7, %1 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>
    tt.store %8, %output, %mask : tensor<1024x!tt.ptr<f32>>
    tt.return
  }
}

// CHECK-LABEL: func.func @vec_add_kernel_2d
// CHECK-SAME: %arg0: !pto.ptr<f32>
// CHECK-SAME: %arg1: !pto.ptr<f32>
// CHECK-SAME: %arg2: !pto.ptr<f32>
// CHECK-NOT: get_block_idx
// CHECK: pto.make_tensor_view %arg0, shape = [%c32, {{.*}}]
// CHECK: pto.partition_view {{.*}} sizes = [%c32, {{.*}}]
// PTOAS dialect: tile_buf uses loc=vec (AddressSpace::VEC), key-value format
// CHECK: pto.alloc_tile : {{.*}}loc=vec, dtype=f32, rows=32, cols=32,
// CHECK: pto.alloc_tile
// CHECK: pto.tload
// CHECK: pto.tadd
// CHECK: pto.tstore
