// Test: Triton element-wise exp kernel → PTO holistic rewrite.
// Structure: get_program_id, masked load, math.exp, masked store.
//
// RUN: %triton_to_pto_mlir %s -o - | %filecheck %s
//
module {
  tt.func public @exp_kernel(
    %input_ptr: !tt.ptr<f32>,
    %output_ptr: !tt.ptr<f32>,
    %n_elements: i32
  ) attributes {noinline = false} {
    %c32_i32 = arith.constant 32 : i32
    %pid = tt.get_program_id x : i32
    %block_start = arith.muli %pid, %c32_i32 : i32
    %offsets = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
    %0 = tt.splat %block_start : i32 -> tensor<32xi32>
    %1 = arith.addi %0, %offsets : tensor<32xi32>
    %2 = tt.splat %n_elements : i32 -> tensor<32xi32>
    %mask = arith.cmpi slt, %1, %2 : tensor<32xi32>
    %3 = tt.splat %input_ptr : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>>
    %4 = tt.addptr %3, %1 : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
    %x = tt.load %4, %mask : tensor<32x!tt.ptr<f32>>
    %y = math.exp %x : tensor<32xf32>
    %5 = tt.splat %output_ptr : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>>
    %6 = tt.addptr %5, %1 : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
    tt.store %6, %y, %mask : tensor<32x!tt.ptr<f32>>
    tt.return
  }
}

// CHECK-LABEL: func.func @exp_kernel
// CHECK-SAME: %arg0: !pto.ptr<f32>
// CHECK-SAME: %arg1: !pto.ptr<f32>
// CHECK-SAME: %arg2: i32)
// CHECK: pto.get_block_idx
// CHECK: pto.get_block_num
// CHECK: arith.index_cast
// CHECK: arith.index_cast
// CHECK: arith.ceildivui
// CHECK: scf.for
// CHECK: arith.muli {{.*}} : index
// CHECK: arith.subi {{.*}} : index
// CHECK: arith.minui
// CHECK: pto.make_tensor_view %arg0
// CHECK: pto.make_tensor_view %arg1
// CHECK: pto.partition_view
// CHECK: pto.alloc_tile valid_row
// CHECK: pto.alloc_tile valid_row
// CHECK: pto.tload
// CHECK: pto.texp
// CHECK: pto.partition_view
// CHECK: pto.tstore
// CHECK: }
// CHECK: return
// CHECK-NOT: tt.
