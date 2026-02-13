// RUN: %triton_to_pto_mlir %s -o - | %filecheck %s
//
// -----------------------------------------------------------------------------
// Expected PTO-AS reference (vector add) — see pto-isa docs/grammar/PTO-AS.md,
// docs/isa/TLOAD.md, TADD.md, TSTORE.md, and docs/machine/abstract-machine.md.
// Full lowering converts tt.func to func.func with !pto.ptr args and
// tt.load/arith.addf/tt.store to pto.tload/pto.tadd/pto.tstore with !pto.tile.
// -----------------------------------------------------------------------------
//
// Real Triton IR (get_program_id, masked load/store, offset arithmetic).
//
module {
  tt.func public @add_kernel(
    %x_ptr: !tt.ptr<f32>,
    %y_ptr: !tt.ptr<f32>,
    %output_ptr: !tt.ptr<f32>,
    %n_elements: i32
  ) attributes {noinline = false} {
    %c1024_i32 = arith.constant 1024 : i32
    %pid = tt.get_program_id x : i32
    %block_start = arith.muli %pid, %c1024_i32 : i32
    %offsets = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32>
    %0 = tt.splat %block_start : i32 -> tensor<1024xi32>
    %1 = arith.addi %0, %offsets : tensor<1024xi32>
    %2 = tt.splat %n_elements : i32 -> tensor<1024xi32>
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

// CHECK-LABEL: func.func @add_kernel
// CHECK-SAME: %arg0: !pto.ptr<f32>
// CHECK-SAME: %arg1: !pto.ptr<f32>
// CHECK-SAME: %arg2: !pto.ptr<f32>
// CHECK-SAME: %arg3: i32)
// CHECK-DAG: arith.constant 0 : index
// CHECK: pto.make_tensor_view %arg0
// CHECK: pto.partition_view
// CHECK: pto.alloc_tile
// CHECK: pto.tload ins({{.*}}) outs({{.*}})
// CHECK: pto.make_tensor_view %arg1
// CHECK: pto.tload ins({{.*}}) outs({{.*}})
// CHECK: pto.alloc_tile
// CHECK: pto.tadd ins({{.*}}, {{.*}}) outs({{.*}})
// CHECK: pto.make_tensor_view %arg2
// CHECK: pto.tstore ins({{.*}}) outs({{.*}})
// CHECK: return
// CHECK-NOT: tt.
