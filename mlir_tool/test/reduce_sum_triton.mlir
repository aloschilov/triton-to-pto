// RUN: %triton_to_pto_mlir %s -o - | %filecheck %s
//
// Row-sum reduction kernel: real Triton IR (get_program_id, masked load,
// offset arithmetic) feeding a tt.reduce to a scalar return value.
//
module {
  tt.func public @sum_kernel(
    %input_ptr: !tt.ptr<f32>,
    %n_elements: i32
  ) -> f32 attributes {noinline = false} {
    %c1024_i32 = arith.constant 1024 : i32
    %pid = tt.get_program_id x : i32
    %block_start = arith.muli %pid, %c1024_i32 : i32
    %offsets = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32>
    %0 = tt.splat %block_start : i32 -> tensor<1024xi32>
    %1 = arith.addi %0, %offsets : tensor<1024xi32>
    %2 = tt.splat %n_elements : i32 -> tensor<1024xi32>
    %mask = arith.cmpi slt, %1, %2 : tensor<1024xi32>
    %3 = tt.splat %input_ptr : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
    %4 = tt.addptr %3, %1 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>
    %x = tt.load %4, %mask : tensor<1024x!tt.ptr<f32>>
    // Generic tt.reduce op (unregistered), reduced to scalar f32.
    %sum = "tt.reduce"(%x) ({
    ^bb0(%a: f32, %b: f32):
      %add = arith.addf %a, %b : f32
      "tt.reduce.return"(%add) : (f32) -> ()
    }) {axis = 0 : i32} : (tensor<1024xf32>) -> f32
    tt.return %sum : f32
  }
}

// CHECK-LABEL: func.func @sum_kernel
// CHECK-SAME: %arg0: !pto.ptr<f32>
// CHECK-SAME: %arg1: i32) -> f32
// CHECK-DAG: arith.constant 0 : index
// CHECK: pto.make_tensor_view %arg0
// CHECK: pto.partition_view
// CHECK: pto.alloc_tile
// CHECK: pto.tload ins({{.*}}) outs({{.*}})
// CHECK: pto.treduce_sum {{.*}} -> f32
// CHECK: return {{.*}} : f32
// CHECK-NOT: tt.

