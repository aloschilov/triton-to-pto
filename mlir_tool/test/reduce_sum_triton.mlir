// RUN: %triton_to_pto_mlir %s -o - | %filecheck %s
//
// Row-sum reduction kernel: real Triton IR (get_program_id, masked load with
// other, offset arithmetic) feeding tt.reduce to scalar, then scalar store to
// output_ptr + pid.
//
module {
  tt.func public @sum_kernel(
    %input_ptr: !tt.ptr<f32>,
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
    %3 = tt.splat %input_ptr : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
    %4 = tt.addptr %3, %1 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>
    %cst = arith.constant dense<0.000000e+00> : tensor<1024xf32>
    %x = tt.load %4, %mask, %cst : tensor<1024x!tt.ptr<f32>>
    %sum = "tt.reduce"(%x) ({
    ^bb0(%a: f32, %b: f32):
      %add = arith.addf %a, %b : f32
      "tt.reduce.return"(%add) : (f32) -> ()
    }) {axis = 0 : i32} : (tensor<1024xf32>) -> f32
    %out_ptr = tt.addptr %output_ptr, %pid : !tt.ptr<f32>, i32
    tt.store %out_ptr, %sum : !tt.ptr<f32>
    tt.return
  }
}

// CHECK-LABEL: func.func @sum_kernel
// CHECK-SAME: %arg0: !pto.ptr<f32>
// CHECK-SAME: %arg1: !pto.ptr<f32>
// CHECK-SAME: %arg2: i32)
// CHECK-DAG: arith.constant 0 : index
// CHECK: pto.make_tensor_view %arg0
// CHECK: pto.partition_view
// CHECK: pto.alloc_tile
// CHECK: pto.tload ins({{.*}}) outs({{.*}})
// CHECK: pto.treduce_sum {{.*}} -> f32
// CHECK: pto.sstore {{.*}}, %arg1[
// CHECK: return
// CHECK-NOT: tt.
