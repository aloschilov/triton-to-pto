// Test: Triton fused softmax kernel -> PTO holistic rewrite.
// Structure: grid-stride row loop (get_program_id/get_num_programs), per-row:
// load, reduce(maximumf), subf, exp, reduce(addf), divf, store.
// Based on triton/python/tutorials/02-fused-softmax.py (BLOCK_SIZE=256).
//
// RUN: %triton_to_pto_mlir %s -o - | %filecheck %s
//
module {
  tt.func public @softmax_kernel(
    %output_ptr: !tt.ptr<f32>,
    %input_ptr: !tt.ptr<f32>,
    %input_row_stride: i32,
    %output_row_stride: i32,
    %n_rows: i32,
    %n_cols: i32
  ) attributes {noinline = false} {
    %row_start = tt.get_program_id x : i32
    %row_step = tt.get_num_programs x : i32
    %row_start_idx = arith.index_cast %row_start : i32 to index
    %n_rows_idx = arith.index_cast %n_rows : i32 to index
    %row_step_idx = arith.index_cast %row_step : i32 to index
    scf.for %iv = %row_start_idx to %n_rows_idx step %row_step_idx {
      %row_idx = arith.index_cast %iv : index to i32
      %row_offset = arith.muli %row_idx, %input_row_stride : i32
      %row_start_ptr = tt.addptr %input_ptr, %row_offset : !tt.ptr<f32>, i32
      %col_offsets = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32>
      %0 = tt.splat %row_start_ptr : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>>
      %input_ptrs = tt.addptr %0, %col_offsets : tensor<256x!tt.ptr<f32>>, tensor<256xi32>
      %1 = tt.splat %n_cols : i32 -> tensor<256xi32>
      %mask = arith.cmpi slt, %col_offsets, %1 : tensor<256xi32>
      %cst = arith.constant dense<0.000000e+00> : tensor<256xf32>
      %row = tt.load %input_ptrs, %mask, %cst : tensor<256x!tt.ptr<f32>>
      %row_max = "tt.reduce"(%row) ({
      ^bb0(%arg0: f32, %arg1: f32):
        %m = arith.maximumf %arg0, %arg1 : f32
        "tt.reduce.return"(%m) : (f32) -> ()
      }) {axis = 0 : i32} : (tensor<256xf32>) -> f32
      %2 = tt.splat %row_max : f32 -> tensor<256xf32>
      %row_minus_max = arith.subf %row, %2 : tensor<256xf32>
      %numerator = math.exp %row_minus_max : tensor<256xf32>
      %denominator = "tt.reduce"(%numerator) ({
      ^bb0(%arg0: f32, %arg1: f32):
        %a = arith.addf %arg0, %arg1 : f32
        "tt.reduce.return"(%a) : (f32) -> ()
      }) {axis = 0 : i32} : (tensor<256xf32>) -> f32
      %3 = tt.splat %denominator : f32 -> tensor<256xf32>
      %softmax_output = arith.divf %numerator, %3 : tensor<256xf32>
      %out_row_offset = arith.muli %row_idx, %output_row_stride : i32
      %output_row_start_ptr = tt.addptr %output_ptr, %out_row_offset : !tt.ptr<f32>, i32
      %4 = tt.splat %output_row_start_ptr : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>>
      %output_ptrs = tt.addptr %4, %col_offsets : tensor<256x!tt.ptr<f32>>, tensor<256xi32>
      tt.store %output_ptrs, %softmax_output, %mask : tensor<256x!tt.ptr<f32>>
    }
    tt.return
  }
}

// CHECK-LABEL: func.func @softmax_kernel
// CHECK-SAME: %arg0: !pto.ptr<f32>
// CHECK-SAME: %arg1: !pto.ptr<f32>
// CHECK-SAME: %arg2: i32
// CHECK-SAME: %arg3: i32
// CHECK-SAME: %arg4: i32
// CHECK-SAME: %arg5: i32
// CHECK: pto.get_block_idx
// CHECK: pto.get_block_num
// CHECK: scf.for
// CHECK: arith.minui
// CHECK: pto.make_tensor_view %arg1
// CHECK: pto.partition_view
// CHECK: pto.alloc_tile
// CHECK: pto.tload
// CHECK: pto.trowmax
// CHECK: pto.tgetval
// CHECK: pto.tsubs
// CHECK: pto.texp
// CHECK: pto.trowsum
// CHECK: pto.tgetval
// CHECK: pto.tdivs
// CHECK: pto.make_tensor_view %arg0
// CHECK: pto.partition_view
// CHECK: pto.tstore
// CHECK: }
// CHECK: return
// CHECK-NOT: tt.
