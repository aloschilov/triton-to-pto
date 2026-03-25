// Test: Triton matrix multiplication kernel -> PTO holistic rewrite.
// Simplified matmul: C = A @ B, f32, BLOCK_SIZE=32, no activation fusion,
// no L2 grouped ordering, no boundary masking (M, N, K must be multiples
// of 32). Based on triton-lang/triton tutorials/03-matrix-multiplication.py.
//
// RUN: %triton_to_pto_mlir %s -o - | %filecheck %s
//
module {
  tt.func public @matmul_kernel(
    %a_ptr: !tt.ptr<f32>,
    %b_ptr: !tt.ptr<f32>,
    %c_ptr: !tt.ptr<f32>,
    %M: i32, %N: i32, %K: i32,
    %stride_am: i32, %stride_ak: i32,
    %stride_bk: i32, %stride_bn: i32,
    %stride_cm: i32, %stride_cn: i32
  ) attributes {noinline = false} {
    %c32_i32 = arith.constant 32 : i32
    %c0_i32 = arith.constant 0 : i32

    // Map 1D program_id to 2D (pid_m, pid_n) via simple row-major ordering.
    %pid = tt.get_program_id x : i32
    %num_pid_n = arith.ceildivsi %N, %c32_i32 : i32
    %pid_m = arith.divsi %pid, %num_pid_n : i32
    %pid_n = arith.remsi %pid, %num_pid_n : i32

    // Base offsets into A and C rows / B and C columns.
    %offs_am = arith.muli %pid_m, %c32_i32 : i32
    %offs_bn = arith.muli %pid_n, %c32_i32 : i32

    // Row/column index vectors.
    %range_m = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
    %range_n = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
    %range_k = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>

    // Row indices for A / C: offs_am + range_m  (shape: [32])
    %splat_offs_am = tt.splat %offs_am : i32 -> tensor<32xi32>
    %row_idx = arith.addi %splat_offs_am, %range_m : tensor<32xi32>

    // Col indices for B / C: offs_bn + range_n  (shape: [32])
    %splat_offs_bn = tt.splat %offs_bn : i32 -> tensor<32xi32>
    %col_idx = arith.addi %splat_offs_bn, %range_n : tensor<32xi32>

    // Accumulator: tensor<32x32xf32> = 0
    %acc_init = arith.constant dense<0.0> : tensor<32x32xf32>

    // Loop over K in steps of 32.
    %c0 = arith.constant 0 : index
    %K_idx = arith.index_cast %K : i32 to index
    %c32 = arith.constant 32 : index
    %acc_final = scf.for %ki = %c0 to %K_idx step %c32
        iter_args(%acc = %acc_init) -> (tensor<32x32xf32>) {
      %ki_i32 = arith.index_cast %ki : index to i32

      // A block pointers: A[row_idx, ki + range_k]
      //   ptr = a_ptr + row_idx[:,None]*stride_am + (ki+range_k)[None,:]*stride_ak
      %splat_stride_am = tt.splat %stride_am : i32 -> tensor<32xi32>
      %a_row_offsets = arith.muli %row_idx, %splat_stride_am : tensor<32xi32>

      %splat_ki = tt.splat %ki_i32 : i32 -> tensor<32xi32>
      %k_idx = arith.addi %splat_ki, %range_k : tensor<32xi32>
      %splat_stride_ak = tt.splat %stride_ak : i32 -> tensor<32xi32>
      %a_col_offsets = arith.muli %k_idx, %splat_stride_ak : tensor<32xi32>

      // Broadcast to 2D: [32,1] + [1,32] -> [32,32]
      %a_row_2d = tt.expand_dims %a_row_offsets {axis = 1 : i32} : tensor<32xi32> -> tensor<32x1xi32>
      %a_col_2d = tt.expand_dims %a_col_offsets {axis = 0 : i32} : tensor<32xi32> -> tensor<1x32xi32>
      %a_row_bc = tt.broadcast %a_row_2d : tensor<32x1xi32> -> tensor<32x32xi32>
      %a_col_bc = tt.broadcast %a_col_2d : tensor<1x32xi32> -> tensor<32x32xi32>
      %a_offsets = arith.addi %a_row_bc, %a_col_bc : tensor<32x32xi32>
      %a_ptr_splat = tt.splat %a_ptr : !tt.ptr<f32> -> tensor<32x32x!tt.ptr<f32>>
      %a_ptrs = tt.addptr %a_ptr_splat, %a_offsets : tensor<32x32x!tt.ptr<f32>>, tensor<32x32xi32>
      %a_tile = tt.load %a_ptrs : tensor<32x32x!tt.ptr<f32>>

      // B block pointers: B[ki + range_k, col_idx]
      %splat_stride_bk = tt.splat %stride_bk : i32 -> tensor<32xi32>
      %b_row_offsets = arith.muli %k_idx, %splat_stride_bk : tensor<32xi32>
      %splat_stride_bn = tt.splat %stride_bn : i32 -> tensor<32xi32>
      %b_col_offsets = arith.muli %col_idx, %splat_stride_bn : tensor<32xi32>

      %b_row_2d = tt.expand_dims %b_row_offsets {axis = 1 : i32} : tensor<32xi32> -> tensor<32x1xi32>
      %b_col_2d = tt.expand_dims %b_col_offsets {axis = 0 : i32} : tensor<32xi32> -> tensor<1x32xi32>
      %b_row_bc = tt.broadcast %b_row_2d : tensor<32x1xi32> -> tensor<32x32xi32>
      %b_col_bc = tt.broadcast %b_col_2d : tensor<1x32xi32> -> tensor<32x32xi32>
      %b_offsets = arith.addi %b_row_bc, %b_col_bc : tensor<32x32xi32>
      %b_ptr_splat = tt.splat %b_ptr : !tt.ptr<f32> -> tensor<32x32x!tt.ptr<f32>>
      %b_ptrs = tt.addptr %b_ptr_splat, %b_offsets : tensor<32x32x!tt.ptr<f32>>, tensor<32x32xi32>
      %b_tile = tt.load %b_ptrs : tensor<32x32x!tt.ptr<f32>>

      // Accumulate: acc += dot(a_tile, b_tile)
      %new_acc = tt.dot %a_tile, %b_tile, %acc : tensor<32x32xf32> * tensor<32x32xf32> -> tensor<32x32xf32>

      scf.yield %new_acc : tensor<32x32xf32>
    }

    // Store C block: C[row_idx, col_idx] = acc_final
    %splat_stride_cm = tt.splat %stride_cm : i32 -> tensor<32xi32>
    %c_row_offsets = arith.muli %row_idx, %splat_stride_cm : tensor<32xi32>
    %splat_stride_cn = tt.splat %stride_cn : i32 -> tensor<32xi32>
    %c_col_offsets = arith.muli %col_idx, %splat_stride_cn : tensor<32xi32>

    %c_row_2d = tt.expand_dims %c_row_offsets {axis = 1 : i32} : tensor<32xi32> -> tensor<32x1xi32>
    %c_col_2d = tt.expand_dims %c_col_offsets {axis = 0 : i32} : tensor<32xi32> -> tensor<1x32xi32>
    %c_row_bc = tt.broadcast %c_row_2d : tensor<32x1xi32> -> tensor<32x32xi32>
    %c_col_bc = tt.broadcast %c_col_2d : tensor<1x32xi32> -> tensor<32x32xi32>
    %c_offsets = arith.addi %c_row_bc, %c_col_bc : tensor<32x32xi32>
    %c_ptr_splat = tt.splat %c_ptr : !tt.ptr<f32> -> tensor<32x32x!tt.ptr<f32>>
    %c_ptrs = tt.addptr %c_ptr_splat, %c_offsets : tensor<32x32x!tt.ptr<f32>>, tensor<32x32xi32>
    tt.store %c_ptrs, %acc_final : tensor<32x32x!tt.ptr<f32>>

    tt.return
  }
}

// CHECK-LABEL: func.func @matmul_kernel
// CHECK-SAME: %arg0: !pto.ptr<f32>
// CHECK-SAME: %arg1: !pto.ptr<f32>
// CHECK-SAME: %arg2: !pto.ptr<f32>
// CHECK-SAME: %arg3: i32
// CHECK: pto.get_block_idx
// CHECK: pto.make_tensor_view %arg0
// CHECK: pto.make_tensor_view %arg1
// CHECK: pto.make_tensor_view %arg2
// CHECK: pto.alloc_tile
// CHECK: pto.alloc_tile
// CHECK: pto.alloc_tile
// CHECK: pto.alloc_tile
// CHECK: pto.alloc_tile
// CHECK: scf.for
// CHECK: pto.partition_view
// CHECK: pto.partition_view
// CHECK: pto.tload
// CHECK: pto.tload
// CHECK: pto.set_flag
// CHECK: pto.wait_flag
// CHECK: pto.tmov
// CHECK: pto.tmov
// CHECK: pto.set_flag
// CHECK: pto.wait_flag
// CHECK: pto.tmatmul
// CHECK: pto.set_flag
// CHECK: pto.wait_flag
// CHECK: }
// CHECK: pto.set_flag
// CHECK: pto.wait_flag
// CHECK: pto.partition_view
// CHECK: pto.tstore
// CHECK: return
// CHECK-NOT: tt.
