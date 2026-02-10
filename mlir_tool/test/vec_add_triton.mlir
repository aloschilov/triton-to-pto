// RUN: %triton_to_pto_mlir %s -o - | %filecheck %s

// A more realistic Triton-style vector add kernel that uses a 1D range,
// pointer arithmetic, vector loads/stores, and a size argument.
//
module {
  tt.func public @vector_add_kernel(
    %arg0: !tt.ptr<f32>,   // x_ptr
    %arg1: !tt.ptr<f32>,   // y_ptr
    %arg2: !tt.ptr<f32>,   // out_ptr
    %arg3: i32             // n_elements
  ) attributes {noinline = false} {

    // 1. Setup Block Size (e.g., 1024)
    %cst_1024 = arith.constant 1024 : i32

    // 2. Generate Offsets: tl.arange(0, 1024)
    %range = tt.make_range {start = 0 : i32, end = 1024 : i32} : tensor<1024xi32>

    // 3. Pointer Arithmetic for X: x_ptr + offsets
    %x_base = tt.splat %arg0 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
    %x_ptrs = tt.addptr %x_base, %range
      : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>

    // 4. Pointer Arithmetic for Y: y_ptr + offsets
    %y_base = tt.splat %arg1 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
    %y_ptrs = tt.addptr %y_base, %range
      : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>

    // 5. Load Data
    %x_vals = tt.load %x_ptrs : tensor<1024x!tt.ptr<f32>>
    %y_vals = tt.load %y_ptrs : tensor<1024x!tt.ptr<f32>>

    // 6. The Core Vector Addition
    %result = arith.addf %x_vals, %y_vals : tensor<1024xf32>

    // 7. Store Data: out_ptr + offsets
    %out_base = tt.splat %arg2 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
    %out_ptrs = tt.addptr %out_base, %range
      : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>
    tt.store %out_ptrs, %result : tensor<1024x!tt.ptr<f32>>

    tt.return
  }
}

// CHECK-LABEL: tt.func public @vector_add_kernel
// Expect two loads lowered to PTO, one add consuming them, and one store.
// Output uses numeric SSA names and may include attributes on PTO ops.
// CHECK: %[[X:.*]] = "pto.tload"(%{{.*}}) {{.*}} : (tensor<1024x!tt.ptr<f32>>) -> tensor<1024xf32>
// CHECK: %[[Y:.*]] = "pto.tload"(%{{.*}}) {{.*}} : (tensor<1024x!tt.ptr<f32>>) -> tensor<1024xf32>
// CHECK: %[[RES:.*]] = "pto.tadd"(%[[X]], %[[Y]]) {{.*}} : (tensor<1024xf32>, tensor<1024xf32>) -> tensor<1024xf32>
// CHECK: "pto.tstore"(%{{.*}}, %[[RES]]) {{.*}} : (tensor<1024x!tt.ptr<f32>>, tensor<1024xf32>) -> ()

