// KVForge: Fused projection + quantization kernel
// Performs low-rank projection and INT4/INT8 quantization in a single pass.
//
// Input:  kv_cache [num_heads, seq_len, head_dim] (f32 or f16)
//         basis_vt [num_heads, rank, head_dim] (f32)
// Output: quantized [num_heads, seq_len, rank] (packed INT4 or INT8)
//         scales/mins per quantization group

#include <cuda_runtime.h>
#include <cstdint>

extern "C" {

__global__ void fused_project_quantize_int4_kernel(
    const float* __restrict__ input,    // [num_heads, seq_len, head_dim]
    const float* __restrict__ basis_vt, // [num_heads, rank, head_dim]
    uint8_t* __restrict__ output,       // packed INT4 output
    float* __restrict__ scales,         // per-group scales
    float* __restrict__ mins,           // per-group minimums
    int num_heads,
    int seq_len,
    int head_dim,
    int rank
) {
    // TODO: Implement fused projection + INT4 quantization
    // 1. Project: out[h,t,r] = sum_c(input[h,t,c] * basis_vt[h,r,c])
    // 2. Find per-group min/max
    // 3. Quantize to INT4 and pack two values per byte
}

__global__ void fused_project_quantize_int8_kernel(
    const float* __restrict__ input,
    const float* __restrict__ basis_vt,
    uint8_t* __restrict__ output,
    float* __restrict__ scales,
    float* __restrict__ mins,
    int num_heads,
    int seq_len,
    int head_dim,
    int rank
) {
    // TODO: Implement fused projection + INT8 quantization
}

} // extern "C"
