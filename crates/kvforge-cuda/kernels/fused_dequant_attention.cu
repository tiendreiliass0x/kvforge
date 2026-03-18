// KVForge: Fused dequantization + attention kernel
// Dequantizes KV cache and computes attention in a single pass,
// avoiding materialization of the full f32 KV cache.
//
// Input:  quantized_k [num_heads, seq_len, rank_or_dim] (packed INT4/INT8)
//         quantized_v [num_heads, seq_len, rank_or_dim] (packed INT4/INT8)
//         query [num_heads, 1, head_dim] (f32)
//         k_scales, k_mins, v_scales, v_mins
//         basis_vt_k, basis_vt_v (optional, for projection reconstruction)
// Output: attn_output [num_heads, 1, head_dim] (f32)

#include <cuda_runtime.h>
#include <cstdint>

extern "C" {

__global__ void fused_dequant_attention_int4_kernel(
    const uint8_t* __restrict__ quantized_k,
    const uint8_t* __restrict__ quantized_v,
    const float* __restrict__ query,
    const float* __restrict__ k_scales,
    const float* __restrict__ k_mins,
    const float* __restrict__ v_scales,
    const float* __restrict__ v_mins,
    float* __restrict__ output,
    int num_heads,
    int seq_len,
    int head_dim,
    float scale_factor
) {
    // TODO: Implement fused dequant + attention
    // 1. For each head: dequantize K row on-the-fly
    // 2. Compute Q @ K^T (dot product with dequantized K)
    // 3. Softmax over seq_len
    // 4. Dequantize V row on-the-fly, accumulate weighted sum
}

__global__ void fused_dequant_attention_int8_kernel(
    const uint8_t* __restrict__ quantized_k,
    const uint8_t* __restrict__ quantized_v,
    const float* __restrict__ query,
    const float* __restrict__ k_scales,
    const float* __restrict__ k_mins,
    const float* __restrict__ v_scales,
    const float* __restrict__ v_mins,
    float* __restrict__ output,
    int num_heads,
    int seq_len,
    int head_dim,
    float scale_factor
) {
    // TODO: Implement fused dequant + attention for INT8
}

} // extern "C"
