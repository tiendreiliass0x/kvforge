// KVForge: Fused entropy decode + dequantization kernel
// Decodes rANS compressed stream and dequantizes in a single pass.
//
// Input:  compressed [M] (uint32_t) — rANS compressed stream
//         frequencies [256] (uint32_t) — symbol frequency table
//         scales, mins — quantization parameters
// Output: output [N] (float) — dequantized values

#include <cuda_runtime.h>
#include <cstdint>

extern "C" {

__global__ void entropy_decode_dequant_int4_kernel(
    const uint32_t* __restrict__ compressed,
    const uint32_t* __restrict__ frequencies,
    const float* __restrict__ scales,
    const float* __restrict__ mins,
    float* __restrict__ output,
    int compressed_len,
    int output_len,
    int group_size
) {
    // TODO: Implement fused entropy decode + INT4 dequantization
    // 1. Decode rANS stream to recover packed INT4 bytes
    // 2. Unpack INT4 pairs
    // 3. Dequantize: x = q * scale + min
}

__global__ void entropy_decode_dequant_int8_kernel(
    const uint32_t* __restrict__ compressed,
    const uint32_t* __restrict__ frequencies,
    const float* __restrict__ scales,
    const float* __restrict__ mins,
    float* __restrict__ output,
    int compressed_len,
    int output_len,
    int group_size
) {
    // TODO: Implement fused entropy decode + INT8 dequantization
}

} // extern "C"
