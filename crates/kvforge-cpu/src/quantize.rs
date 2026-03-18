use kvforge_core::codec::{QuantizeConfig, Quantizer};
use kvforge_core::error::{KvForgeError, Result};
use kvforge_core::types::{DataType, QuantAxis, QuantParams};
use ndarray::Array3;

/// CPU quantizer implementing KIVI-style asymmetric quantization.
///
/// - Keys: per-channel (along head_dim, axis 2) — min/max per column
/// - Values: per-token (along seq_len, axis 1) — min/max per row
///
/// INT4 packing: two values per byte, lower nibble = first value.
#[derive(Debug, Clone)]
pub struct CpuQuantizer;

impl CpuQuantizer {
    pub fn new() -> Self {
        Self
    }
}

impl Default for CpuQuantizer {
    fn default() -> Self {
        Self::new()
    }
}

/// Number of quantization levels for a data type.
fn quant_levels(dt: DataType) -> f32 {
    match dt {
        DataType::INT4 => 15.0, // 0..15
        DataType::INT8 => 255.0, // 0..255
        DataType::F32 => 1.0,
    }
}

/// Pack two INT4 values into a single byte. Lower nibble = first value.
fn pack_int4(a: u8, b: u8) -> u8 {
    (a & 0x0F) | ((b & 0x0F) << 4)
}

/// Unpack a byte into two INT4 values. Returns (lower, upper).
fn unpack_int4(byte: u8) -> (u8, u8) {
    (byte & 0x0F, (byte >> 4) & 0x0F)
}

impl Quantizer for CpuQuantizer {
    fn quantize(
        &self,
        data: &Array3<f32>,
        config: &QuantizeConfig,
    ) -> Result<(Vec<u8>, QuantParams)> {
        if config.data_type == DataType::F32 {
            return Err(KvForgeError::QuantizationError(
                "F32 data type does not need quantization".into(),
            ));
        }

        let shape = data.shape().to_vec();
        let num_heads = shape[0];
        let seq_len = shape[1];
        let head_dim = shape[2];

        let levels = quant_levels(config.data_type);

        // Compute per-group min/max depending on axis
        let (scales, mins, quantized_vals) = match config.axis {
            QuantAxis::PerChannel => {
                // Per-channel: one scale/min per (head, channel).
                // Channel = head_dim dimension (axis 2).
                // For each head h and each channel c, find min/max across all tokens.
                let num_params = num_heads * head_dim;
                let mut scales = Vec::with_capacity(num_params);
                let mut mins = Vec::with_capacity(num_params);

                // Quantize all values
                let total_vals = num_heads * seq_len * head_dim;
                let mut qvals = vec![0u8; total_vals];

                for h in 0..num_heads {
                    for c in 0..head_dim {
                        let mut col_min = f32::INFINITY;
                        let mut col_max = f32::NEG_INFINITY;
                        for t in 0..seq_len {
                            let v = data[[h, t, c]];
                            col_min = col_min.min(v);
                            col_max = col_max.max(v);
                        }
                        let range = col_max - col_min;
                        let scale = if range < 1e-10 { 1.0 } else { range / levels };
                        scales.push(scale);
                        mins.push(col_min);

                        for t in 0..seq_len {
                            let v = data[[h, t, c]];
                            let q = ((v - col_min) / scale).round().clamp(0.0, levels) as u8;
                            qvals[h * seq_len * head_dim + t * head_dim + c] = q;
                        }
                    }
                }
                (scales, mins, qvals)
            }
            QuantAxis::PerToken => {
                // Per-token: one scale/min per (head, token).
                // Token = seq_len dimension (axis 1).
                let num_params = num_heads * seq_len;
                let mut scales = Vec::with_capacity(num_params);
                let mut mins = Vec::with_capacity(num_params);

                let total_vals = num_heads * seq_len * head_dim;
                let mut qvals = vec![0u8; total_vals];

                for h in 0..num_heads {
                    for t in 0..seq_len {
                        let row = data.slice(ndarray::s![h, t, ..]);
                        let row_min = row.fold(f32::INFINITY, |a, &b| a.min(b));
                        let row_max = row.fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                        let range = row_max - row_min;
                        let scale = if range < 1e-10 { 1.0 } else { range / levels };
                        scales.push(scale);
                        mins.push(row_min);

                        for c in 0..head_dim {
                            let v = data[[h, t, c]];
                            let q = ((v - row_min) / scale).round().clamp(0.0, levels) as u8;
                            qvals[h * seq_len * head_dim + t * head_dim + c] = q;
                        }
                    }
                }
                (scales, mins, qvals)
            }
        };

        // Pack into output bytes
        let packed = match config.data_type {
            DataType::INT8 => quantized_vals,
            DataType::INT4 => {
                // Pack pairs of values. If odd length, last value goes solo in low nibble.
                let len = quantized_vals.len();
                let packed_len = len.div_ceil(2);
                let mut packed = Vec::with_capacity(packed_len);
                let mut i = 0;
                while i + 1 < len {
                    packed.push(pack_int4(quantized_vals[i], quantized_vals[i + 1]));
                    i += 2;
                }
                if i < len {
                    packed.push(pack_int4(quantized_vals[i], 0));
                }
                packed
            }
            DataType::F32 => unreachable!(),
        };

        let params = QuantParams {
            scales,
            mins,
            data_type: config.data_type,
            axis: config.axis,
            shape,
        };

        Ok((packed, params))
    }

    fn dequantize(&self, data: &[u8], params: &QuantParams) -> Result<Array3<f32>> {
        let num_heads = params.shape[0];
        let seq_len = params.shape[1];
        let head_dim = params.shape[2];
        let total = num_heads * seq_len * head_dim;

        // Unpack bytes to quantized values
        let qvals: Vec<u8> = match params.data_type {
            DataType::INT8 => data.to_vec(),
            DataType::INT4 => {
                let mut vals = Vec::with_capacity(total);
                for &byte in data {
                    let (lo, hi) = unpack_int4(byte);
                    vals.push(lo);
                    vals.push(hi);
                }
                vals.truncate(total);
                vals
            }
            DataType::F32 => {
                return Err(KvForgeError::QuantizationError(
                    "Cannot dequantize F32 data type".into(),
                ));
            }
        };

        if qvals.len() != total {
            return Err(KvForgeError::QuantizationError(format!(
                "Expected {} values, got {}",
                total,
                qvals.len()
            )));
        }

        let mut result = Array3::<f32>::zeros((num_heads, seq_len, head_dim));

        match params.axis {
            QuantAxis::PerChannel => {
                // scales/mins indexed by [h * head_dim + c]
                for h in 0..num_heads {
                    for t in 0..seq_len {
                        for c in 0..head_dim {
                            let param_idx = h * head_dim + c;
                            let val_idx = h * seq_len * head_dim + t * head_dim + c;
                            let q = qvals[val_idx] as f32;
                            result[[h, t, c]] = q * params.scales[param_idx] + params.mins[param_idx];
                        }
                    }
                }
            }
            QuantAxis::PerToken => {
                // scales/mins indexed by [h * seq_len + t]
                for h in 0..num_heads {
                    for t in 0..seq_len {
                        let param_idx = h * seq_len + t;
                        for c in 0..head_dim {
                            let val_idx = h * seq_len * head_dim + t * head_dim + c;
                            let q = qvals[val_idx] as f32;
                            result[[h, t, c]] = q * params.scales[param_idx] + params.mins[param_idx];
                        }
                    }
                }
            }
        }

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array3;
    use rand::Rng;

    fn random_tensor(shape: (usize, usize, usize)) -> Array3<f32> {
        let mut rng = rand::thread_rng();
        Array3::from_shape_fn(shape, |_| rng.gen_range(-1.0..1.0))
    }

    #[test]
    fn test_int4_packing() {
        assert_eq!(pack_int4(3, 12), 0xC3);
        let (lo, hi) = unpack_int4(0xC3);
        assert_eq!(lo, 3);
        assert_eq!(hi, 12);

        // Edge cases
        assert_eq!(pack_int4(0, 0), 0x00);
        assert_eq!(pack_int4(15, 15), 0xFF);
        let (lo, hi) = unpack_int4(0xFF);
        assert_eq!(lo, 15);
        assert_eq!(hi, 15);
    }

    #[test]
    fn test_roundtrip_int8_per_channel() {
        let data = random_tensor((4, 64, 32));
        let quantizer = CpuQuantizer::new();
        let config = QuantizeConfig::keys_int8();

        let (packed, params) = quantizer.quantize(&data, &config).unwrap();
        let restored = quantizer.dequantize(&packed, &params).unwrap();

        // INT8 per-channel should have low error
        let mse = (&data - &restored).mapv(|x| x * x).mean().unwrap();
        assert!(mse < 0.001, "INT8 per-channel MSE too high: {}", mse);
    }

    #[test]
    fn test_roundtrip_int4_per_channel() {
        let data = random_tensor((4, 64, 32));
        let quantizer = CpuQuantizer::new();
        let config = QuantizeConfig::keys_int4();

        let (packed, params) = quantizer.quantize(&data, &config).unwrap();
        let restored = quantizer.dequantize(&packed, &params).unwrap();

        let mse = (&data - &restored).mapv(|x| x * x).mean().unwrap();
        assert!(mse < 0.01, "INT4 per-channel MSE too high: {}", mse);
    }

    #[test]
    fn test_roundtrip_int8_per_token() {
        let data = random_tensor((4, 64, 32));
        let quantizer = CpuQuantizer::new();
        let config = QuantizeConfig::values_int8();

        let (packed, params) = quantizer.quantize(&data, &config).unwrap();
        let restored = quantizer.dequantize(&packed, &params).unwrap();

        let mse = (&data - &restored).mapv(|x| x * x).mean().unwrap();
        assert!(mse < 0.001, "INT8 per-token MSE too high: {}", mse);
    }

    #[test]
    fn test_roundtrip_int4_per_token() {
        let data = random_tensor((4, 64, 32));
        let quantizer = CpuQuantizer::new();
        let config = QuantizeConfig::values_int4();

        let (packed, params) = quantizer.quantize(&data, &config).unwrap();
        let restored = quantizer.dequantize(&packed, &params).unwrap();

        let mse = (&data - &restored).mapv(|x| x * x).mean().unwrap();
        assert!(mse < 0.01, "INT4 per-token MSE too high: {}", mse);
    }

    #[test]
    fn test_kivi_asymmetric() {
        // Keys use per-channel, values use per-token
        let keys = random_tensor((2, 16, 8));
        let values = random_tensor((2, 16, 8));
        let quantizer = CpuQuantizer::new();

        let (k_packed, k_params) = quantizer
            .quantize(&keys, &QuantizeConfig::keys_int4())
            .unwrap();
        let (v_packed, v_params) = quantizer
            .quantize(&values, &QuantizeConfig::values_int4())
            .unwrap();

        assert_eq!(k_params.axis, QuantAxis::PerChannel);
        assert_eq!(v_params.axis, QuantAxis::PerToken);

        // Per-channel: num_heads * head_dim params
        assert_eq!(k_params.scales.len(), 2 * 8);
        // Per-token: num_heads * seq_len params
        assert_eq!(v_params.scales.len(), 2 * 16);

        let k_restored = quantizer.dequantize(&k_packed, &k_params).unwrap();
        let v_restored = quantizer.dequantize(&v_packed, &v_params).unwrap();

        assert_eq!(k_restored.shape(), keys.shape());
        assert_eq!(v_restored.shape(), values.shape());
    }

    #[test]
    fn test_int4_compression_ratio() {
        let data = random_tensor((4, 128, 64));
        let quantizer = CpuQuantizer::new();
        let (packed, _) = quantizer
            .quantize(&data, &QuantizeConfig::keys_int4())
            .unwrap();

        let original_bytes = data.len() * 4; // f32
        let packed_bytes = packed.len();
        let ratio = original_bytes as f64 / packed_bytes as f64;
        // INT4 should give ~8x compression on raw data (4 bytes -> 0.5 bytes)
        assert!(ratio > 7.0, "INT4 compression ratio too low: {}", ratio);
    }
}
