use kvforge_core::codec::*;
use kvforge_core::error::Result;
use kvforge_core::pipeline::{CompressionPipeline, PipelineConfig};
use kvforge_core::types::*;
use ndarray::Array3;

use crate::calibration::CpuSensitivityCalibrator;
use crate::entropy::CpuEntropyCodec;
use crate::projection::CpuProjector;
use crate::quantize::CpuQuantizer;

/// CPU compression pipeline chaining calibration → projection → quantization → entropy coding.
pub struct CpuCompressionPipeline {
    calibrator: CpuSensitivityCalibrator,
    projector: CpuProjector,
    quantizer: CpuQuantizer,
    entropy_codec: CpuEntropyCodec,
}

impl CpuCompressionPipeline {
    pub fn new() -> Self {
        Self {
            calibrator: CpuSensitivityCalibrator::new(),
            projector: CpuProjector::new(),
            quantizer: CpuQuantizer::new(),
            entropy_codec: CpuEntropyCodec::new(),
        }
    }
}

impl Default for CpuCompressionPipeline {
    fn default() -> Self {
        Self::new()
    }
}

/// Compress a single tensor (keys or values) through the pipeline stages.
fn compress_tensor(
    data: &Array3<f32>,
    quant_config: &QuantizeConfig,
    projection_config: Option<&ProjectionConfig>,
    entropy_config: &EntropyConfig,
    projector: &CpuProjector,
    quantizer: &CpuQuantizer,
    entropy_codec: &CpuEntropyCodec,
) -> Result<CompressedTensor> {
    // Stage 1: Projection (optional)
    let (to_quantize, projection_state) = if let Some(proj_config) = projection_config {
        let (projected, state) = projector.project(data, proj_config)?;
        (projected, Some(state))
    } else {
        (data.clone(), None)
    };

    // Stage 2: Quantize
    let (quantized_data, quant_params) = quantizer.quantize(&to_quantize, quant_config)?;

    // Stage 3: Entropy coding (optional)
    let (final_data, entropy_state) = entropy_codec.encode(&quantized_data, entropy_config)?;
    let entropy_state = if entropy_config.enabled {
        Some(entropy_state)
    } else {
        None
    };

    Ok(CompressedTensor {
        quantized_data: final_data,
        quant_params,
        projection_state,
        entropy_state,
    })
}

/// Decompress a single tensor back to f32.
fn decompress_tensor(
    compressed: &CompressedTensor,
    projector: &CpuProjector,
    quantizer: &CpuQuantizer,
    entropy_codec: &CpuEntropyCodec,
) -> Result<Array3<f32>> {
    // Stage 1: Entropy decode
    let quantized_data = if let Some(ref entropy_state) = compressed.entropy_state {
        entropy_codec.decode(&compressed.quantized_data, entropy_state)?
    } else {
        compressed.quantized_data.clone()
    };

    // Stage 2: Dequantize
    let dequantized = quantizer.dequantize(&quantized_data, &compressed.quant_params)?;

    // Stage 3: Reconstruct from projection
    if let Some(ref proj_state) = compressed.projection_state {
        projector.reconstruct(&dequantized, proj_state)
    } else {
        Ok(dequantized)
    }
}

impl CompressionPipeline for CpuCompressionPipeline {
    fn compress(&self, kv: &KVCache, config: &PipelineConfig) -> Result<CompressedKV> {
        let original_size_bytes = kv.size_bytes();
        let original_shape = kv.shape();

        // Optional calibration (scores not yet used for adaptive per-head config,
        // but computed for future use)
        if config.enable_calibration {
            let _scores = self.calibrator.calibrate(
                kv,
                &CalibrationConfig::default(),
            );
        }

        let projection_config = if config.enable_projection {
            Some(ProjectionConfig {
                variance_threshold: config.projection_variance_threshold,
                max_rank: None,
            })
        } else {
            None
        };

        let entropy_config = EntropyConfig {
            enabled: config.enable_entropy_coding,
        };

        // Compress keys: per-channel quantization (KIVI)
        let key_quant_config = QuantizeConfig {
            data_type: config.key_data_type,
            axis: QuantAxis::PerChannel,
        };
        let key_data = compress_tensor(
            &kv.keys,
            &key_quant_config,
            projection_config.as_ref(),
            &entropy_config,
            &self.projector,
            &self.quantizer,
            &self.entropy_codec,
        )?;

        // Compress values: per-token quantization (KIVI)
        let value_quant_config = QuantizeConfig {
            data_type: config.value_data_type,
            axis: QuantAxis::PerToken,
        };
        let value_data = compress_tensor(
            &kv.values,
            &value_quant_config,
            projection_config.as_ref(),
            &entropy_config,
            &self.projector,
            &self.quantizer,
            &self.entropy_codec,
        )?;

        Ok(CompressedKV {
            layer_idx: kv.layer_idx,
            original_shape,
            key_data,
            value_data,
            original_size_bytes,
        })
    }

    fn decompress(&self, compressed: &CompressedKV) -> Result<KVCache> {
        let keys = decompress_tensor(
            &compressed.key_data,
            &self.projector,
            &self.quantizer,
            &self.entropy_codec,
        )?;

        let values = decompress_tensor(
            &compressed.value_data,
            &self.projector,
            &self.quantizer,
            &self.entropy_codec,
        )?;

        Ok(KVCache {
            keys,
            values,
            layer_idx: compressed.layer_idx,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kvforge_core::pipeline::CompressionPreset;
    use rand::Rng;

    fn random_kv(num_heads: usize, seq_len: usize, head_dim: usize) -> KVCache {
        let mut rng = rand::thread_rng();
        let keys = Array3::from_shape_fn((num_heads, seq_len, head_dim), |_| {
            rng.gen_range(-1.0..1.0)
        });
        let values = Array3::from_shape_fn((num_heads, seq_len, head_dim), |_| {
            rng.gen_range(-1.0..1.0)
        });
        KVCache::new(keys, values, 0)
    }

    #[test]
    fn test_conservative_roundtrip() {
        let pipeline = CpuCompressionPipeline::new();
        let kv = random_kv(4, 128, 32);
        let config = PipelineConfig::from_preset(CompressionPreset::Conservative);

        let compressed = pipeline.compress(&kv, &config).unwrap();
        let restored = pipeline.decompress(&compressed).unwrap();

        assert_eq!(restored.keys.shape(), kv.keys.shape());
        assert_eq!(restored.values.shape(), kv.values.shape());

        let key_mse = (&kv.keys - &restored.keys).mapv(|x| x * x).mean().unwrap();
        let val_mse = (&kv.values - &restored.values).mapv(|x| x * x).mean().unwrap();
        assert!(key_mse < 0.001, "Conservative key MSE too high: {}", key_mse);
        assert!(val_mse < 0.001, "Conservative val MSE too high: {}", val_mse);
    }

    #[test]
    fn test_balanced_roundtrip() {
        let pipeline = CpuCompressionPipeline::new();
        let kv = random_kv(4, 128, 32);
        let config = PipelineConfig::from_preset(CompressionPreset::Balanced);

        let compressed = pipeline.compress(&kv, &config).unwrap();
        let restored = pipeline.decompress(&compressed).unwrap();

        assert_eq!(restored.keys.shape(), kv.keys.shape());

        let key_mse = (&kv.keys - &restored.keys).mapv(|x| x * x).mean().unwrap();
        let val_mse = (&kv.values - &restored.values).mapv(|x| x * x).mean().unwrap();
        assert!(key_mse < 0.1, "Balanced key MSE too high: {}", key_mse);
        assert!(val_mse < 0.1, "Balanced val MSE too high: {}", val_mse);
    }

    #[test]
    fn test_aggressive_roundtrip() {
        let pipeline = CpuCompressionPipeline::new();
        let kv = random_kv(4, 128, 32);
        let config = PipelineConfig::from_preset(CompressionPreset::Aggressive);

        let compressed = pipeline.compress(&kv, &config).unwrap();
        let restored = pipeline.decompress(&compressed).unwrap();

        assert_eq!(restored.keys.shape(), kv.keys.shape());

        // Aggressive is lossy, but should still produce reasonable output
        let key_mse = (&kv.keys - &restored.keys).mapv(|x| x * x).mean().unwrap();
        let val_mse = (&kv.values - &restored.values).mapv(|x| x * x).mean().unwrap();
        assert!(key_mse < 1.0, "Aggressive key MSE unreasonably high: {}", key_mse);
        assert!(val_mse < 1.0, "Aggressive val MSE unreasonably high: {}", val_mse);
    }

    #[test]
    fn test_compression_ratio() {
        let pipeline = CpuCompressionPipeline::new();
        // Use a large tensor so projection state overhead is amortized
        let kv = random_kv(8, 1024, 128);

        let conservative = pipeline
            .compress(&kv, &PipelineConfig::from_preset(CompressionPreset::Conservative))
            .unwrap();
        let balanced = pipeline
            .compress(&kv, &PipelineConfig::from_preset(CompressionPreset::Balanced))
            .unwrap();

        let conservative_ratio = conservative.compression_ratio();
        let balanced_ratio = balanced.compression_ratio();

        // Balanced should compress more than conservative
        assert!(
            balanced_ratio > conservative_ratio,
            "Balanced ({:.1}x) should compress more than conservative ({:.1}x)",
            balanced_ratio,
            conservative_ratio
        );
    }
}
