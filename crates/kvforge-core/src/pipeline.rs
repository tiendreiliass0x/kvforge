use crate::error::Result;
use crate::types::{CompressedKV, DataType, KVCache};
use serde::{Deserialize, Serialize};

/// Preset compression profiles.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CompressionPreset {
    /// ~4x ratio, minimal quality loss. INT8 quantization, no projection.
    Conservative,
    /// ~12x ratio, moderate quality loss. INT4 quantization + projection.
    Balanced,
    /// ~20x+ ratio, higher quality loss. INT4 + aggressive projection + entropy coding.
    Aggressive,
    /// Automatically selects preset based on bandwidth/latency constraints.
    Adaptive,
}

/// Configuration for the compression pipeline.
#[derive(Debug, Clone)]
pub struct PipelineConfig {
    pub preset: CompressionPreset,
    pub key_data_type: DataType,
    pub value_data_type: DataType,
    pub enable_projection: bool,
    pub projection_variance_threshold: f64,
    pub enable_entropy_coding: bool,
    pub enable_calibration: bool,
}

impl PipelineConfig {
    pub fn from_preset(preset: CompressionPreset) -> Self {
        match preset {
            CompressionPreset::Conservative => Self {
                preset,
                key_data_type: DataType::INT8,
                value_data_type: DataType::INT8,
                enable_projection: false,
                projection_variance_threshold: 0.99,
                enable_entropy_coding: false,
                enable_calibration: false,
            },
            CompressionPreset::Balanced => Self {
                preset,
                key_data_type: DataType::INT4,
                value_data_type: DataType::INT4,
                enable_projection: true,
                projection_variance_threshold: 0.90,
                enable_entropy_coding: false,
                enable_calibration: false,
            },
            CompressionPreset::Aggressive => Self {
                preset,
                key_data_type: DataType::INT4,
                value_data_type: DataType::INT4,
                enable_projection: true,
                projection_variance_threshold: 0.75,
                enable_entropy_coding: true,
                enable_calibration: true,
            },
            CompressionPreset::Adaptive => Self {
                preset,
                key_data_type: DataType::INT4,
                value_data_type: DataType::INT4,
                enable_projection: true,
                projection_variance_threshold: 0.90,
                enable_entropy_coding: true,
                enable_calibration: true,
            },
        }
    }
}

/// The main compression pipeline trait.
pub trait CompressionPipeline: Send + Sync {
    /// Compress a KV cache according to the pipeline configuration.
    fn compress(&self, kv: &KVCache, config: &PipelineConfig) -> Result<CompressedKV>;

    /// Decompress a compressed KV cache back to f32 tensors.
    fn decompress(&self, compressed: &CompressedKV) -> Result<KVCache>;
}
