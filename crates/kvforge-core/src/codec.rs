use crate::error::Result;
use crate::types::{DataType, EntropyState, KVCache, ProjectionState, QuantAxis, QuantParams};
use ndarray::Array3;

/// Configuration for sensitivity calibration.
#[derive(Debug, Clone)]
pub struct CalibrationConfig {
    /// Minimum number of tokens before calibration is meaningful.
    pub min_tokens: usize,
}

impl Default for CalibrationConfig {
    fn default() -> Self {
        Self { min_tokens: 64 }
    }
}

/// Per-head sensitivity scores indicating how much each head is affected by compression.
/// Higher score = more sensitive = should be compressed less aggressively.
#[derive(Debug, Clone)]
pub struct SensitivityScores {
    /// One score per head, normalized to [0, 1].
    pub scores: Vec<f32>,
}

/// Analyzes KV cache to determine per-head compression sensitivity.
pub trait SensitivityCalibrator: Send + Sync {
    fn calibrate(&self, kv: &KVCache, config: &CalibrationConfig) -> Result<SensitivityScores>;
}

/// Configuration for low-rank projection.
#[derive(Debug, Clone)]
pub struct ProjectionConfig {
    /// Fraction of cumulative variance to retain (0.0–1.0).
    pub variance_threshold: f64,
    /// Optional hard cap on rank.
    pub max_rank: Option<usize>,
}

impl Default for ProjectionConfig {
    fn default() -> Self {
        Self {
            variance_threshold: 0.95,
            max_rank: None,
        }
    }
}

/// Low-rank projection for dimensionality reduction of KV heads.
pub trait Projector: Send + Sync {
    /// Project a 3D tensor [num_heads, seq_len, head_dim] to lower rank.
    /// Returns (projected [num_heads, seq_len, rank], projection state for reconstruction).
    fn project(
        &self,
        data: &Array3<f32>,
        config: &ProjectionConfig,
    ) -> Result<(Array3<f32>, ProjectionState)>;

    /// Reconstruct from projected data and saved state.
    fn reconstruct(
        &self,
        projected: &Array3<f32>,
        state: &ProjectionState,
    ) -> Result<Array3<f32>>;
}

/// Configuration for quantization.
#[derive(Debug, Clone)]
pub struct QuantizeConfig {
    pub data_type: DataType,
    pub axis: QuantAxis,
}

impl QuantizeConfig {
    pub fn keys_int4() -> Self {
        Self {
            data_type: DataType::INT4,
            axis: QuantAxis::PerChannel,
        }
    }

    pub fn keys_int8() -> Self {
        Self {
            data_type: DataType::INT8,
            axis: QuantAxis::PerChannel,
        }
    }

    pub fn values_int4() -> Self {
        Self {
            data_type: DataType::INT4,
            axis: QuantAxis::PerToken,
        }
    }

    pub fn values_int8() -> Self {
        Self {
            data_type: DataType::INT8,
            axis: QuantAxis::PerToken,
        }
    }
}

/// Quantizer for KV cache tensors.
pub trait Quantizer: Send + Sync {
    /// Quantize a 3D f32 tensor to packed bytes with quantization parameters.
    fn quantize(
        &self,
        data: &Array3<f32>,
        config: &QuantizeConfig,
    ) -> Result<(Vec<u8>, QuantParams)>;

    /// Dequantize packed bytes back to f32 tensor.
    fn dequantize(&self, data: &[u8], params: &QuantParams) -> Result<Array3<f32>>;
}

/// Configuration for entropy coding.
#[derive(Debug, Clone)]
pub struct EntropyConfig {
    /// Whether to use entropy coding at all.
    pub enabled: bool,
}

impl Default for EntropyConfig {
    fn default() -> Self {
        Self { enabled: true }
    }
}

/// Entropy codec for lossless compression of quantized byte streams.
pub trait EntropyCodec: Send + Sync {
    /// Compress a byte stream using entropy coding.
    fn encode(&self, data: &[u8], config: &EntropyConfig) -> Result<(Vec<u8>, EntropyState)>;

    /// Decompress a byte stream using saved state.
    fn decode(&self, data: &[u8], state: &EntropyState) -> Result<Vec<u8>>;
}
