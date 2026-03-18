use ndarray::Array3;
use serde::{Deserialize, Serialize};

/// A KV cache for a single layer.
///
/// Shape convention: `[num_heads, seq_len, head_dim]`
#[derive(Debug, Clone)]
pub struct KVCache {
    pub keys: Array3<f32>,
    pub values: Array3<f32>,
    pub layer_idx: usize,
}

impl KVCache {
    pub fn new(keys: Array3<f32>, values: Array3<f32>, layer_idx: usize) -> Self {
        assert_eq!(keys.shape(), values.shape(), "keys and values must have the same shape");
        Self { keys, values, layer_idx }
    }

    pub fn shape(&self) -> Shape {
        let s = self.keys.shape();
        Shape {
            num_heads: s[0],
            seq_len: s[1],
            head_dim: s[2],
        }
    }

    /// Total size in bytes if stored as f32.
    pub fn size_bytes(&self) -> usize {
        self.keys.len() * 4 + self.values.len() * 4
    }
}

/// Compressed representation of a KV cache.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressedKV {
    pub layer_idx: usize,
    pub original_shape: Shape,
    pub key_data: CompressedTensor,
    pub value_data: CompressedTensor,
    pub original_size_bytes: usize,
}

impl CompressedKV {
    /// Compressed size in bytes.
    pub fn compressed_size_bytes(&self) -> usize {
        self.key_data.size_bytes() + self.value_data.size_bytes()
    }

    /// Compression ratio (original / compressed).
    pub fn compression_ratio(&self) -> f64 {
        self.original_size_bytes as f64 / self.compressed_size_bytes() as f64
    }
}

/// A compressed tensor with optional projection and entropy coding state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressedTensor {
    pub quantized_data: Vec<u8>,
    pub quant_params: QuantParams,
    pub projection_state: Option<ProjectionState>,
    pub entropy_state: Option<EntropyState>,
}

impl CompressedTensor {
    pub fn size_bytes(&self) -> usize {
        let mut total = self.quantized_data.len();
        total += self.quant_params.size_bytes();
        if let Some(ref ps) = self.projection_state {
            total += ps.size_bytes();
        }
        if let Some(ref es) = self.entropy_state {
            total += es.size_bytes();
        }
        total
    }
}

/// Quantization parameters for a tensor.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantParams {
    pub scales: Vec<f32>,
    pub mins: Vec<f32>,
    pub data_type: DataType,
    pub axis: QuantAxis,
    /// Shape of the tensor before quantization: [num_heads, seq_len, head_dim]
    pub shape: Vec<usize>,
}

impl QuantParams {
    pub fn size_bytes(&self) -> usize {
        (self.scales.len() + self.mins.len()) * 4
    }
}

/// Which axis quantization parameters are computed along.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum QuantAxis {
    /// Per-channel (along head_dim, axis 2) — used for keys in KIVI.
    PerChannel,
    /// Per-token (along seq_len, axis 1) — used for values in KIVI.
    PerToken,
}

/// Quantized data type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DataType {
    /// 4-bit integer, two values packed per byte (lower nibble first).
    INT4,
    /// 8-bit integer.
    INT8,
    /// 32-bit float (no quantization).
    F32,
}

/// State needed to reconstruct from a low-rank projection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectionState {
    /// The right singular vectors (basis): shape [num_heads, rank, head_dim]
    /// Stored as flattened f32.
    pub basis_vt: Vec<f32>,
    /// Rank used per head.
    pub ranks: Vec<usize>,
    pub num_heads: usize,
    pub head_dim: usize,
    /// Projected shape: [num_heads, seq_len, rank]
    pub projected_shape: Vec<usize>,
}

impl ProjectionState {
    pub fn size_bytes(&self) -> usize {
        self.basis_vt.len() * 4 + self.ranks.len() * 8
    }
}

/// State needed to decode entropy-coded data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntropyState {
    pub frequencies: Vec<u32>,
    pub original_len: usize,
}

impl EntropyState {
    pub fn size_bytes(&self) -> usize {
        self.frequencies.len() * 4 + 8
    }
}

/// Shape descriptor for a 3D tensor: [num_heads, seq_len, head_dim].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Shape {
    pub num_heads: usize,
    pub seq_len: usize,
    pub head_dim: usize,
}

impl Shape {
    pub fn num_elements(&self) -> usize {
        self.num_heads * self.seq_len * self.head_dim
    }
}
