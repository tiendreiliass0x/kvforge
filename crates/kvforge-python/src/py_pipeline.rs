use pyo3::prelude::*;

use kvforge_core::pipeline::{
    CompressionPipeline as CompressionPipelineTrait, CompressionPreset, PipelineConfig,
};
use kvforge_cpu::CpuCompressionPipeline;

use crate::py_types::{CompressedKV, KVCache};

/// Python wrapper for the CPU compression pipeline.
#[pyclass]
pub struct CompressionPipeline {
    inner: CpuCompressionPipeline,
}

fn parse_preset(preset: &str) -> PyResult<CompressionPreset> {
    match preset.to_lowercase().as_str() {
        "conservative" => Ok(CompressionPreset::Conservative),
        "balanced" => Ok(CompressionPreset::Balanced),
        "aggressive" => Ok(CompressionPreset::Aggressive),
        "adaptive" => Ok(CompressionPreset::Adaptive),
        _ => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Unknown preset '{}'. Choose from: conservative, balanced, aggressive, adaptive",
            preset
        ))),
    }
}

#[pymethods]
impl CompressionPipeline {
    #[new]
    fn new() -> Self {
        Self {
            inner: CpuCompressionPipeline::new(),
        }
    }

    /// Compress a KVCache.
    ///
    /// Args:
    ///     kv: KVCache to compress
    ///     preset: Compression preset ("conservative", "balanced", "aggressive", "adaptive")
    #[pyo3(signature = (kv, preset="balanced"))]
    fn compress(&self, kv: &KVCache, preset: &str) -> PyResult<CompressedKV> {
        let preset = parse_preset(preset)?;
        let config = PipelineConfig::from_preset(preset);
        let compressed = self
            .inner
            .compress(&kv.inner, &config)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{}", e)))?;
        Ok(CompressedKV { inner: compressed })
    }

    /// Decompress a CompressedKV back to KVCache.
    fn decompress(&self, compressed: &CompressedKV) -> PyResult<KVCache> {
        let kv = self
            .inner
            .decompress(&compressed.inner)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{}", e)))?;
        Ok(KVCache { inner: kv })
    }
}

/// Module-level compress function.
#[pyfunction]
#[pyo3(signature = (kv, preset="balanced"))]
pub fn compress(kv: &KVCache, preset: &str) -> PyResult<CompressedKV> {
    let pipeline = CompressionPipeline::new();
    pipeline.compress(kv, preset)
}

/// Module-level decompress function.
#[pyfunction]
pub fn decompress(compressed: &CompressedKV) -> PyResult<KVCache> {
    let pipeline = CompressionPipeline::new();
    pipeline.decompress(compressed)
}
