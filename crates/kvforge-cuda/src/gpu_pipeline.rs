use kvforge_core::error::Result;
use kvforge_core::pipeline::{CompressionPipeline, PipelineConfig};
use kvforge_core::types::{CompressedKV, KVCache};
use kvforge_cpu::CpuCompressionPipeline;

/// GPU compression pipeline.
///
/// Currently delegates all operations to the CPU pipeline.
/// Will be replaced with CUDA kernel calls once kernels are implemented.
pub struct GpuCompressionPipeline {
    cpu_fallback: CpuCompressionPipeline,
}

impl GpuCompressionPipeline {
    pub fn new() -> Self {
        Self {
            cpu_fallback: CpuCompressionPipeline::new(),
        }
    }
}

impl Default for GpuCompressionPipeline {
    fn default() -> Self {
        Self::new()
    }
}

impl CompressionPipeline for GpuCompressionPipeline {
    fn compress(&self, kv: &KVCache, config: &PipelineConfig) -> Result<CompressedKV> {
        // TODO: Route to CUDA kernels when available
        self.cpu_fallback.compress(kv, config)
    }

    fn decompress(&self, compressed: &CompressedKV) -> Result<KVCache> {
        // TODO: Route to CUDA kernels when available
        self.cpu_fallback.decompress(compressed)
    }
}
