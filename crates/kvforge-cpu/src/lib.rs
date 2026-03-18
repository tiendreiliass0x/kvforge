pub mod calibration;
pub mod cost;
pub mod entropy;
pub mod pipeline;
pub mod projection;
pub mod quantize;

pub use calibration::CpuSensitivityCalibrator;
pub use cost::CpuCostModel;
pub use entropy::CpuEntropyCodec;
pub use pipeline::CpuCompressionPipeline;
pub use projection::CpuProjector;
pub use quantize::CpuQuantizer;

/// Convenience builder: creates a default CPU compression pipeline.
pub fn default_pipeline() -> CpuCompressionPipeline {
    CpuCompressionPipeline::new()
}
