use thiserror::Error;

#[derive(Debug, Error)]
pub enum KvForgeError {
    #[error("Shape mismatch: expected {expected}, got {got}")]
    ShapeMismatch { expected: String, got: String },

    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),

    #[error("Quantization error: {0}")]
    QuantizationError(String),

    #[error("Projection error: {0}")]
    ProjectionError(String),

    #[error("Entropy coding error: {0}")]
    EntropyError(String),

    #[error("Serialization error: {0}")]
    SerializationError(String),

    #[error("CUDA error: {0}")]
    CudaError(String),

    #[error("Pipeline error: {0}")]
    PipelineError(String),
}

pub type Result<T> = std::result::Result<T, KvForgeError>;
