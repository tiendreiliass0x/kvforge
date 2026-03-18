use crate::error::Result;
use crate::pipeline::CompressionPreset;
use serde::{Deserialize, Serialize};

/// Network bandwidth tiers for cost estimation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BandwidthTier {
    /// PCIe / NVLink (~25 GB/s)
    Local,
    /// InfiniBand / RDMA (~12.5 GB/s)
    Rdma,
    /// 100 Gbps Ethernet (~10 GB/s)
    Ethernet100G,
    /// 25 Gbps Ethernet (~2.5 GB/s)
    Ethernet25G,
    /// 10 Gbps Ethernet (~1 GB/s)
    Ethernet10G,
    /// WAN / cross-region (~100 MB/s)
    Wan,
}

impl BandwidthTier {
    /// Effective throughput in bytes per microsecond.
    pub fn bytes_per_us(&self) -> f64 {
        match self {
            BandwidthTier::Local => 25_000.0,       // 25 GB/s
            BandwidthTier::Rdma => 12_500.0,        // 12.5 GB/s
            BandwidthTier::Ethernet100G => 10_000.0, // 10 GB/s
            BandwidthTier::Ethernet25G => 2_500.0,   // 2.5 GB/s
            BandwidthTier::Ethernet10G => 1_000.0,   // 1 GB/s
            BandwidthTier::Wan => 100.0,             // 100 MB/s
        }
    }
}

/// Cost estimate for a particular compression preset at a given bandwidth.
#[derive(Debug, Clone)]
pub struct CostEstimate {
    pub preset: CompressionPreset,
    pub compress_us: f64,
    pub transfer_us: f64,
    pub decompress_us: f64,
    pub total_us: f64,
    pub compression_ratio: f64,
}

/// Cost model for selecting optimal compression presets.
pub trait CostModel: Send + Sync {
    /// Estimate end-to-end latency for a given data size, preset, and bandwidth.
    fn estimate(
        &self,
        original_bytes: usize,
        preset: CompressionPreset,
        bandwidth: BandwidthTier,
    ) -> Result<CostEstimate>;

    /// Select the best preset for a given scenario.
    /// Returns presets sorted by total latency (best first).
    fn select_codec(
        &self,
        original_bytes: usize,
        bandwidth: BandwidthTier,
        min_quality: Option<f64>,
    ) -> Result<Vec<CostEstimate>>;
}
