use kvforge_core::cost::{BandwidthTier, CostEstimate, CostModel};
use kvforge_core::error::Result;
use kvforge_core::pipeline::CompressionPreset;

/// CPU cost model using table-driven latency estimates.
///
/// For each preset, stores expected compression ratio and
/// per-byte compress/decompress costs.
#[derive(Debug, Clone)]
pub struct CpuCostModel {
    entries: Vec<PresetCost>,
}

#[derive(Debug, Clone)]
struct PresetCost {
    preset: CompressionPreset,
    compression_ratio: f64,
    /// Microseconds per byte to compress.
    compress_cost_per_byte: f64,
    /// Microseconds per byte to decompress.
    decompress_cost_per_byte: f64,
    /// Quality score [0, 1] where 1 = lossless.
    quality: f64,
}

impl CpuCostModel {
    pub fn new() -> Self {
        Self {
            entries: vec![
                PresetCost {
                    preset: CompressionPreset::Conservative,
                    compression_ratio: 4.0,
                    compress_cost_per_byte: 0.0005,    // ~0.5 ns/byte (simple INT8 quant)
                    decompress_cost_per_byte: 0.0003,  // ~0.3 ns/byte
                    quality: 0.95,
                },
                PresetCost {
                    preset: CompressionPreset::Balanced,
                    compression_ratio: 12.0,
                    compress_cost_per_byte: 0.002,     // ~2 ns/byte (SVD + INT4)
                    decompress_cost_per_byte: 0.001,   // ~1 ns/byte
                    quality: 0.80,
                },
                PresetCost {
                    preset: CompressionPreset::Aggressive,
                    compression_ratio: 20.0,
                    compress_cost_per_byte: 0.004,     // ~4 ns/byte (SVD + INT4 + entropy)
                    decompress_cost_per_byte: 0.002,   // ~2 ns/byte
                    quality: 0.60,
                },
            ],
        }
    }
}

impl Default for CpuCostModel {
    fn default() -> Self {
        Self::new()
    }
}

impl CostModel for CpuCostModel {
    fn estimate(
        &self,
        original_bytes: usize,
        preset: CompressionPreset,
        bandwidth: BandwidthTier,
    ) -> Result<CostEstimate> {
        let entry = self
            .entries
            .iter()
            .find(|e| e.preset == preset)
            .unwrap_or(&self.entries[0]); // default to conservative

        let bytes_f64 = original_bytes as f64;
        let compressed_bytes = bytes_f64 / entry.compression_ratio;

        let compress_us = bytes_f64 * entry.compress_cost_per_byte;
        let transfer_us = compressed_bytes / bandwidth.bytes_per_us();
        let decompress_us = bytes_f64 * entry.decompress_cost_per_byte;
        let total_us = compress_us + transfer_us + decompress_us;

        Ok(CostEstimate {
            preset: entry.preset,
            compress_us,
            transfer_us,
            decompress_us,
            total_us,
            compression_ratio: entry.compression_ratio,
        })
    }

    fn select_codec(
        &self,
        original_bytes: usize,
        bandwidth: BandwidthTier,
        min_quality: Option<f64>,
    ) -> Result<Vec<CostEstimate>> {
        let min_q = min_quality.unwrap_or(0.0);

        let mut estimates: Vec<CostEstimate> = self
            .entries
            .iter()
            .filter(|e| e.quality >= min_q)
            .map(|e| {
                self.estimate(original_bytes, e.preset, bandwidth)
                    .unwrap()
            })
            .collect();

        estimates.sort_by(|a, b| a.total_us.partial_cmp(&b.total_us).unwrap());

        Ok(estimates)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_estimate_basic() {
        let model = CpuCostModel::new();
        let est = model
            .estimate(1_000_000, CompressionPreset::Conservative, BandwidthTier::Ethernet10G)
            .unwrap();

        assert!(est.compress_us > 0.0);
        assert!(est.transfer_us > 0.0);
        assert!(est.decompress_us > 0.0);
        assert_eq!(est.total_us, est.compress_us + est.transfer_us + est.decompress_us);
        assert_eq!(est.compression_ratio, 4.0);
    }

    #[test]
    fn test_select_codec_wan() {
        let model = CpuCostModel::new();
        // On WAN (slow bandwidth), more aggressive compression should beat conservative
        // because transfer time dominates
        let estimates = model
            .select_codec(10_000_000, BandwidthTier::Wan, None)
            .unwrap();

        assert!(!estimates.is_empty());
        // All presets should be present
        assert_eq!(estimates.len(), 3);
        // Verify sorted by total_us (best first)
        for w in estimates.windows(2) {
            assert!(w[0].total_us <= w[1].total_us);
        }
        // On WAN, higher compression saves transfer time
        // Conservative transfer: 10M/4 / 100 = 25K us
        // Aggressive transfer: 10M/20 / 100 = 5K us => 20K us savings
        // Verify that aggressive has lower transfer time
        let conservative = estimates.iter().find(|e| e.preset == CompressionPreset::Conservative).unwrap();
        let aggressive = estimates.iter().find(|e| e.preset == CompressionPreset::Aggressive).unwrap();
        assert!(aggressive.transfer_us < conservative.transfer_us);
    }

    #[test]
    fn test_select_codec_local() {
        let model = CpuCostModel::new();
        // On local (fast bandwidth), conservative should win (least overhead)
        let estimates = model
            .select_codec(10_000_000, BandwidthTier::Local, None)
            .unwrap();

        assert!(!estimates.is_empty());
        assert_eq!(estimates[0].preset, CompressionPreset::Conservative);
    }

    #[test]
    fn test_quality_filter() {
        let model = CpuCostModel::new();
        let estimates = model
            .select_codec(10_000_000, BandwidthTier::Wan, Some(0.9))
            .unwrap();

        // Only conservative (quality=0.95) should pass the 0.9 threshold
        assert_eq!(estimates.len(), 1);
        assert_eq!(estimates[0].preset, CompressionPreset::Conservative);
    }
}
