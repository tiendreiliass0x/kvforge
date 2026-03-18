use kvforge_core::cost::{BandwidthTier, CostModel};
use kvforge_core::pipeline::CompressionPreset;
use kvforge_cpu::CpuCostModel;

#[test]
fn test_cost_estimates_positive() {
    let model = CpuCostModel::new();
    let presets = [
        CompressionPreset::Conservative,
        CompressionPreset::Balanced,
        CompressionPreset::Aggressive,
    ];
    let tiers = [
        BandwidthTier::Local,
        BandwidthTier::Rdma,
        BandwidthTier::Ethernet100G,
        BandwidthTier::Ethernet25G,
        BandwidthTier::Ethernet10G,
        BandwidthTier::Wan,
    ];

    for &preset in &presets {
        for &tier in &tiers {
            let est = model.estimate(1_000_000, preset, tier).unwrap();
            assert!(est.compress_us > 0.0);
            assert!(est.transfer_us > 0.0);
            assert!(est.decompress_us > 0.0);
            assert!(
                (est.total_us - (est.compress_us + est.transfer_us + est.decompress_us)).abs()
                    < 1e-6
            );
        }
    }
}

#[test]
fn test_transfer_decreases_with_compression() {
    let model = CpuCostModel::new();
    let tier = BandwidthTier::Ethernet10G;

    let conservative = model
        .estimate(10_000_000, CompressionPreset::Conservative, tier)
        .unwrap();
    let aggressive = model
        .estimate(10_000_000, CompressionPreset::Aggressive, tier)
        .unwrap();

    // More compression = less transfer time
    assert!(
        aggressive.transfer_us < conservative.transfer_us,
        "Aggressive transfer ({:.1} us) should be less than conservative ({:.1} us)",
        aggressive.transfer_us,
        conservative.transfer_us
    );
}

#[test]
fn test_local_favors_simple() {
    let model = CpuCostModel::new();
    // On local bus (very high bandwidth), simple is best
    let estimates = model
        .select_codec(1_000_000, BandwidthTier::Local, None)
        .unwrap();

    assert_eq!(estimates[0].preset, CompressionPreset::Conservative);
}

#[test]
fn test_quality_constraint_filters() {
    let model = CpuCostModel::new();

    // High quality constraint should filter out aggressive options
    let high_quality = model
        .select_codec(1_000_000, BandwidthTier::Wan, Some(0.9))
        .unwrap();

    for est in &high_quality {
        // Only conservative (quality=0.95) should remain
        assert_eq!(est.preset, CompressionPreset::Conservative);
    }
}

#[test]
fn test_select_codec_returns_sorted() {
    let model = CpuCostModel::new();

    for &tier in &[BandwidthTier::Local, BandwidthTier::Wan, BandwidthTier::Ethernet10G] {
        let estimates = model
            .select_codec(5_000_000, tier, None)
            .unwrap();

        for w in estimates.windows(2) {
            assert!(
                w[0].total_us <= w[1].total_us,
                "Results not sorted: {} > {}",
                w[0].total_us,
                w[1].total_us
            );
        }
    }
}
