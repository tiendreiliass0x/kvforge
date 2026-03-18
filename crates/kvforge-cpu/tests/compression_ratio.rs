use kvforge_core::pipeline::{CompressionPipeline, CompressionPreset, PipelineConfig};
use kvforge_core::types::KVCache;
use kvforge_cpu::CpuCompressionPipeline;
use ndarray::Array3;
use rand::Rng;

fn random_kv(num_heads: usize, seq_len: usize, head_dim: usize) -> KVCache {
    let mut rng = rand::thread_rng();
    let keys =
        Array3::from_shape_fn((num_heads, seq_len, head_dim), |_| rng.gen_range(-1.0..1.0));
    let values =
        Array3::from_shape_fn((num_heads, seq_len, head_dim), |_| rng.gen_range(-1.0..1.0));
    KVCache::new(keys, values, 0)
}

#[test]
fn test_conservative_ratio() {
    let pipeline = CpuCompressionPipeline::new();
    let kv = random_kv(8, 1024, 128);
    let config = PipelineConfig::from_preset(CompressionPreset::Conservative);

    let compressed = pipeline.compress(&kv, &config).unwrap();
    let ratio = compressed.compression_ratio();

    // Conservative: INT8, no projection → ~4x (f32→u8 = 4x, minus param overhead)
    assert!(
        ratio > 3.0,
        "Conservative ratio too low: {:.1}x",
        ratio
    );
    assert!(
        ratio < 6.0,
        "Conservative ratio unexpectedly high: {:.1}x",
        ratio
    );
    eprintln!("Conservative ratio: {:.2}x", ratio);
}

#[test]
fn test_balanced_ratio() {
    let pipeline = CpuCompressionPipeline::new();
    let kv = random_kv(8, 1024, 128);
    let config = PipelineConfig::from_preset(CompressionPreset::Balanced);

    let compressed = pipeline.compress(&kv, &config).unwrap();
    let ratio = compressed.compression_ratio();

    // Balanced: INT4 + projection → should be higher than conservative
    assert!(
        ratio > 4.0,
        "Balanced ratio too low: {:.1}x",
        ratio
    );
    eprintln!("Balanced ratio: {:.2}x", ratio);
}

#[test]
fn test_aggressive_ratio() {
    let pipeline = CpuCompressionPipeline::new();
    let kv = random_kv(8, 1024, 128);
    let config = PipelineConfig::from_preset(CompressionPreset::Aggressive);

    let compressed = pipeline.compress(&kv, &config).unwrap();
    let ratio = compressed.compression_ratio();

    // Aggressive: INT4 + aggressive projection + entropy → highest ratio
    assert!(
        ratio > 4.0,
        "Aggressive ratio too low: {:.1}x",
        ratio
    );
    eprintln!("Aggressive ratio: {:.2}x", ratio);
}

#[test]
fn test_ratio_ordering() {
    let pipeline = CpuCompressionPipeline::new();
    let kv = random_kv(8, 1024, 128);

    let conservative = pipeline
        .compress(&kv, &PipelineConfig::from_preset(CompressionPreset::Conservative))
        .unwrap();
    let balanced = pipeline
        .compress(&kv, &PipelineConfig::from_preset(CompressionPreset::Balanced))
        .unwrap();
    let aggressive = pipeline
        .compress(&kv, &PipelineConfig::from_preset(CompressionPreset::Aggressive))
        .unwrap();

    let r_con = conservative.compression_ratio();
    let r_bal = balanced.compression_ratio();
    let r_agg = aggressive.compression_ratio();

    eprintln!("Ratios: conservative={:.2}x, balanced={:.2}x, aggressive={:.2}x", r_con, r_bal, r_agg);

    // More aggressive presets should compress more
    assert!(r_bal > r_con, "Balanced ({:.1}x) should beat conservative ({:.1}x)", r_bal, r_con);
    assert!(r_agg > r_bal, "Aggressive ({:.1}x) should beat balanced ({:.1}x)", r_agg, r_bal);
}
