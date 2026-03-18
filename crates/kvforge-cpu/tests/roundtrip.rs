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

fn compute_mse(a: &Array3<f32>, b: &Array3<f32>) -> f32 {
    (a - b).mapv(|x| x * x).mean().unwrap()
}

#[test]
fn test_roundtrip_conservative() {
    let pipeline = CpuCompressionPipeline::new();
    let kv = random_kv(4, 128, 32);
    let config = PipelineConfig::from_preset(CompressionPreset::Conservative);

    let compressed = pipeline.compress(&kv, &config).unwrap();
    let restored = pipeline.decompress(&compressed).unwrap();

    assert_eq!(restored.keys.shape(), kv.keys.shape());
    assert_eq!(restored.values.shape(), kv.values.shape());
    assert_eq!(restored.layer_idx, kv.layer_idx);

    let key_mse = compute_mse(&kv.keys, &restored.keys);
    let val_mse = compute_mse(&kv.values, &restored.values);

    // Conservative (INT8, no projection): very low MSE
    assert!(key_mse < 0.0005, "Conservative key MSE: {}", key_mse);
    assert!(val_mse < 0.0005, "Conservative val MSE: {}", val_mse);
}

#[test]
fn test_roundtrip_balanced() {
    let pipeline = CpuCompressionPipeline::new();
    let kv = random_kv(4, 128, 32);
    let config = PipelineConfig::from_preset(CompressionPreset::Balanced);

    let compressed = pipeline.compress(&kv, &config).unwrap();
    let restored = pipeline.decompress(&compressed).unwrap();

    assert_eq!(restored.keys.shape(), kv.keys.shape());
    assert_eq!(restored.values.shape(), kv.values.shape());

    let key_mse = compute_mse(&kv.keys, &restored.keys);
    let val_mse = compute_mse(&kv.values, &restored.values);

    // Balanced (INT4 + projection): moderate MSE
    assert!(key_mse < 0.05, "Balanced key MSE: {}", key_mse);
    assert!(val_mse < 0.05, "Balanced val MSE: {}", val_mse);
}

#[test]
fn test_roundtrip_aggressive() {
    let pipeline = CpuCompressionPipeline::new();
    let kv = random_kv(4, 128, 32);
    let config = PipelineConfig::from_preset(CompressionPreset::Aggressive);

    let compressed = pipeline.compress(&kv, &config).unwrap();
    let restored = pipeline.decompress(&compressed).unwrap();

    assert_eq!(restored.keys.shape(), kv.keys.shape());
    assert_eq!(restored.values.shape(), kv.values.shape());

    let key_mse = compute_mse(&kv.keys, &restored.keys);
    let val_mse = compute_mse(&kv.values, &restored.values);

    // Aggressive (INT4 + aggressive projection + entropy): higher MSE but bounded
    assert!(key_mse < 0.5, "Aggressive key MSE: {}", key_mse);
    assert!(val_mse < 0.5, "Aggressive val MSE: {}", val_mse);
}
