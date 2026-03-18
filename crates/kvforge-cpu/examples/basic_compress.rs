use kvforge_core::pipeline::{CompressionPipeline, CompressionPreset, PipelineConfig};
use kvforge_core::types::KVCache;
use kvforge_cpu::default_pipeline;
use ndarray::Array3;
use rand::Rng;

fn main() {
    let mut rng = rand::thread_rng();
    let num_heads = 32;
    let seq_len = 2048;
    let head_dim = 128;

    println!(
        "Generating random KV cache: {} heads, {} tokens, {} dim",
        num_heads, seq_len, head_dim
    );

    let keys =
        Array3::from_shape_fn((num_heads, seq_len, head_dim), |_| rng.gen_range(-1.0..1.0));
    let values =
        Array3::from_shape_fn((num_heads, seq_len, head_dim), |_| rng.gen_range(-1.0..1.0));
    let kv = KVCache::new(keys, values, 0);

    let pipeline = default_pipeline();

    for preset in &[
        CompressionPreset::Conservative,
        CompressionPreset::Balanced,
        CompressionPreset::Aggressive,
    ] {
        let config = PipelineConfig::from_preset(*preset);
        let compressed = pipeline.compress(&kv, &config).unwrap();
        let restored = pipeline.decompress(&compressed).unwrap();

        let key_mse = (&kv.keys - &restored.keys)
            .mapv(|x| x * x)
            .mean()
            .unwrap();
        let val_mse = (&kv.values - &restored.values)
            .mapv(|x| x * x)
            .mean()
            .unwrap();

        println!(
            "{:?}: ratio={:.1}x, key_mse={:.6}, val_mse={:.6}, compressed={} bytes",
            preset,
            compressed.compression_ratio(),
            key_mse,
            val_mse,
            compressed.compressed_size_bytes(),
        );
    }
}
