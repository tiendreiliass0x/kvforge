use criterion::{black_box, criterion_group, criterion_main, Criterion};
use kvforge_core::codec::*;
use kvforge_core::pipeline::{CompressionPipeline, CompressionPreset, PipelineConfig};
use kvforge_core::types::KVCache;
use kvforge_cpu::*;
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

fn bench_quantize(c: &mut Criterion) {
    let quantizer = CpuQuantizer::new();
    let mut rng = rand::thread_rng();
    let data = Array3::from_shape_fn((4, 256, 64), |_| rng.gen_range(-1.0f32..1.0));

    c.bench_function("quantize_int4_per_channel", |b| {
        b.iter(|| quantizer.quantize(black_box(&data), &QuantizeConfig::keys_int4()))
    });

    c.bench_function("quantize_int8_per_token", |b| {
        b.iter(|| quantizer.quantize(black_box(&data), &QuantizeConfig::values_int8()))
    });
}

fn bench_projection(c: &mut Criterion) {
    let projector = CpuProjector::new();
    let mut rng = rand::thread_rng();
    let data = Array3::from_shape_fn((4, 256, 64), |_| rng.gen_range(-1.0f32..1.0));
    let config = ProjectionConfig {
        variance_threshold: 0.90,
        max_rank: None,
    };

    c.bench_function("project_svd", |b| {
        b.iter(|| projector.project(black_box(&data), &config))
    });
}

fn bench_entropy(c: &mut Criterion) {
    let codec = CpuEntropyCodec::new();
    let data: Vec<u8> = (0..10000).map(|i| (i % 256) as u8).collect();
    let config = EntropyConfig { enabled: true };

    c.bench_function("entropy_encode", |b| {
        b.iter(|| codec.encode(black_box(&data), &config))
    });
}

fn bench_pipeline(c: &mut Criterion) {
    let pipeline = CpuCompressionPipeline::new();
    let kv = random_kv(4, 256, 64);

    let conservative_config = PipelineConfig::from_preset(CompressionPreset::Conservative);
    c.bench_function("pipeline_conservative", |b| {
        b.iter(|| pipeline.compress(black_box(&kv), &conservative_config))
    });

    let balanced_config = PipelineConfig::from_preset(CompressionPreset::Balanced);
    c.bench_function("pipeline_balanced", |b| {
        b.iter(|| pipeline.compress(black_box(&kv), &balanced_config))
    });
}

criterion_group!(benches, bench_quantize, bench_projection, bench_entropy, bench_pipeline);
criterion_main!(benches);
