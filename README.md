# KVForge

High-performance KV cache compression engine for LLM inference. Implements a multi-stage codec pipeline: sensitivity calibration, low-rank projection, KIVI-style asymmetric quantization, and rANS entropy coding.

## Architecture

```
kvforge-core     Traits + types (zero implementation)
kvforge-cpu      CPU reference implementations
kvforge-cuda     CUDA backend (feature-gated, stubs)
kvforge-python   Python bindings via PyO3 + maturin
```

### Compression Pipeline

1. **Calibration** — Variance-based per-head sensitivity scoring
2. **Projection** — Truncated SVD for dimensionality reduction
3. **Quantization** — KIVI-style: keys per-channel INT4/INT8, values per-token INT4/INT8
4. **Entropy Coding** — rANS lossless compression

### Presets

| Preset | Quantization | Projection | Entropy | Target Ratio |
|---|---|---|---|---|
| Conservative | INT8 | No | No | ~4x |
| Balanced | INT4 | Yes (90% var) | No | ~12x |
| Aggressive | INT4 | Yes (75% var) | Yes | ~20x |

## Build

```bash
# Build (excludes CUDA by default)
cargo build --workspace --exclude kvforge-cuda

# Run tests
cargo test --workspace --exclude kvforge-cuda

# Run benchmarks
cargo bench -p kvforge-cpu

# Run example
cargo run --example basic_compress -p kvforge-cpu
```

## Python

```bash
cd crates/kvforge-python
pip install maturin
maturin develop --release
```

```python
import kvforge
import numpy as np

keys = np.random.randn(32, 2048, 128).astype(np.float32)
values = np.random.randn(32, 2048, 128).astype(np.float32)
kv = kvforge.KVCache(keys, values, layer_idx=0)

compressed = kvforge.compress(kv, preset="balanced")
restored = kvforge.decompress(compressed)
print(f"Ratio: {compressed.ratio():.1f}x")
```

## License

Apache-2.0
