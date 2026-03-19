IMPORTANT:
use scratch_pad file inside the .project folder named '.project/[agent_name]_scratch_pad.md' where you will write in your mistakes and/or wrong assumptions and how you fix them. If '.project/[agent_name]_scratch_pad.md' doesn't exist create it
In the event of a major feature addition, architecture change, or design decision update, also update the root `proj-lifetime.md` file so the long-term project context stays current.

---

# KVForge Agent Guidelines

## Project Overview

KVForge is a high-performance KV cache compression engine for LLM inference. It implements a 4-stage codec pipeline: sensitivity calibration, low-rank projection (SVD), KIVI-style asymmetric quantization, and rANS entropy coding.

**License:** Apache-2.0
**Language:** Rust (edition 2021, stable toolchain)
**Status:** Phase 1 complete — CPU reference implementations working, CUDA stubs in place, Python bindings scaffolded.

## Workspace Layout

```
kvforge/
├── Cargo.toml                    # Virtual workspace (no [package])
├── crates/
│   ├── kvforge-core/             # Traits + types ONLY (zero implementation)
│   ├── kvforge-cpu/              # CPU reference implementations
│   │   ├── tests/                # Integration tests live HERE (not workspace root)
│   │   ├── benches/              # Criterion benchmarks live HERE
│   │   └── examples/             # Examples live HERE
│   ├── kvforge-cuda/             # CUDA backend (feature-gated)
│   └── kvforge-python/           # PyO3 + maturin bindings
```

**Critical:** This is a virtual workspace (no `[package]` at root). Tests, benches, and examples MUST live inside member crates, not at the workspace root.

## Architecture Rules

### Crate Boundaries

- **kvforge-core** contains ONLY traits, types, enums, and error definitions. No algorithms, no implementations. Any new backend just implements core traits.
- **kvforge-cpu** contains all CPU implementations. Every struct here implements a trait from `kvforge-core`.
- **kvforge-cuda** depends on both `kvforge-core` and `kvforge-cpu`. The `cuda` feature gates all CUDA-specific code. Without the feature, it compiles on any machine and delegates to CPU.
- **kvforge-python** is a cdylib. It wraps CPU pipeline types for Python. Never `cargo build` it directly — use `cargo check` or `maturin develop`.

### Adding a New Backend

1. Create `crates/kvforge-<backend>/`
2. Add it to workspace `members` in root `Cargo.toml`
3. Depend on `kvforge-core` for traits
4. Implement the 4 codec traits + `CompressionPipeline` + optionally `CostModel`
5. Do NOT modify `kvforge-core` to accommodate backend-specific needs — extend via your own types

### Adding a New Codec Stage

1. Define the trait + config struct in `kvforge-core/src/codec.rs`
2. Re-export from `kvforge-core/src/lib.rs`
3. Implement in `kvforge-cpu/src/<stage>.rs`
4. Wire into `CpuCompressionPipeline` in `kvforge-cpu/src/pipeline.rs`
5. Add unit tests in the implementation file, integration tests in `kvforge-cpu/tests/`

## Key Data Types

### Tensor Convention
All 3D tensors follow the shape `[num_heads, seq_len, head_dim]`:
- Axis 0: `num_heads`
- Axis 1: `seq_len` (token dimension)
- Axis 2: `head_dim` (channel dimension)

### KIVI Quantization Axes
- **Keys:** `PerChannel` — quantization params computed along `head_dim` (axis 2), one scale/min per `(head, channel)`
- **Values:** `PerToken` — quantization params computed along `seq_len` (axis 1), one scale/min per `(head, token)`
- Swapping these axes breaks the KIVI scheme. Do not change without understanding the paper.

### INT4 Packing Convention
Two values per byte. Lower nibble = first value, upper nibble = second value.
```
byte = (val_0 & 0x0F) | ((val_1 & 0x0F) << 4)
```
This convention is used in both quantize and dequantize. Changing it breaks serialization compatibility.

### Compression Pipeline Flow
```
Compress:  KVCache -> [calibrate] -> [project] -> quantize -> [entropy encode] -> CompressedKV
Decompress: CompressedKV -> [entropy decode] -> dequantize -> [reconstruct] -> KVCache
```
Stages in brackets are optional depending on `PipelineConfig`.

### Presets
| Preset | Quant | Projection | Entropy | Calibration |
|---|---|---|---|---|
| Conservative | INT8 | No | No | No |
| Balanced | INT4 | Yes (0.90 var) | No | No |
| Aggressive | INT4 | Yes (0.75 var) | Yes | Yes |
| Adaptive | INT4 | Yes (0.90 var) | Yes | Yes |

## Build & Test Commands

```bash
# Build (safe for any machine)
cargo build --workspace --exclude kvforge-cuda --exclude kvforge-python

# Check everything including Python bindings
cargo check --workspace --exclude kvforge-cuda

# Run all tests
cargo test --workspace --exclude kvforge-cuda --exclude kvforge-python

# Run only unit tests (faster, skips integration tests)
cargo test -p kvforge-core -p kvforge-cpu --lib

# Run benchmarks
cargo bench -p kvforge-cpu

# Clippy (treat warnings as errors)
cargo clippy -p kvforge-core -p kvforge-cpu -p kvforge-cuda -- -D warnings

# CUDA crate without CUDA SDK (should always work)
cargo build -p kvforge-cuda

# Python bindings (requires maturin + Python)
cd crates/kvforge-python && maturin develop --release
```

## Testing Expectations

- **39 total tests** (27 unit + 12 integration) must pass before any PR
- Unit tests live inside their implementation files (`#[cfg(test)] mod tests`)
- Integration tests live in `crates/kvforge-cpu/tests/`
- The `test_compression_ratio` and full-pipeline tests use large tensors and are slow in debug mode (~40-90s). This is expected. Use `--release` for faster runs if needed.
- MSE thresholds in tests are calibrated for random uniform data in [-1, 1]. If you change test data distributions, thresholds may need adjustment.

## Dependencies — Known Quirks

| Dependency | Gotcha |
|---|---|
| `constriction` 0.4 | `from_floating_point_probabilities_fast` returns `Result<_, ()>`. Use `{:?}` in format strings, not `{}`. |
| `nalgebra` | Column-major internally. The `ndarray_to_dmatrix` helper in `projection.rs` handles the conversion. |
| `pyo3` / `numpy` | cdylib linking fails with `cargo build`. Always use `cargo check` or `maturin`. |
| Clippy (stable 1.94+) | Requires `is_multiple_of()` instead of `% N != 0`, and `div_ceil()` instead of `(x+1)/2`. |

## Code Style

- Run `cargo fmt` before committing
- Run `cargo clippy -- -D warnings` and fix all warnings
- All public types implement `Debug` and `Clone`
- Traits require `Send + Sync` for thread safety
- Config structs implement `Default` where sensible
- Error handling via `thiserror` — use `KvForgeError` variants, not `.unwrap()` in library code (tests may unwrap)
- Internal math uses `f32`. Only use `f64` for accumulation (variance, SVD thresholds) where precision matters.

## Performance Notes

- SVD (projection) is by far the most expensive stage (~3ms for 4 heads x 256 seq x 64 dim in release)
- Quantization is fast (~75-154 us for same size)
- Entropy coding is fast (~61 us for 10K bytes)
- Conservative pipeline: ~216 us; Balanced pipeline: ~6.5 ms (dominated by SVD)
- CUDA kernels should prioritize fusing projection + quantization to avoid the SVD bottleneck

## CUDA Development Notes

- Feature-gated: `cargo build -p kvforge-cuda` compiles without CUDA SDK (feature `cuda` is off by default)
- `cargo build -p kvforge-cuda --features cuda` requires nvcc and targets sm_80, sm_89, sm_90
- Kernel `.cu` files are in `crates/kvforge-cuda/kernels/`
- `build.rs` uses the `cc` crate with `.cuda(true)` for compilation
- `KernelLauncher` methods are all `todo!()` — implement one at a time with tests
- `GpuCompressionPipeline` currently delegates everything to `CpuCompressionPipeline`
- When implementing kernels, test by comparing GPU output against CPU reference output with tolerance

## Python Bindings Notes

- Module name is `kvforge._kvforge` (native), re-exported from `kvforge/__init__.py`
- `maturin` must be run from `crates/kvforge-python/` or with `--manifest-path`
- `PyKVCache` wraps `kvforge_core::types::KVCache`, converts to/from numpy `Array3<f32>`
- `CompressedKV.to_bytes()` / `CompressedKV.from_bytes()` use `bincode` serialization
- Preset strings: `"conservative"`, `"balanced"`, `"aggressive"`, `"adaptive"` (case-insensitive)

## What NOT to Do

- Do not add a `[package]` to the workspace root `Cargo.toml` — it's a virtual workspace by design
- Do not put tests/benches/examples at the workspace root — they won't be found
- Do not change INT4 packing convention without a migration path — it breaks serialized data
- Do not swap KIVI quantization axes (keys=PerChannel, values=PerToken)
- Do not `cargo build` the Python crate — use `cargo check` or `maturin`
- Do not add `cudarc` as a non-optional dependency — it must stay behind `features = ["cuda"]`
- Do not `.unwrap()` in library code (crates/kvforge-core, kvforge-cpu) — return `KvForgeError`
