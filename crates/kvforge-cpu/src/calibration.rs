use kvforge_core::codec::{CalibrationConfig, SensitivityCalibrator, SensitivityScores};
use kvforge_core::error::{KvForgeError, Result};
use kvforge_core::types::KVCache;

/// CPU sensitivity calibrator using variance-based heuristic.
///
/// Per-head: computes variance of key vectors across tokens.
/// Higher variance = more sensitive to compression = higher score.
/// Scores are normalized to [0, 1].
#[derive(Debug, Clone)]
pub struct CpuSensitivityCalibrator;

impl CpuSensitivityCalibrator {
    pub fn new() -> Self {
        Self
    }
}

impl Default for CpuSensitivityCalibrator {
    fn default() -> Self {
        Self::new()
    }
}

impl SensitivityCalibrator for CpuSensitivityCalibrator {
    fn calibrate(&self, kv: &KVCache, config: &CalibrationConfig) -> Result<SensitivityScores> {
        let shape = kv.shape();

        if shape.seq_len < config.min_tokens {
            return Err(KvForgeError::InvalidConfig(format!(
                "Need at least {} tokens for calibration, got {}",
                config.min_tokens, shape.seq_len
            )));
        }

        let num_heads = shape.num_heads;
        let seq_len = shape.seq_len;
        let head_dim = shape.head_dim;
        let n = (seq_len * head_dim) as f64;

        let mut variances = Vec::with_capacity(num_heads);

        for h in 0..num_heads {
            // Compute variance of all key values for this head
            let mut sum = 0.0f64;
            let mut sum_sq = 0.0f64;

            for t in 0..seq_len {
                for c in 0..head_dim {
                    let v = kv.keys[[h, t, c]] as f64;
                    sum += v;
                    sum_sq += v * v;
                }
            }

            let mean = sum / n;
            let variance = (sum_sq / n) - (mean * mean);
            variances.push(variance.max(0.0));
        }

        // Normalize to [0, 1]
        let max_var = variances.iter().cloned().fold(0.0f64, f64::max);
        let scores = if max_var < 1e-12 {
            vec![0.5f32; num_heads] // uniform if all zero
        } else {
            variances
                .iter()
                .map(|&v| (v / max_var) as f32)
                .collect()
        };

        Ok(SensitivityScores { scores })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array3;
    use rand::Rng;

    #[test]
    fn test_calibrate_basic() {
        let mut rng = rand::thread_rng();
        // Head 0: low variance, Head 1: high variance
        let mut keys = Array3::<f32>::zeros((2, 128, 16));
        let values = Array3::<f32>::zeros((2, 128, 16));

        for t in 0..128 {
            for c in 0..16 {
                keys[[0, t, c]] = rng.gen_range(-0.1..0.1); // low variance
                keys[[1, t, c]] = rng.gen_range(-10.0..10.0); // high variance
            }
        }

        let kv = KVCache::new(keys, values, 0);
        let calibrator = CpuSensitivityCalibrator::new();
        let scores = calibrator
            .calibrate(&kv, &CalibrationConfig::default())
            .unwrap();

        assert_eq!(scores.scores.len(), 2);
        // Head 1 should have higher sensitivity score
        assert!(
            scores.scores[1] > scores.scores[0],
            "High-variance head should be more sensitive: {} vs {}",
            scores.scores[1],
            scores.scores[0]
        );
        // Scores should be in [0, 1]
        for &s in &scores.scores {
            assert!(s >= 0.0 && s <= 1.0, "Score out of range: {}", s);
        }
    }

    #[test]
    fn test_calibrate_too_few_tokens() {
        let keys = Array3::<f32>::zeros((2, 32, 16));
        let values = Array3::<f32>::zeros((2, 32, 16));
        let kv = KVCache::new(keys, values, 0);
        let calibrator = CpuSensitivityCalibrator::new();

        let result = calibrator.calibrate(&kv, &CalibrationConfig { min_tokens: 64 });
        assert!(result.is_err());
    }

    #[test]
    fn test_calibrate_uniform() {
        // All heads have same variance — should all get similar scores
        let mut rng = rand::thread_rng();
        let keys = Array3::from_shape_fn((4, 128, 16), |_| rng.gen_range(-1.0..1.0));
        let values = Array3::from_shape_fn((4, 128, 16), |_| rng.gen_range(-1.0..1.0));
        let kv = KVCache::new(keys, values, 0);

        let calibrator = CpuSensitivityCalibrator::new();
        let scores = calibrator
            .calibrate(&kv, &CalibrationConfig::default())
            .unwrap();

        // All scores should be roughly similar (within 0.3 of each other)
        let max = scores.scores.iter().cloned().fold(0.0f32, f32::max);
        let min = scores.scores.iter().cloned().fold(1.0f32, f32::min);
        assert!(
            max - min < 0.3,
            "Uniform heads should have similar scores: range {} - {} = {}",
            max,
            min,
            max - min
        );
    }
}
