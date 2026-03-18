use kvforge_core::codec::{ProjectionConfig, Projector};
use kvforge_core::error::{KvForgeError, Result};
use kvforge_core::types::ProjectionState;
use nalgebra::DMatrix;
use ndarray::Array3;

/// CPU projector using truncated SVD via nalgebra for low-rank projection.
#[derive(Debug, Clone)]
pub struct CpuProjector;

impl CpuProjector {
    pub fn new() -> Self {
        Self
    }
}

impl Default for CpuProjector {
    fn default() -> Self {
        Self::new()
    }
}

/// Convert an ndarray 2D slice to a nalgebra DMatrix.
fn ndarray_to_dmatrix(data: &ndarray::ArrayView2<f32>) -> DMatrix<f32> {
    let (rows, cols) = (data.shape()[0], data.shape()[1]);
    // nalgebra is column-major, so we build from row-major data
    DMatrix::from_fn(rows, cols, |r, c| data[[r, c]])
}

/// Determine the rank to keep based on cumulative variance threshold.
fn select_rank(singular_values: &[f32], threshold: f64, max_rank: Option<usize>) -> usize {
    if singular_values.is_empty() {
        return 0;
    }

    // Squared singular values represent variance
    let total_variance: f64 = singular_values.iter().map(|&s| (s as f64) * (s as f64)).sum();
    if total_variance < 1e-12 {
        return 1;
    }

    let mut cumulative = 0.0;
    let mut rank = 0;
    for &s in singular_values {
        cumulative += (s as f64) * (s as f64);
        rank += 1;
        if cumulative / total_variance >= threshold {
            break;
        }
    }

    // Apply max_rank cap if specified
    if let Some(max_r) = max_rank {
        rank = rank.min(max_r);
    }

    // At least rank 1
    rank.max(1)
}

impl Projector for CpuProjector {
    fn project(
        &self,
        data: &Array3<f32>,
        config: &ProjectionConfig,
    ) -> Result<(Array3<f32>, ProjectionState)> {
        let num_heads = data.shape()[0];
        let seq_len = data.shape()[1];
        let head_dim = data.shape()[2];

        if seq_len == 0 || head_dim == 0 {
            return Err(KvForgeError::ProjectionError(
                "Cannot project empty tensor".into(),
            ));
        }

        // First pass: compute rank for each head via SVD
        let mut ranks = Vec::with_capacity(num_heads);
        let mut svd_results = Vec::with_capacity(num_heads);

        for h in 0..num_heads {
            let head_slice = data.slice(ndarray::s![h, .., ..]);
            let mat = ndarray_to_dmatrix(&head_slice);
            let svd = mat.svd(true, true);

            let sv: Vec<f32> = svd.singular_values.iter().copied().collect();
            let rank = select_rank(&sv, config.variance_threshold, config.max_rank);
            ranks.push(rank);
            svd_results.push(svd);
        }

        let max_rank = *ranks.iter().max().unwrap_or(&1);

        // Second pass: build projected data and basis
        // Projected shape: [num_heads, seq_len, max_rank] — pad shorter ranks with zeros
        let mut projected = Array3::<f32>::zeros((num_heads, seq_len, max_rank));
        let mut basis_vt_all = Vec::new();

        for h in 0..num_heads {
            let rank = ranks[h];
            let svd = &svd_results[h];

            // U: [seq_len, k], S: [k], V^T: [k, head_dim] where k = min(seq_len, head_dim)
            let u = svd.u.as_ref().ok_or_else(|| {
                KvForgeError::ProjectionError("SVD did not compute U".into())
            })?;
            let vt = svd.v_t.as_ref().ok_or_else(|| {
                KvForgeError::ProjectionError("SVD did not compute V^T".into())
            })?;

            // Truncated projection: U_k * S_k => [seq_len, rank]
            for t in 0..seq_len {
                for r in 0..rank {
                    projected[[h, t, r]] = u[(t, r)] * svd.singular_values[r];
                }
            }

            // Store V^T truncated to [rank, head_dim] for this head
            // Pad to [max_rank, head_dim] for uniform storage
            for r in 0..max_rank {
                for c in 0..head_dim {
                    if r < rank {
                        basis_vt_all.push(vt[(r, c)]);
                    } else {
                        basis_vt_all.push(0.0);
                    }
                }
            }
        }

        let state = ProjectionState {
            basis_vt: basis_vt_all,
            ranks,
            num_heads,
            head_dim,
            projected_shape: vec![num_heads, seq_len, max_rank],
        };

        Ok((projected, state))
    }

    fn reconstruct(
        &self,
        projected: &Array3<f32>,
        state: &ProjectionState,
    ) -> Result<Array3<f32>> {
        let num_heads = state.num_heads;
        let head_dim = state.head_dim;
        let seq_len = projected.shape()[1];
        let max_rank = projected.shape()[2];

        let mut result = Array3::<f32>::zeros((num_heads, seq_len, head_dim));

        for h in 0..num_heads {
            let rank = state.ranks[h];
            // basis_vt for head h: starts at h * max_rank * head_dim, shape [max_rank, head_dim]
            let vt_offset = h * max_rank * head_dim;

            // Reconstruct: projected[h] @ V^T[h] => [seq_len, head_dim]
            for t in 0..seq_len {
                for c in 0..head_dim {
                    let mut sum = 0.0f32;
                    for r in 0..rank {
                        sum += projected[[h, t, r]] * state.basis_vt[vt_offset + r * head_dim + c];
                    }
                    result[[h, t, c]] = sum;
                }
            }
        }

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array3;
    use rand::Rng;

    fn random_tensor(shape: (usize, usize, usize)) -> Array3<f32> {
        let mut rng = rand::thread_rng();
        Array3::from_shape_fn(shape, |_| rng.gen_range(-1.0..1.0))
    }

    #[test]
    fn test_project_reconstruct_high_variance() {
        let data = random_tensor((2, 32, 16));
        let projector = CpuProjector::new();
        let config = ProjectionConfig {
            variance_threshold: 0.99,
            max_rank: None,
        };

        let (projected, state) = projector.project(&data, &config).unwrap();
        let restored = projector.reconstruct(&projected, &state).unwrap();

        assert_eq!(restored.shape(), data.shape());

        // With 0.99 variance threshold, MSE should be very low
        let mse = (&data - &restored).mapv(|x| x * x).mean().unwrap();
        assert!(mse < 0.01, "High-variance projection MSE too high: {}", mse);
    }

    #[test]
    fn test_project_reconstruct_low_variance() {
        let data = random_tensor((2, 32, 16));
        let projector = CpuProjector::new();
        let config = ProjectionConfig {
            variance_threshold: 0.5,
            max_rank: None,
        };

        let (projected, state) = projector.project(&data, &config).unwrap();
        let restored = projector.reconstruct(&projected, &state).unwrap();

        assert_eq!(restored.shape(), data.shape());
        // Lower threshold = more lossy, but should still be somewhat close
        let mse = (&data - &restored).mapv(|x| x * x).mean().unwrap();
        assert!(mse < 1.0, "Low-variance projection MSE unreasonably high: {}", mse);
    }

    #[test]
    fn test_rank_reduction() {
        let data = random_tensor((1, 64, 32));
        let projector = CpuProjector::new();
        let config = ProjectionConfig {
            variance_threshold: 0.80,
            max_rank: Some(8),
        };

        let (projected, state) = projector.project(&data, &config).unwrap();
        assert!(
            projected.shape()[2] <= 8,
            "Max rank cap not respected: {}",
            projected.shape()[2]
        );
        assert!(state.ranks[0] <= 8);
    }

    #[test]
    fn test_select_rank_fn() {
        let sv = vec![10.0, 5.0, 2.0, 1.0, 0.5];
        // total variance = 100 + 25 + 4 + 1 + 0.25 = 130.25
        // 0.95 threshold: 100+25 = 125 / 130.25 = 0.9597 >= 0.95 -> rank 2
        assert_eq!(select_rank(&sv, 0.95, None), 2);

        // With max_rank = 2
        assert_eq!(select_rank(&sv, 0.95, Some(2)), 2);

        // Very low threshold
        assert_eq!(select_rank(&sv, 0.5, None), 1);
    }
}
