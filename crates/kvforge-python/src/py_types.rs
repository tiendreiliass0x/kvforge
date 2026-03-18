use ndarray::Array3;
use numpy::{PyArray3, PyArrayMethods, IntoPyArray};
use pyo3::prelude::*;

/// Python wrapper for KVCache.
#[pyclass]
#[derive(Clone)]
pub struct KVCache {
    pub inner: kvforge_core::types::KVCache,
}

#[pymethods]
impl KVCache {
    /// Create a new KVCache from numpy arrays.
    ///
    /// Args:
    ///     keys: numpy array of shape [num_heads, seq_len, head_dim], dtype float32
    ///     values: numpy array of shape [num_heads, seq_len, head_dim], dtype float32
    ///     layer_idx: layer index
    #[new]
    fn new(
        keys: &Bound<'_, PyArray3<f32>>,
        values: &Bound<'_, PyArray3<f32>>,
        layer_idx: usize,
    ) -> PyResult<Self> {
        let keys_array: Array3<f32> = keys.to_owned_array();
        let values_array: Array3<f32> = values.to_owned_array();

        if keys_array.shape() != values_array.shape() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "keys and values must have the same shape",
            ));
        }

        Ok(Self {
            inner: kvforge_core::types::KVCache::new(keys_array, values_array, layer_idx),
        })
    }

    /// Get the keys as a numpy array.
    fn keys<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray3<f32>> {
        self.inner.keys.clone().into_pyarray(py)
    }

    /// Get the values as a numpy array.
    fn values<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray3<f32>> {
        self.inner.values.clone().into_pyarray(py)
    }

    #[getter]
    fn layer_idx(&self) -> usize {
        self.inner.layer_idx
    }

    #[getter]
    fn shape(&self) -> (usize, usize, usize) {
        let s = self.inner.shape();
        (s.num_heads, s.seq_len, s.head_dim)
    }

    fn size_bytes(&self) -> usize {
        self.inner.size_bytes()
    }

    fn __repr__(&self) -> String {
        let s = self.inner.shape();
        format!(
            "KVCache(num_heads={}, seq_len={}, head_dim={}, layer_idx={})",
            s.num_heads, s.seq_len, s.head_dim, self.inner.layer_idx
        )
    }
}

/// Python wrapper for CompressedKV.
#[pyclass]
#[derive(Clone)]
pub struct CompressedKV {
    pub inner: kvforge_core::types::CompressedKV,
}

#[pymethods]
impl CompressedKV {
    /// Compression ratio (original / compressed).
    fn ratio(&self) -> f64 {
        self.inner.compression_ratio()
    }

    /// Compressed size in bytes.
    fn compressed_size_bytes(&self) -> usize {
        self.inner.compressed_size_bytes()
    }

    /// Original size in bytes.
    fn original_size_bytes(&self) -> usize {
        self.inner.original_size_bytes
    }

    /// Serialize to bytes.
    fn to_bytes(&self) -> PyResult<Vec<u8>> {
        bincode::serialize(&self.inner)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("Serialization failed: {}", e)))
    }

    /// Deserialize from bytes.
    #[staticmethod]
    fn from_bytes(data: &[u8]) -> PyResult<Self> {
        let inner: kvforge_core::types::CompressedKV = bincode::deserialize(data)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("Deserialization failed: {}", e)))?;
        Ok(Self { inner })
    }

    fn __repr__(&self) -> String {
        format!(
            "CompressedKV(ratio={:.1}x, compressed_bytes={}, original_bytes={})",
            self.inner.compression_ratio(),
            self.inner.compressed_size_bytes(),
            self.inner.original_size_bytes,
        )
    }
}
