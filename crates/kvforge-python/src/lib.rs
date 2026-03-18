mod py_pipeline;
mod py_types;

use pyo3::prelude::*;

#[pymodule]
fn _kvforge(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<py_types::KVCache>()?;
    m.add_class::<py_types::CompressedKV>()?;
    m.add_class::<py_pipeline::CompressionPipeline>()?;
    m.add_function(wrap_pyfunction!(py_pipeline::compress, m)?)?;
    m.add_function(wrap_pyfunction!(py_pipeline::decompress, m)?)?;
    Ok(())
}
