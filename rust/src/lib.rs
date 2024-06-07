use numpy::ndarray::{ArrayD,  ArrayViewD};
use numpy::{
    IntoPyArray,  PyArrayDyn,
    PyReadonlyArrayDyn,
};
use pyo3::{
    pymodule,
    types::PyModule,
    Bound, PyResult, Python,
};


#[pymodule]
fn sklearn_rust_engine<'py>(m: &Bound<'py, PyModule>) -> PyResult<()> {
    // example using immutable borrows producing a new array
    fn axpy_(a: f64, x: ArrayViewD<'_, f64>, y: ArrayViewD<'_, f64>) -> ArrayD<f64> {
        a * &x + &y
    }
    // wrapper of `axpy`
    #[pyfn(m)]
    #[pyo3(name = "truc")]
    fn axpy<'py>(
        py: Python<'py>,
        a: f64,
        x: PyReadonlyArrayDyn<'py, f64>,
        y: PyReadonlyArrayDyn<'py, f64>,
    ) -> Bound<'py, PyArrayDyn<f64>> {
        let x = x.as_array();
        let y = y.as_array();
        let z = axpy_(a, x, y);
        z.into_pyarray_bound(py)
    }
    #[pyfn(m)]
    #[pyo3(name = "truc")]
    fn lloyd_step<'py>(
        py: Python<'py>,
        a: f64,
        x: PyReadonlyArrayDyn<'py, f64>,
        y: PyReadonlyArrayDyn<'py, f64>,
    ) -> Bound<'py, PyArrayDyn<f64>> {
        todo()
    }

    /// Formats the sum of two numbers as string.
    #[pyfn(m)]
    #[pyo3()]
    fn my_sum_as_string(a: usize, b: usize) -> PyResult<String> {
         Ok((a + b).to_string())
    }

   Ok(())
}
