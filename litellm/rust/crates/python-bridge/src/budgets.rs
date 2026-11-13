use std::sync::Arc;

use litellm_callbacks_budgets::{Budgets, Check, Reservation, Shared, decide};
use litellm_callbacks_python::NativeCallback;
use litellm_host_python::{Pythonized, from_py_argument, run_async_value, to_py};
use pyo3::{exceptions::PyRuntimeError, prelude::*};
use serde::de::DeserializeOwned;

use crate::redis::store::spend_counters::NativeSpendCounters;

fn from_model<T: DeserializeOwned>(model: &Bound<'_, PyAny>) -> PyResult<T> {
    from_py_argument(&model.call_method0("model_dump")?)
}

#[pyclass(frozen, name = "NativeBudgets")]
pub(crate) struct NativeBudgets(Arc<Budgets>);

#[pymethods]
impl NativeBudgets {
    #[new]
    fn new(spend_counters: PyRef<'_, NativeSpendCounters>) -> Self {
        Self(Arc::new(Budgets::new(spend_counters.store())))
    }

    fn callback(&self) -> PyResult<NativeCallback> {
        Ok(NativeCallback::new(Arc::new(Shared(Arc::clone(&self.0))))?)
    }

    fn settle<'py>(
        &self,
        py: Python<'py>,
        #[pyo3(from_py_with = from_model)] reservation: Reservation,
        actual: f64,
        finalize: bool,
    ) -> PyResult<Bound<'py, PyAny>> {
        let budgets = Arc::clone(&self.0);
        run_async_value(py, async move {
            budgets
                .settle(reservation, actual, finalize)
                .await
                .map(Pythonized)
                .map_err(|error| PyRuntimeError::new_err(error.to_string()))
        })
    }
}

#[pyfunction]
pub(crate) fn budget_check_decision(
    py: Python<'_>,
    #[pyo3(from_py_with = from_model)] check: Check,
) -> PyResult<Py<PyAny>> {
    to_py(py, &decide(&check))
}
