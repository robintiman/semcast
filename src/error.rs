//! Crate-wide error type.

use datafusion::error::DataFusionError;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum SemcastError {
    #[error("datafusion error: {0}")]
    DataFusion(#[from] DataFusionError),

    #[error("model error: {0}")]
    Model(String),

    #[error("calibration error: {0}")]
    Calibration(String),

    #[error("index error: {0}")]
    Index(String),
}

pub type Result<T, E = SemcastError> = std::result::Result<T, E>;

impl From<SemcastError> for DataFusionError {
    fn from(err: SemcastError) -> Self {
        match err {
            SemcastError::DataFusion(inner) => inner,
            other => DataFusionError::External(Box::new(other)),
        }
    }
}
