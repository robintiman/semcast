//! `means(text, 'condition')` — the parse-level placeholder for the `MEANS`
//! operator.
//!
//! The UDF exists so queries type-check and plan; it must never actually run.
//! [`MeansRewriteRule`] replaces `Filter(means(..))` with a `SemFilter`
//! extension node, and execution goes through `VerifyExec`. Volatile, so the
//! optimizer never constant-folds it away before the rewrite sees it.
//!
//! [`MeansRewriteRule`]: crate::optimizer::rewrite::MeansRewriteRule

use std::sync::Arc;

use datafusion::arrow::datatypes::DataType;
use datafusion::common::not_impl_err;
use datafusion::logical_expr::{ColumnarValue, ScalarUDF, Volatility, create_udf};

pub const MEANS_UDF_NAME: &str = "means";

pub fn means_udf() -> ScalarUDF {
    create_udf(
        MEANS_UDF_NAME,
        vec![DataType::Utf8, DataType::Utf8],
        DataType::Boolean,
        Volatility::Volatile,
        Arc::new(|_args: &[ColumnarValue]| {
            not_impl_err!(
                "`means` is a marker for the MEANS operator and cannot be evaluated \
                 directly; the semcast optimizer rule rewrites it into a SemFilter \
                 node (roadmap step 1)"
            )
        }),
    )
}
