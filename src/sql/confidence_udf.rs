//! `confidence(text MEANS 'condition')` — the parse-level placeholder for
//! `CONFIDENCE()`: the model's P("yes") for the question the inner `MEANS`
//! asks, as a `DOUBLE` in [0, 1].
//!
//! Like [`means`], it must never run: the classify rewrite replaces it with
//! the column a [`SemClassifyNode`] computes. Its argument is the `MEANS`
//! itself, so the probability is pinned to exactly the question — and the
//! cache entry — a `WHERE ... MEANS` over the same text would use.
//!
//! [`means`]: crate::sql::means_udf
//! [`SemClassifyNode`]: crate::logical::SemClassifyNode

use datafusion::arrow::datatypes::DataType;
use datafusion::common::not_impl_err;
use datafusion::error::Result;
use datafusion::logical_expr::{
    ColumnarValue, ScalarFunctionArgs, ScalarUDF, ScalarUDFImpl, Signature, Volatility,
};

pub const CONFIDENCE_UDF_NAME: &str = "confidence";

#[derive(Debug, PartialEq, Eq, Hash)]
struct Confidence {
    signature: Signature,
}

impl ScalarUDFImpl for Confidence {
    fn name(&self) -> &str {
        CONFIDENCE_UDF_NAME
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        Ok(DataType::Float64)
    }

    fn invoke_with_args(&self, _args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        not_impl_err!(
            "`confidence` is a marker for CONFIDENCE(text MEANS 'condition') and \
             cannot be evaluated directly; run the query through semcast::sql so \
             the semcast optimizer rule can rewrite it"
        )
    }
}

pub fn confidence_udf() -> ScalarUDF {
    ScalarUDF::from(Confidence {
        signature: Signature::exact(vec![DataType::Boolean], Volatility::Volatile),
    })
}
