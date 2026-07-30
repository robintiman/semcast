//! `relevance(text, 'query')` — the parse-level placeholder for semantic
//! ranking.
//!
//! Same contract as [`means`]: the UDF exists so queries type-check and plan,
//! and must never actually run. [`RelevanceRewriteRule`] lifts it out of the
//! `Projection` or `Sort` that holds it into a `SemRank` extension node, and
//! execution goes through `SemRankExec`. Volatile, so the optimizer never
//! constant-folds it away before the rewrite sees it.
//!
//! Unlike `means()`, this returns a score rather than a verdict — it is the
//! one semcast operator whose output is a value the rest of the query can
//! select, filter, and sort on.
//!
//! [`means`]: crate::sql::means_udf
//! [`RelevanceRewriteRule`]: crate::optimizer::rank::RelevanceRewriteRule

use datafusion::arrow::datatypes::DataType;
use datafusion::common::not_impl_err;
use datafusion::error::Result;
use datafusion::logical_expr::{
    ColumnarValue, ScalarFunctionArgs, ScalarUDF, ScalarUDFImpl, Signature, Volatility,
};

pub const RELEVANCE_UDF_NAME: &str = "relevance";

#[derive(Debug, PartialEq, Eq, Hash)]
struct Relevance {
    signature: Signature,
}

impl ScalarUDFImpl for Relevance {
    fn name(&self) -> &str {
        RELEVANCE_UDF_NAME
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        Ok(DataType::Float64)
    }

    fn invoke_with_args(&self, _args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        not_impl_err!(
            "`relevance` is a marker for the RELEVANCE TO operator and cannot be \
             evaluated directly; the semcast optimizer rule rewrites it into a \
             SemRank node"
        )
    }
}

pub fn relevance_udf() -> ScalarUDF {
    ScalarUDF::from(Relevance {
        signature: Signature::exact(vec![DataType::Utf8, DataType::Utf8], Volatility::Volatile),
    })
}
