//! `means(text, 'condition' [, recall])` — the parse-level placeholder for
//! the `MEANS` operator.
//!
//! The UDF exists so queries type-check and plan; it must never actually run.
//! [`MeansRewriteRule`] replaces `Filter(means(..))` with a `SemFilter`
//! extension node, and execution goes through `VerifyExec`. Volatile, so the
//! optimizer never constant-folds it away before the rewrite sees it. The
//! optional third argument is the recall target — how a statement-level
//! `WITH RECALL` rides an expression tree ([`apply_recall`]), and how
//! `ctx.sql()` callers declare one without the semcast dialect.
//!
//! [`MeansRewriteRule`]: crate::optimizer::rewrite::MeansRewriteRule
//! [`apply_recall`]: crate::optimizer::rewrite::apply_recall

use datafusion::arrow::datatypes::DataType;
use datafusion::common::not_impl_err;
use datafusion::error::Result;
use datafusion::logical_expr::{
    ColumnarValue, ScalarFunctionArgs, ScalarUDF, ScalarUDFImpl, Signature, TypeSignature,
    Volatility,
};

pub const MEANS_UDF_NAME: &str = "means";

#[derive(Debug, PartialEq, Eq, Hash)]
struct Means {
    signature: Signature,
}

impl ScalarUDFImpl for Means {
    fn name(&self) -> &str {
        MEANS_UDF_NAME
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        Ok(DataType::Boolean)
    }

    fn invoke_with_args(&self, _args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        not_impl_err!(
            "`means` is a marker for the MEANS operator and cannot be evaluated \
             directly; the semcast optimizer rule rewrites it into a SemFilter \
             node (roadmap step 1)"
        )
    }
}

pub fn means_udf() -> ScalarUDF {
    ScalarUDF::from(Means {
        signature: Signature::one_of(
            vec![
                TypeSignature::Exact(vec![DataType::Utf8, DataType::Utf8]),
                TypeSignature::Exact(vec![DataType::Utf8, DataType::Utf8, DataType::Float64]),
            ],
            Volatility::Volatile,
        ),
    })
}
