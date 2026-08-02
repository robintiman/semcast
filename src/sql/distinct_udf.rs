//! `semantic_key(text [, similarity])` — the parse-level placeholder for
//! `SEMANTIC DISTINCT ON`.
//!
//! Same contract as the other markers: it exists so queries type-check and
//! plan, and must never actually run. [`DistinctRewriteRule`] lifts it into a
//! `SemDistinct` extension node, and execution goes through
//! `SemDistinctExec`.
//!
//! It returns a *key*, not a label: a short stable string shared by every
//! document in a near-duplicate group. That is what makes it usable as an
//! ordinary `DISTINCT ON` expression once the node has materialized it — the
//! deduplication itself stays DataFusion's. The optional second argument
//! carries a statement-level `WITH SIMILARITY`.
//!
//! [`DistinctRewriteRule`]: crate::optimizer::distinct::DistinctRewriteRule

use datafusion::arrow::datatypes::DataType;
use datafusion::common::not_impl_err;
use datafusion::error::Result;
use datafusion::logical_expr::{
    ColumnarValue, ScalarFunctionArgs, ScalarUDF, ScalarUDFImpl, Signature, TypeSignature,
    Volatility,
};

pub const SEMANTIC_KEY_UDF_NAME: &str = "semantic_key";

#[derive(Debug, PartialEq, Eq, Hash)]
struct SemanticKey {
    signature: Signature,
}

impl ScalarUDFImpl for SemanticKey {
    fn name(&self) -> &str {
        SEMANTIC_KEY_UDF_NAME
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        Ok(DataType::Utf8)
    }

    fn invoke_with_args(&self, _args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        not_impl_err!(
            "`semantic_key` is a marker for SEMANTIC DISTINCT ON and cannot be \
             evaluated directly; the semcast optimizer rule rewrites it into a \
             SemDistinct node"
        )
    }
}

pub fn semantic_key_udf() -> ScalarUDF {
    ScalarUDF::from(SemanticKey {
        signature: Signature::one_of(
            vec![
                TypeSignature::Exact(vec![DataType::Utf8]),
                TypeSignature::Exact(vec![DataType::Utf8, DataType::Float64]),
            ],
            Volatility::Volatile,
        ),
    })
}
