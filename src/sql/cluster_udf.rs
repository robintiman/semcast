//! `meaning_of(text [, k])` — the parse-level placeholder for `GROUP BY
//! MEANING OF`.
//!
//! Same contract as [`means`] and [`relevance`]: the UDF exists so queries
//! type-check and plan, and must never actually run. [`ClusterRewriteRule`]
//! lifts it out of the aggregate or projection that holds it into a
//! `SemCluster` extension node, and execution goes through `SemClusterExec`.
//!
//! It returns a label — a string naming the group a row landed in — which is
//! why it can serve as an ordinary `GROUP BY` key once the node has
//! materialized it. The two-argument form carries the explicit `INTO k`.
//!
//! [`means`]: crate::sql::means_udf
//! [`relevance`]: crate::sql::rank_udf
//! [`ClusterRewriteRule`]: crate::optimizer::cluster::ClusterRewriteRule

use datafusion::arrow::datatypes::DataType;
use datafusion::common::not_impl_err;
use datafusion::error::Result;
use datafusion::logical_expr::{
    ColumnarValue, ScalarFunctionArgs, ScalarUDF, ScalarUDFImpl, Signature, TypeSignature,
    Volatility,
};

pub const MEANING_OF_UDF_NAME: &str = "meaning_of";

#[derive(Debug, PartialEq, Eq, Hash)]
struct MeaningOf {
    signature: Signature,
}

impl ScalarUDFImpl for MeaningOf {
    fn name(&self) -> &str {
        MEANING_OF_UDF_NAME
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        Ok(DataType::Utf8)
    }

    fn invoke_with_args(&self, _args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        not_impl_err!(
            "`meaning_of` is a marker for the MEANING OF operator and cannot be \
             evaluated directly; the semcast optimizer rule rewrites it into a \
             SemCluster node"
        )
    }
}

pub fn meaning_of_udf() -> ScalarUDF {
    ScalarUDF::from(MeaningOf {
        signature: Signature::one_of(
            vec![
                TypeSignature::Exact(vec![DataType::Utf8]),
                TypeSignature::Exact(vec![DataType::Utf8, DataType::Int64]),
            ],
            Volatility::Volatile,
        ),
    })
}
