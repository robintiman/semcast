//! Marker UDFs for typed extraction — the parse-level placeholders that
//! `CAST(x AS T).f`, `CAST(x AS T)`, and inline `EXTRACT(f TY 'd' FROM x)`
//! desugar to.
//!
//! Like [`means`], these never run: [`ExtractRewriteRule`] replaces them with
//! a `SemExtract` extension node before execution. They exist so a typed
//! query plans — in particular so projection schemas type-check — which is why
//! each reports an accurate output type from `return_field_from_args`,
//! reading the literal type/field arguments and consulting the registry.
//! Volatile, so the optimizer never constant-folds them before the rewrite.
//!
//! [`means`]: crate::sql::means_udf
//! [`ExtractRewriteRule`]: crate::optimizer::extract::ExtractRewriteRule

use std::sync::Arc;

use datafusion::arrow::datatypes::{DataType, Field, FieldRef, Fields};
use datafusion::common::{ScalarValue, not_impl_err, plan_err};
use datafusion::error::Result;
use datafusion::logical_expr::{
    ColumnarValue, ReturnFieldArgs, ScalarFunctionArgs, ScalarUDF, ScalarUDFImpl, Signature,
    Volatility,
};

use crate::sql::ddl::parse_field_type_str;
use crate::types::registry::TypeRegistry;

pub const SEM_EXTRACT_UDF_NAME: &str = "sem_extract";
pub const SEM_EXTRACT_FIELD_UDF_NAME: &str = "sem_extract_field";
pub const SEM_EXTRACT_INLINE_UDF_NAME: &str = "sem_extract_inline";

/// The `i`th argument as a string literal, if it is one. Marker literals
/// (type name, field name, typespec, doc) are always constant string args.
fn literal_str<'a>(args: &ReturnFieldArgs<'a>, idx: usize) -> Option<&'a str> {
    match args.scalar_arguments.get(idx).copied().flatten() {
        Some(ScalarValue::Utf8(Some(s)))
        | Some(ScalarValue::LargeUtf8(Some(s)))
        | Some(ScalarValue::Utf8View(Some(s))) => Some(s.as_str()),
        _ => None,
    }
}

/// `equals`/`hash_value` (via `DynEq`/`DynHash`) key on the signature and the
/// registry's identity — two UDFs backed by the same registry are the same.
macro_rules! registry_udf_identity {
    ($t:ty) => {
        impl PartialEq for $t {
            fn eq(&self, other: &Self) -> bool {
                self.signature == other.signature && Arc::ptr_eq(&self.registry, &other.registry)
            }
        }
        impl Eq for $t {}
        impl std::hash::Hash for $t {
            fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
                self.signature.hash(state);
                Arc::as_ptr(&self.registry).hash(state);
            }
        }
    };
}

/// `sem_extract(source, 'Type')` → a Struct of all the type's fields.
#[derive(Debug)]
struct SemExtract {
    signature: Signature,
    registry: Arc<TypeRegistry>,
}

registry_udf_identity!(SemExtract);

impl ScalarUDFImpl for SemExtract {
    fn name(&self) -> &str {
        SEM_EXTRACT_UDF_NAME
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        // Struct shape needs the type-name literal, which only
        // `return_field_from_args` sees.
        plan_err!("{SEM_EXTRACT_UDF_NAME} requires the semantic type name as a literal")
    }

    fn return_field_from_args(&self, args: ReturnFieldArgs) -> Result<FieldRef> {
        let type_name = literal_str(&args, 1).ok_or_else(|| {
            datafusion::error::DataFusionError::Plan(format!(
                "{SEM_EXTRACT_UDF_NAME}: the semantic type name must be a string literal"
            ))
        })?;
        let ty = self
            .registry
            .get(type_name)
            .ok_or_else(|| unknown_type(type_name))?;
        let mut fields = Vec::with_capacity(ty.fields.len());
        for spec in &ty.fields {
            fields.push(Arc::new(Field::new(
                &spec.name,
                spec.ty.arrow_type()?,
                true,
            )));
        }
        Ok(Arc::new(Field::new(
            &ty.name,
            DataType::Struct(Fields::from(fields)),
            true,
        )))
    }

    fn invoke_with_args(&self, _args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        not_impl_err!(
            "{SEM_EXTRACT_UDF_NAME} is a marker for typed extraction and cannot be \
             evaluated directly; the semcast optimizer rewrites it into a SemExtract node"
        )
    }
}

/// `sem_extract_field(source, 'Type', 'field')` → that field's Arrow type.
#[derive(Debug)]
struct SemExtractField {
    signature: Signature,
    registry: Arc<TypeRegistry>,
}

registry_udf_identity!(SemExtractField);

impl ScalarUDFImpl for SemExtractField {
    fn name(&self) -> &str {
        SEM_EXTRACT_FIELD_UDF_NAME
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        plan_err!("{SEM_EXTRACT_FIELD_UDF_NAME} requires the type and field names as literals")
    }

    fn return_field_from_args(&self, args: ReturnFieldArgs) -> Result<FieldRef> {
        let type_name = literal_str(&args, 1).ok_or_else(|| {
            datafusion::error::DataFusionError::Plan(format!(
                "{SEM_EXTRACT_FIELD_UDF_NAME}: the semantic type name must be a string literal"
            ))
        })?;
        let field = literal_str(&args, 2).ok_or_else(|| {
            datafusion::error::DataFusionError::Plan(format!(
                "{SEM_EXTRACT_FIELD_UDF_NAME}: the field name must be a string literal"
            ))
        })?;
        let ty = self
            .registry
            .get(type_name)
            .ok_or_else(|| unknown_type(type_name))?;
        Ok(Arc::new(Field::new(field, ty.arrow_type(field)?, true)))
    }

    fn invoke_with_args(&self, _args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        not_impl_err!(
            "{SEM_EXTRACT_FIELD_UDF_NAME} is a marker for typed extraction and cannot be \
             evaluated directly; the semcast optimizer rewrites it into a SemExtract node"
        )
    }
}

/// `sem_extract_inline(source, 'field', 'typespec', 'doc')` → the type
/// encoded by `typespec` (a canonical `FieldType` rendering).
#[derive(Debug)]
struct SemExtractInline {
    signature: Signature,
    registry: Arc<TypeRegistry>,
}

registry_udf_identity!(SemExtractInline);

impl ScalarUDFImpl for SemExtractInline {
    fn name(&self) -> &str {
        SEM_EXTRACT_INLINE_UDF_NAME
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        plan_err!("{SEM_EXTRACT_INLINE_UDF_NAME} requires the field type spec as a literal")
    }

    fn return_field_from_args(&self, args: ReturnFieldArgs) -> Result<FieldRef> {
        let field = literal_str(&args, 1).ok_or_else(|| {
            datafusion::error::DataFusionError::Plan(format!(
                "{SEM_EXTRACT_INLINE_UDF_NAME}: the field name must be a string literal"
            ))
        })?;
        let typespec = literal_str(&args, 2).ok_or_else(|| {
            datafusion::error::DataFusionError::Plan(format!(
                "{SEM_EXTRACT_INLINE_UDF_NAME}: the field type must be a string literal"
            ))
        })?;
        let ty = parse_field_type_str(typespec)?;
        Ok(Arc::new(Field::new(field, ty.arrow_type()?, true)))
    }

    fn invoke_with_args(&self, _args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        not_impl_err!(
            "{SEM_EXTRACT_INLINE_UDF_NAME} is a marker for typed extraction and cannot be \
             evaluated directly; the semcast optimizer rewrites it into a SemExtract node"
        )
    }
}

fn unknown_type(name: &str) -> datafusion::error::DataFusionError {
    datafusion::error::DataFusionError::Plan(format!(
        "unknown semantic type {name}; define it with CREATE SEMANTIC TYPE first"
    ))
}

/// The three marker UDFs, each holding the shared type registry. Registered in
/// the context builder next to `means_udf`.
pub fn extract_udfs(registry: Arc<TypeRegistry>) -> [ScalarUDF; 3] {
    let utf8 = DataType::Utf8;
    [
        ScalarUDF::from(SemExtract {
            signature: Signature::exact(vec![utf8.clone(), utf8.clone()], Volatility::Volatile),
            registry: Arc::clone(&registry),
        }),
        ScalarUDF::from(SemExtractField {
            signature: Signature::exact(
                vec![utf8.clone(), utf8.clone(), utf8.clone()],
                Volatility::Volatile,
            ),
            registry: Arc::clone(&registry),
        }),
        ScalarUDF::from(SemExtractInline {
            signature: Signature::exact(
                vec![utf8.clone(), utf8.clone(), utf8.clone(), utf8],
                Volatility::Volatile,
            ),
            registry,
        }),
    ]
}
