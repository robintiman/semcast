//! AST rewrite for `CAST(x AS SemanticType)` and `CAST(x AS T).field`.
//!
//! sqlparser parses an unknown type name in a cast as `DataType::Custom`, and
//! `CAST(x AS T).f` as a `CompoundFieldAccess` over that cast. DataFusion's
//! planner can't handle either (a dot on a non-string, non-identifier root is
//! `not_impl_err`), so we desugar both to marker UDF calls *before*
//! `statement_to_plan` — the same trick the infix `MEANS` dialect uses, one
//! layer up. Only casts to a *registered* type are touched; an unknown name is
//! left for DataFusion's normal type resolution to reject.
//!
//! [`visit_expressions_mut`] visits children before parents, so the inner
//! `Cast` becomes `sem_extract(x, 'T')` first and the enclosing field access
//! then matches on that rewritten root — no ordering plumbing needed.

use std::ops::ControlFlow;

use datafusion::error::DataFusionError;
use datafusion::sql::parser::Statement;
use datafusion::sql::sqlparser::ast::{
    AccessExpr, DataType, Expr, Function, FunctionArg, FunctionArgExpr, FunctionArgumentList,
    FunctionArguments, Ident, ObjectName, Value, visit_expressions_mut,
};

use crate::sql::extract_udf::{SEM_EXTRACT_FIELD_UDF_NAME, SEM_EXTRACT_UDF_NAME};
use crate::types::registry::TypeRegistry;

/// Desugar semantic casts in `statement` in place. Only the plain-SQL
/// statement variant carries a sqlparser AST; DataFusion's own extensions
/// (`CREATE EXTERNAL TABLE`, `COPY TO`, ...) have no cast expressions and are
/// skipped.
pub fn rewrite_semantic_casts(
    statement: &mut Statement,
    registry: &TypeRegistry,
) -> crate::Result<()> {
    let Statement::Statement(inner) = statement else {
        return Ok(());
    };
    match visit_expressions_mut(inner.as_mut(), |expr| rewrite_expr(expr, registry)) {
        ControlFlow::Continue(()) => Ok(()),
        ControlFlow::Break(err) => Err(err),
    }
}

fn rewrite_expr(expr: &mut Expr, registry: &TypeRegistry) -> ControlFlow<crate::SemcastError> {
    // Bare `CAST(x AS T)` where T is a registered semantic type → the
    // whole-struct marker. Unregistered custom types are left untouched.
    let cast_rewrite = match expr {
        Expr::Cast {
            data_type: DataType::Custom(name, _),
            expr: source,
            ..
        } => {
            let type_name = name.to_string();
            registry.get(&type_name).map(|_| {
                call(
                    SEM_EXTRACT_UDF_NAME,
                    vec![(**source).clone(), str_lit(&type_name)],
                )
            })
        }
        _ => None,
    };
    if let Some(rewritten) = cast_rewrite {
        *expr = rewritten;
        return ControlFlow::Continue(());
    }

    // `CAST(x AS T).field` — after the inner cast became `sem_extract(x, 'T')`,
    // rewrite the field access to the per-field marker.
    if let Expr::CompoundFieldAccess { root, access_chain } = expr
        && let Some((source, type_name)) = as_sem_extract_call(root)
    {
        match access_chain.as_slice() {
            [AccessExpr::Dot(Expr::Identifier(field))] => {
                let rewritten = call(
                    SEM_EXTRACT_FIELD_UDF_NAME,
                    vec![source, str_lit(&type_name), str_lit(&field.value)],
                );
                *expr = rewritten;
            }
            // `CAST(x AS T).field[i]` / `.a.b` — deeper access chains are a
            // deferred slice; fail clearly rather than plan a broken tree.
            [AccessExpr::Dot(Expr::Identifier(_)), ..] => {
                return ControlFlow::Break(crate::SemcastError::DataFusion(DataFusionError::Plan(
                    "only a single field access is supported on \
                             CAST(... AS SemanticType); wrap it in a subquery for more"
                        .to_owned(),
                )));
            }
            _ => {}
        }
    }
    ControlFlow::Continue(())
}

/// If `expr` is a `sem_extract(source, 'Type')` call this rewrite produced,
/// pull the source expression and the type-name literal back out.
fn as_sem_extract_call(expr: &Expr) -> Option<(Expr, String)> {
    let Expr::Function(func) = expr else {
        return None;
    };
    if func.name.to_string() != SEM_EXTRACT_UDF_NAME {
        return None;
    }
    let FunctionArguments::List(list) = &func.args else {
        return None;
    };
    let [
        FunctionArg::Unnamed(FunctionArgExpr::Expr(source)),
        FunctionArg::Unnamed(FunctionArgExpr::Expr(type_arg)),
    ] = list.args.as_slice()
    else {
        return None;
    };
    let type_name = string_literal(type_arg)?;
    Some((source.clone(), type_name))
}

fn string_literal(expr: &Expr) -> Option<String> {
    match expr {
        Expr::Value(v) => match &v.value {
            Value::SingleQuotedString(s) => Some(s.clone()),
            _ => None,
        },
        _ => None,
    }
}

fn str_lit(s: &str) -> Expr {
    Expr::Value(Value::SingleQuotedString(s.to_owned()).with_empty_span())
}

fn call(name: &str, args: Vec<Expr>) -> Expr {
    Expr::Function(Function {
        name: ObjectName::from(vec![Ident::new(name)]),
        uses_odbc_syntax: false,
        parameters: FunctionArguments::None,
        args: FunctionArguments::List(FunctionArgumentList {
            duplicate_treatment: None,
            args: args
                .into_iter()
                .map(|e| FunctionArg::Unnamed(FunctionArgExpr::Expr(e)))
                .collect(),
            clauses: vec![],
        }),
        filter: None,
        null_treatment: None,
        over: None,
        within_group: vec![],
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sql::recall::parse_statement_with_recall;
    use crate::types::SemanticType;

    fn registry_with(name: &str) -> TypeRegistry {
        let registry = TypeRegistry::default();
        registry
            .register(SemanticType {
                name: name.to_owned(),
                version: "v1".to_owned(),
                fields: vec![],
                together: vec![],
            })
            .unwrap();
        registry
    }

    fn rewritten(sql: &str, registry: &TypeRegistry) -> String {
        let (mut statement, _) = parse_statement_with_recall(sql).unwrap();
        rewrite_semantic_casts(&mut statement, registry).unwrap();
        statement.to_string()
    }

    #[test]
    fn field_access_desugars_to_sem_extract_field() {
        let registry = registry_with("MeetingFacts");
        let out = rewritten(
            "SELECT CAST(transcript AS MeetingFacts).launch_stage FROM meetings",
            &registry,
        );
        assert!(
            out.contains("sem_extract_field(transcript, 'MeetingFacts', 'launch_stage')"),
            "got: {out}",
        );
    }

    #[test]
    fn bare_cast_desugars_to_sem_extract() {
        let registry = registry_with("MeetingFacts");
        let out = rewritten(
            "SELECT CAST(transcript AS MeetingFacts) AS facts FROM meetings",
            &registry,
        );
        assert!(
            out.contains("sem_extract(transcript, 'MeetingFacts')"),
            "got: {out}",
        );
    }

    #[test]
    fn unregistered_type_is_left_untouched() {
        let registry = TypeRegistry::default();
        let out = rewritten(
            "SELECT CAST(transcript AS Unknown).x FROM meetings",
            &registry,
        );
        assert!(!out.contains("sem_extract"), "got: {out}");
    }

    #[test]
    fn deeper_access_chain_is_a_clear_error() {
        let registry = registry_with("MeetingFacts");
        let (mut statement, _) = parse_statement_with_recall(
            "SELECT CAST(transcript AS MeetingFacts).products[1] FROM meetings",
        )
        .unwrap();
        let err = rewrite_semantic_casts(&mut statement, &registry).unwrap_err();
        assert!(
            err.to_string().contains("single field access"),
            "got: {err}"
        );
    }
}
