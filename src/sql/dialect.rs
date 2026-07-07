//! [`SemcastDialect`] — sqlparser dialect adding the infix `MEANS` operator.
//!
//! `text MEANS 'condition'` desugars at parse time into a call to the
//! [`means`] marker UDF, so the dialect is pure surface syntax: the rewrite
//! rule, `SemFilter` node, and `VerifyExec` see exactly what they see today.
//!
//! The dialect wraps [`GenericDialect`] (DataFusion's default) and forwards
//! its feature flags, so everything Generic accepts still parses. One
//! deliberate cost: `means` is effectively a reserved word — an unquoted
//! trailing alias like `SELECT x means FROM t` no longer parses. Quote it
//! (`"means"`) if you need it as an identifier.
//!
//! [`means`]: crate::sql::means_udf

use datafusion::sql::sqlparser::ast::{
    Expr, Function, FunctionArg, FunctionArgExpr, FunctionArgumentList, FunctionArguments, Ident,
    ObjectName,
};
use datafusion::sql::sqlparser::dialect::{Dialect, GenericDialect, Precedence};
use datafusion::sql::sqlparser::keywords::Keyword;
use datafusion::sql::sqlparser::parser::{Parser, ParserError};
use datafusion::sql::sqlparser::tokenizer::Token;

use crate::sql::means_udf::MEANS_UDF_NAME;

/// Forward `fn(&self) -> bool` feature flags to the wrapped [`GenericDialect`]
/// so we track its behavior instead of the trait's conservative defaults.
macro_rules! delegate_flags {
    ($($method:ident),* $(,)?) => {
        $(fn $method(&self) -> bool { self.generic.$method() })*
    };
}

#[derive(Debug, Default)]
pub struct SemcastDialect {
    generic: GenericDialect,
}

/// Is the next token the (case-insensitive, unquoted) word `MEANS`?
///
/// `MEANS` is not a sqlparser keyword, so it tokenizes as a plain word with
/// `Keyword::NoKeyword`; a quoted `"means"` keeps its quote style and stays
/// an ordinary identifier.
fn peek_is_means(parser: &Parser) -> bool {
    match &parser.peek_token_ref().token {
        Token::Word(w) => {
            w.keyword == Keyword::NoKeyword
                && w.quote_style.is_none()
                && w.value.eq_ignore_ascii_case("MEANS")
        }
        _ => false,
    }
}

impl Dialect for SemcastDialect {
    fn is_delimited_identifier_start(&self, ch: char) -> bool {
        self.generic.is_delimited_identifier_start(ch)
    }

    fn is_identifier_start(&self, ch: char) -> bool {
        self.generic.is_identifier_start(ch)
    }

    fn is_identifier_part(&self, ch: char) -> bool {
        self.generic.is_identifier_part(ch)
    }

    /// `MEANS` binds like `LIKE`: tighter than `NOT` / `AND` / `OR`, looser
    /// than comparisons. `NOT a MEANS 'x' AND b > 1` groups as
    /// `(NOT (a MEANS 'x')) AND (b > 1)`.
    fn get_next_precedence(&self, parser: &Parser) -> Option<Result<u8, ParserError>> {
        if peek_is_means(parser) {
            return Some(Ok(self.prec_value(Precedence::Like)));
        }
        None
    }

    fn parse_infix(
        &self,
        parser: &mut Parser,
        expr: &Expr,
        _precedence: u8,
    ) -> Option<Result<Expr, ParserError>> {
        if !peek_is_means(parser) {
            return None;
        }
        parser.next_token(); // consume MEANS

        // Parse the condition at MEANS's own precedence, then desugar
        // `lhs MEANS rhs` to `means(lhs, rhs)` — the marker UDF the
        // optimizer rewrite already understands.
        let rhs = match parser.parse_subexpr(self.prec_value(Precedence::Like)) {
            Ok(rhs) => rhs,
            Err(e) => return Some(Err(e)),
        };
        Some(Ok(Expr::Function(Function {
            name: ObjectName::from(vec![Ident::new(MEANS_UDF_NAME)]),
            uses_odbc_syntax: false,
            parameters: FunctionArguments::None,
            args: FunctionArguments::List(FunctionArgumentList {
                duplicate_treatment: None,
                args: vec![
                    FunctionArg::Unnamed(FunctionArgExpr::Expr(expr.clone())),
                    FunctionArg::Unnamed(FunctionArgExpr::Expr(rhs)),
                ],
                clauses: vec![],
            }),
            filter: None,
            null_treatment: None,
            over: None,
            within_group: vec![],
        })))
    }

    delegate_flags!(
        supports_unicode_string_literal,
        supports_partition_by_after_order_by,
        supports_array_join_syntax,
        supports_group_by_expr,
        supports_group_by_with_modifier,
        supports_left_associative_joins_without_parens,
        supports_connect_by,
        supports_match_recognize,
        supports_pipe_operator,
        supports_start_transaction_modifier,
        supports_window_function_null_treatment_arg,
        supports_dictionary_syntax,
        supports_window_clause_named_window_reference,
        supports_parenthesized_set_variables,
        supports_select_wildcard_except,
        support_map_literal_syntax,
        allow_extract_custom,
        allow_extract_single_quotes,
        supports_extract_comma_syntax,
        supports_create_view_comment_syntax,
        supports_parens_around_table_factor,
        supports_values_as_table_factor,
        supports_create_index_with_clause,
        supports_explain_with_utility_options,
        supports_limit_comma,
        supports_update_order_by,
        supports_from_first_select,
        supports_projection_trailing_commas,
        supports_asc_desc_in_column_definition,
        supports_try_convert,
        supports_bitwise_shift_operators,
        supports_comment_on,
        supports_load_extension,
        supports_named_fn_args_with_assignment_operator,
        supports_struct_literal,
        supports_empty_projections,
        supports_nested_comments,
        supports_multiline_comment_hints,
        supports_user_host_grantee,
        supports_string_escape_constant,
        supports_array_typedef_with_brackets,
        supports_match_against,
        supports_set_names,
        supports_comma_separated_set_assignments,
        supports_filter_during_aggregation,
        supports_select_wildcard_exclude,
        supports_data_type_signed_suffix,
        supports_interval_options,
        supports_quote_delimited_string,
        supports_select_wildcard_replace,
        supports_select_wildcard_ilike,
        supports_select_wildcard_rename,
        supports_optimize_table,
        supports_install,
        supports_detach,
        supports_prewhere,
        supports_with_fill,
        supports_limit_by,
        supports_interpolate,
        supports_settings,
        supports_select_format,
        supports_comment_optimizer_hint,
        supports_constraint_keyword_without_name,
        supports_key_column_option,
        supports_comma_separated_trim,
        supports_cte_without_as,
        supports_select_item_multi_column_alias,
        supports_xml_expressions,
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(sql: &str) -> String {
        let stmts = Parser::parse_sql(&SemcastDialect::default(), sql).unwrap();
        assert_eq!(stmts.len(), 1);
        stmts[0].to_string()
    }

    #[test]
    fn means_desugars_to_udf_call() {
        assert_eq!(
            parse("SELECT * FROM meetings WHERE notes MEANS 'action items were assigned'"),
            "SELECT * FROM meetings WHERE means(notes, 'action items were assigned')"
        );
    }

    #[test]
    fn means_is_case_insensitive() {
        assert_eq!(
            parse("SELECT * FROM t WHERE body means 'urgent'"),
            "SELECT * FROM t WHERE means(body, 'urgent')"
        );
    }

    #[test]
    fn means_binds_tighter_than_not_and_or() {
        assert_eq!(
            parse("SELECT * FROM t WHERE NOT a MEANS 'x' AND b MEANS 'y' OR c = 1"),
            "SELECT * FROM t WHERE NOT means(a, 'x') AND means(b, 'y') OR c = 1"
        );
    }

    #[test]
    fn means_condition_can_be_any_expression() {
        assert_eq!(
            parse("SELECT * FROM t WHERE body MEANS 'about ' || topic"),
            "SELECT * FROM t WHERE means(body, 'about ' || topic)"
        );
    }

    #[test]
    fn quoted_means_stays_an_identifier() {
        assert_eq!(
            parse(r#"SELECT x "means" FROM t"#),
            r#"SELECT x AS "means" FROM t"#
        );
    }

    #[test]
    fn plain_sql_still_parses() {
        assert_eq!(
            parse("SELECT a, count(*) FROM t WHERE a LIKE 'x%' GROUP BY 1 + 1"),
            "SELECT a, count(*) FROM t WHERE a LIKE 'x%' GROUP BY 1 + 1"
        );
    }

    #[tokio::test]
    async fn means_executes_end_to_end() {
        use std::sync::Arc;

        use datafusion::arrow::array::AsArray;

        use crate::model::MockModel;
        use crate::semcast_context;

        let ctx = semcast_context(Arc::new(MockModel::answering_yes_to(["offline sync"])));
        ctx.sql(
            "CREATE TABLE notes AS SELECT * FROM (VALUES
                 (1, 'we shipped offline sync'),
                 (2, 'nothing notable happened')
             ) AS t(id, body)",
        )
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();

        let batches = crate::sql(
            &ctx,
            "SELECT id FROM notes WHERE body MEANS 'syncs offline'",
        )
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();

        let ids: Vec<i64> = batches
            .iter()
            .flat_map(|b| {
                b.column(0)
                    .as_primitive::<datafusion::arrow::datatypes::Int64Type>()
                    .values()
                    .to_vec()
            })
            .collect();
        assert_eq!(ids, vec![1]);
    }
}
