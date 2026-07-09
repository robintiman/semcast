//! `CREATE SEMANTIC INDEX / TYPE / PREDICATE` statements.
//!
//! DataFusion hook: a custom statement parser (sqlparser dialect extension)
//! feeding a planner extension. `CREATE SEMANTIC INDEX` and `TYPE` are
//! token-level parsers (no sqlparser `DataType` involvement); `PREDICATE`
//! remains a not-implemented stub.

use std::hash::{DefaultHasher, Hash, Hasher};

use datafusion::error::DataFusionError;
use datafusion::sql::sqlparser::dialect::GenericDialect;
use datafusion::sql::sqlparser::parser::Parser;
use datafusion::sql::sqlparser::tokenizer::Token;

use crate::types::{FieldSpec, FieldType, OrderedF64, SemanticType};

#[derive(Debug, Clone, PartialEq)]
pub enum SemanticDdl {
    /// `CREATE SEMANTIC INDEX ON table(column)`
    CreateIndex { table: String, column: String },
    /// `CREATE SEMANTIC TYPE Name AS (...)`
    CreateType(SemanticType),
    /// `CREATE SEMANTIC PREDICATE name(params) AS template [CHEAP USING ...]`
    CreatePredicate {
        name: String,
        params: Vec<String>,
        /// `MEANS` template with `{param}` placeholders.
        template: String,
        /// Optional `CHEAP USING` override — a user-pinned pre-filter.
        cheap_using: Option<String>,
    },
}

/// Recognize a semantic DDL statement; `Ok(None)` means "not ours, let
/// DataFusion parse it".
pub fn parse_semantic_ddl(sql: &str) -> crate::Result<Option<SemanticDdl>> {
    // Cheap gate so ordinary statements never pay for a second parse: the
    // first two words must be CREATE SEMANTIC.
    let mut words = sql.split_whitespace();
    let prefix_matches = words
        .next()
        .is_some_and(|w| w.eq_ignore_ascii_case("create"))
        && words
            .next()
            .is_some_and(|w| w.eq_ignore_ascii_case("semantic"));
    if !prefix_matches {
        return Ok(None);
    }

    let mut parser = Parser::new(&GenericDialect {})
        .try_with_sql(sql)
        .map_err(|e| DataFusionError::Plan(format!("invalid CREATE SEMANTIC statement: {e}")))?;

    expect_word(&mut parser, "CREATE").map_err(index_error)?;
    expect_word(&mut parser, "SEMANTIC").map_err(index_error)?;
    match parser.next_token().token {
        Token::Word(word) if word.value.eq_ignore_ascii_case("index") => {
            parse_create_index(&mut parser).map(Some)
        }
        Token::Word(word) if word.value.eq_ignore_ascii_case("type") => {
            parse_create_type(&mut parser).map(Some)
        }
        Token::Word(word) if word.value.eq_ignore_ascii_case("predicate") => {
            Err(crate::SemcastError::DataFusion(DataFusionError::Plan(
                "CREATE SEMANTIC PREDICATE is not implemented yet (needs template \
                 substitution and CHEAP USING plumbing)"
                    .to_owned(),
            )))
        }
        other => Err(index_error(format!(
            "expected INDEX, TYPE, or PREDICATE, got {other}"
        ))),
    }
}

/// Parse the tail of `CREATE SEMANTIC INDEX ON table(column)` (the leading
/// `CREATE SEMANTIC INDEX` is already consumed).
fn parse_create_index(parser: &mut Parser) -> crate::Result<SemanticDdl> {
    expect_word(parser, "ON").map_err(index_error)?;
    let table = parser
        .parse_identifier()
        .map_err(|e| index_error(e.to_string()))?;
    parser
        .expect_token(&Token::LParen)
        .map_err(|e| index_error(e.to_string()))?;
    let column = parser
        .parse_identifier()
        .map_err(|e| index_error(e.to_string()))?;
    parser
        .expect_token(&Token::RParen)
        .map_err(|e| index_error(e.to_string()))?;
    let _ = parser.consume_token(&Token::SemiColon);
    parser
        .expect_token(&Token::EOF)
        .map_err(|e| index_error(e.to_string()))?;

    Ok(SemanticDdl::CreateIndex {
        table: table.value,
        column: column.value,
    })
}

/// Parse the tail of `CREATE SEMANTIC TYPE Name AS ( entries )` (the leading
/// `CREATE SEMANTIC TYPE` is already consumed).
///
/// ```text
/// entry     := field_def | TOGETHER ( field_def [, field_def]+ )
/// field_def := <ident> type '<doc string>'
/// type      := base [ '[' ']' ]
/// base      := TEXT | INT | BOOL
///            | REAL [ CHECK ( <num> .. <num> ) ]
///            | ONEOF ( <ident> [, <ident>]* )
///            | LEVEL ( <ident> [, <ident>]* )
/// ```
fn parse_create_type(parser: &mut Parser) -> crate::Result<SemanticDdl> {
    let name = parser
        .parse_identifier()
        .map_err(|e| type_error(e.to_string()))?
        .value;
    expect_word(parser, "AS").map_err(type_error)?;
    parser
        .expect_token(&Token::LParen)
        .map_err(|e| type_error(e.to_string()))?;

    let mut fields: Vec<FieldSpec> = Vec::new();
    let mut together: Vec<Vec<String>> = Vec::new();
    loop {
        if peek_word(parser, "TOGETHER") {
            parser.next_token();
            parser
                .expect_token(&Token::LParen)
                .map_err(|e| type_error(e.to_string()))?;
            let mut group = Vec::new();
            loop {
                let spec = parse_field_def(parser)?;
                group.push(spec.name.clone());
                fields.push(spec);
                match parser.next_token().token {
                    Token::Comma => continue,
                    Token::RParen => break,
                    other => {
                        return Err(type_error(format!(
                            "expected , or ) in TOGETHER, got {other}"
                        )));
                    }
                }
            }
            if group.len() < 2 {
                return Err(type_error(
                    "a TOGETHER group needs at least two fields".to_owned(),
                ));
            }
            together.push(group);
        } else {
            fields.push(parse_field_def(parser)?);
        }
        match parser.next_token().token {
            Token::Comma => continue,
            Token::RParen => break,
            other => {
                return Err(type_error(format!(
                    "expected , or ) between fields, got {other}"
                )));
            }
        }
    }
    let _ = parser.consume_token(&Token::SemiColon);
    parser
        .expect_token(&Token::EOF)
        .map_err(|e| type_error(e.to_string()))?;

    if fields.is_empty() {
        return Err(type_error(
            "a semantic type needs at least one field".to_owned(),
        ));
    }
    // No duplicate field names (checked across groups and singletons).
    for (i, f) in fields.iter().enumerate() {
        if fields[..i].iter().any(|g| g.name == f.name) {
            return Err(type_error(format!("duplicate field name {}", f.name)));
        }
    }

    let version = canonical_version(&fields, &together);
    Ok(SemanticDdl::CreateType(SemanticType {
        name,
        version,
        fields,
        together,
    }))
}

fn parse_field_def(parser: &mut Parser) -> crate::Result<FieldSpec> {
    let name = match parser.next_token().token {
        Token::Word(word) => word.value,
        Token::DoubleQuotedString(s) => s,
        other => return Err(type_error(format!("expected a field name, got {other}"))),
    };
    let ty = parse_field_type(parser)?;
    let doc = match parser.next_token().token {
        Token::SingleQuotedString(s) => s,
        other => {
            return Err(type_error(format!(
                "expected a single-quoted doc string after field {name}, got {other}"
            )));
        }
    };
    Ok(FieldSpec { name, ty, doc })
}

/// Parse a field type from its canonical `FieldType::to_string` rendering —
/// the inline-`EXTRACT` path serializes the type to a string literal at parse
/// time (the dialect hook has no registry), then the optimizer parses it back.
pub(crate) fn parse_field_type_str(spec: &str) -> crate::Result<FieldType> {
    let mut parser = Parser::new(&GenericDialect {})
        .try_with_sql(spec)
        .map_err(|e| type_error(e.to_string()))?;
    let ty = parse_field_type(&mut parser)?;
    parser
        .expect_token(&Token::EOF)
        .map_err(|e| type_error(format!("trailing input after field type `{spec}`: {e}")))?;
    Ok(ty)
}

/// Parse a field type. Shared with the dialect's inline-`EXTRACT` hook, which
/// hands the canonical `FieldType::to_string` rendering back through here.
pub(crate) fn parse_field_type(parser: &mut Parser) -> crate::Result<FieldType> {
    let base = match parser.next_token().token {
        Token::Word(word) => match word.value.to_uppercase().as_str() {
            "TEXT" => FieldType::Text,
            "INT" => FieldType::Int,
            "BOOL" => FieldType::Bool,
            "REAL" => {
                if peek_word(parser, "CHECK") {
                    parser.next_token();
                    let (min, max) = parse_check_range(parser)?;
                    FieldType::RealBounded {
                        min: OrderedF64(min),
                        max: OrderedF64(max),
                    }
                } else {
                    FieldType::Real
                }
            }
            "ONEOF" => FieldType::OneOf(parse_variant_list(parser)?),
            "LEVEL" => FieldType::Level(parse_variant_list(parser)?),
            // A bare unknown type name is a nested semantic type reference.
            other => {
                return Err(type_error(format!(
                    "nested semantic types are not implemented yet (field type {other})"
                )));
            }
        },
        other => return Err(type_error(format!("expected a field type, got {other}"))),
    };
    // Optional `[]` suffix → a list of the base type.
    if matches!(parser.peek_token().token, Token::LBracket) {
        parser.next_token();
        parser
            .expect_token(&Token::RBracket)
            .map_err(|e| type_error(e.to_string()))?;
        Ok(FieldType::List(Box::new(base)))
    } else {
        Ok(base)
    }
}

fn parse_variant_list(parser: &mut Parser) -> crate::Result<Vec<String>> {
    parser
        .expect_token(&Token::LParen)
        .map_err(|e| type_error(e.to_string()))?;
    let mut variants: Vec<String> = Vec::new();
    loop {
        match parser.next_token().token {
            Token::Word(word) => variants.push(word.value),
            other => return Err(type_error(format!("expected a variant name, got {other}"))),
        }
        match parser.next_token().token {
            Token::Comma => continue,
            Token::RParen => break,
            other => {
                return Err(type_error(format!(
                    "expected , or ) in the variant list, got {other}"
                )));
            }
        }
    }
    for (i, v) in variants.iter().enumerate() {
        if variants[..i].iter().any(|w| w == v) {
            return Err(type_error(format!("duplicate variant {v}")));
        }
    }
    Ok(variants)
}

/// Parse `( <num> .. <num> )`. The `..` range operator lexes inconsistently
/// (`0..1` becomes `Number("0.") Period Number("1")`), so collect the raw
/// tokens between the parens, render them back to text, and split on `..`.
fn parse_check_range(parser: &mut Parser) -> crate::Result<(f64, f64)> {
    parser
        .expect_token(&Token::LParen)
        .map_err(|e| type_error(e.to_string()))?;
    let mut rendered = String::new();
    loop {
        match parser.peek_token().token {
            Token::RParen => {
                parser.next_token();
                break;
            }
            Token::EOF => return Err(type_error("unterminated CHECK range".to_owned())),
            other => {
                rendered.push_str(&other.to_string());
                parser.next_token();
            }
        }
    }
    let (lo, hi) = rendered
        .split_once("..")
        .ok_or_else(|| type_error(format!("expected `a..b` in CHECK, got `{rendered}`")))?;
    let lo: f64 = lo
        .trim()
        .parse()
        .map_err(|_| type_error(format!("invalid CHECK lower bound `{lo}`")))?;
    let hi: f64 = hi
        .trim()
        .parse()
        .map_err(|_| type_error(format!("invalid CHECK upper bound `{hi}`")))?;
    Ok((lo, hi))
}

/// The type's `version` — a hash of its canonical rendering. Display/EXPLAIN
/// provenance only; per-field cache keys use per-unit hashes.
fn canonical_version(fields: &[FieldSpec], together: &[Vec<String>]) -> String {
    let mut hasher = DefaultHasher::new();
    for f in fields {
        f.name.hash(&mut hasher);
        f.ty.to_string().hash(&mut hasher);
        f.doc.hash(&mut hasher);
    }
    together.hash(&mut hasher);
    format!("{:x}", hasher.finish())
}

/// `SEMANTIC`, `INDEX`, and `ON` are plain words to the tokenizer (or
/// keywords we don't want keyword semantics for) — match them textually.
fn expect_word(parser: &mut Parser, expected: &str) -> Result<(), String> {
    match parser.next_token().token {
        Token::Word(word) if word.value.eq_ignore_ascii_case(expected) => Ok(()),
        other => Err(format!("expected {expected}, got {other}")),
    }
}

/// Is the next token the keyword `word` (without consuming it)?
fn peek_word(parser: &mut Parser, word: &str) -> bool {
    matches!(parser.peek_token().token, Token::Word(w) if w.value.eq_ignore_ascii_case(word))
}

fn index_error(message: impl Into<String>) -> crate::SemcastError {
    crate::SemcastError::DataFusion(DataFusionError::Plan(format!(
        "invalid CREATE SEMANTIC INDEX statement: {}; \
         expected CREATE SEMANTIC INDEX ON table(column)",
        message.into()
    )))
}

fn type_error(message: impl Into<String>) -> crate::SemcastError {
    crate::SemcastError::DataFusion(DataFusionError::Plan(format!(
        "invalid CREATE SEMANTIC TYPE statement: {}",
        message.into()
    )))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ddl(sql: &str) -> SemanticDdl {
        parse_semantic_ddl(sql).unwrap().expect("recognized as DDL")
    }

    #[test]
    fn parses_create_semantic_index() {
        assert_eq!(
            ddl("CREATE SEMANTIC INDEX ON meetings(transcript)"),
            SemanticDdl::CreateIndex {
                table: "meetings".to_owned(),
                column: "transcript".to_owned(),
            },
        );
    }

    #[test]
    fn is_case_insensitive_and_tolerates_whitespace_and_semicolon() {
        assert_eq!(
            ddl("  create Semantic INDEX on  Meetings ( Transcript ) ;"),
            SemanticDdl::CreateIndex {
                table: "Meetings".to_owned(),
                column: "Transcript".to_owned(),
            },
        );
    }

    #[test]
    fn respects_quoted_identifiers() {
        assert_eq!(
            ddl(r#"CREATE SEMANTIC INDEX ON "my table"("weird column")"#),
            SemanticDdl::CreateIndex {
                table: "my table".to_owned(),
                column: "weird column".to_owned(),
            },
        );
    }

    #[test]
    fn ordinary_statements_are_not_ours() {
        assert_eq!(parse_semantic_ddl("SELECT 1").unwrap(), None);
        assert_eq!(
            parse_semantic_ddl("CREATE TABLE t AS SELECT 1").unwrap(),
            None,
        );
        assert_eq!(parse_semantic_ddl("").unwrap(), None);
        assert_eq!(parse_semantic_ddl("CREATE").unwrap(), None);
    }

    #[test]
    fn malformed_index_statement_is_a_clear_error() {
        let err = parse_semantic_ddl("CREATE SEMANTIC INDEX meetings(transcript)").unwrap_err();
        let message = err.to_string();
        assert!(message.contains("expected ON"), "got: {message}");
        assert!(
            message.contains("CREATE SEMANTIC INDEX ON table(column)"),
            "shows the expected shape: {message}",
        );

        let err = parse_semantic_ddl("CREATE SEMANTIC INDEX ON meetings(transcript) garbage")
            .unwrap_err();
        assert!(err.to_string().contains("EOF"), "got: {err}");
    }

    #[test]
    fn predicate_is_explicit_not_implemented() {
        let err = parse_semantic_ddl("CREATE SEMANTIC PREDICATE p(t) AS t").unwrap_err();
        assert!(err.to_string().contains("not implemented"), "got: {err}");
    }

    fn create_type(sql: &str) -> SemanticType {
        match ddl(sql) {
            SemanticDdl::CreateType(ty) => ty,
            other => panic!("expected CreateType, got {other:?}"),
        }
    }

    #[test]
    fn parses_the_meeting_facts_example() {
        let ty = create_type(
            "CREATE SEMANTIC TYPE MeetingFacts AS (\
               products  TEXT[]   'product names discussed in this meeting',\
               decisions TEXT[]   'concrete decisions that were made',\
               TOGETHER (\
                 launch_stage ONEOF(none, idea, planned, scheduled, shipped) \
                              'the furthest launch stage discussed',\
                 stage_quote  TEXT 'the transcript line that shows that stage'\
               )\
             )",
        );
        assert_eq!(ty.name, "MeetingFacts");
        let names: Vec<&str> = ty.fields.iter().map(|f| f.name.as_str()).collect();
        assert_eq!(
            names,
            vec!["products", "decisions", "launch_stage", "stage_quote"]
        );
        assert_eq!(ty.fields[0].ty, FieldType::List(Box::new(FieldType::Text)),);
        assert_eq!(
            ty.fields[2].ty,
            FieldType::OneOf(vec![
                "none".to_owned(),
                "idea".to_owned(),
                "planned".to_owned(),
                "scheduled".to_owned(),
                "shipped".to_owned(),
            ]),
        );
        assert_eq!(
            ty.together,
            vec![vec!["launch_stage".to_owned(), "stage_quote".to_owned()]],
        );
        assert!(!ty.version.is_empty());
    }

    #[test]
    fn parses_scalar_types_and_check_range() {
        let ty = create_type(
            "CREATE SEMANTIC TYPE T AS (\
               n INT 'a count',\
               ok BOOL 'a flag',\
               score REAL CHECK (0..1) 'a confidence',\
               level LEVEL(low, medium, high) 'a rank'\
             )",
        );
        assert_eq!(ty.fields[0].ty, FieldType::Int);
        assert_eq!(ty.fields[1].ty, FieldType::Bool);
        assert_eq!(
            ty.fields[2].ty,
            FieldType::RealBounded {
                min: OrderedF64(0.0),
                max: OrderedF64(1.0),
            },
        );
        assert_eq!(
            ty.fields[3].ty,
            FieldType::Level(vec![
                "low".to_owned(),
                "medium".to_owned(),
                "high".to_owned()
            ]),
        );
    }

    #[test]
    fn field_type_round_trips_through_display() {
        // parse_field_type must accept FieldType::to_string output.
        for ty in [
            FieldType::Text,
            FieldType::Int,
            FieldType::Bool,
            FieldType::Real,
            FieldType::RealBounded {
                min: OrderedF64(0.0),
                max: OrderedF64(1.0),
            },
            FieldType::OneOf(vec!["a".to_owned(), "b".to_owned()]),
            FieldType::Level(vec!["low".to_owned(), "high".to_owned()]),
            FieldType::List(Box::new(FieldType::Text)),
        ] {
            let rendered = ty.to_string();
            let mut parser = Parser::new(&GenericDialect {})
                .try_with_sql(&rendered)
                .unwrap();
            let parsed = parse_field_type(&mut parser).unwrap();
            assert_eq!(parsed, ty, "round-trip failed for `{rendered}`");
        }
    }

    #[test]
    fn duplicate_field_name_is_an_error() {
        let err =
            parse_semantic_ddl("CREATE SEMANTIC TYPE T AS (x TEXT 'a', x INT 'b')").unwrap_err();
        assert!(
            err.to_string().contains("duplicate field name x"),
            "got: {err}"
        );
    }

    #[test]
    fn together_needs_at_least_two_members() {
        let err =
            parse_semantic_ddl("CREATE SEMANTIC TYPE T AS (TOGETHER (x TEXT 'a'))").unwrap_err();
        assert!(err.to_string().contains("at least two"), "got: {err}");
    }

    #[test]
    fn nested_type_reference_is_deferred() {
        let err = parse_semantic_ddl("CREATE SEMANTIC TYPE T AS (sub OtherType 'a nested thing')")
            .unwrap_err();
        assert!(
            err.to_string().contains("nested semantic types"),
            "got: {err}"
        );
    }

    #[test]
    fn missing_doc_string_is_a_clear_error() {
        let err = parse_semantic_ddl("CREATE SEMANTIC TYPE T AS (x TEXT)").unwrap_err();
        assert!(err.to_string().contains("doc string"), "got: {err}");
    }
}
