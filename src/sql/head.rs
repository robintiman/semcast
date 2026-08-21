//! Reading a statement's leading words off the token stream.
//!
//! A few places have to recognize a statement before a parser owns it —
//! `SUBMIT` hands its payload through byte-for-byte, and the wire command tag
//! is derived from text. Doing that with `split_whitespace` or `trim_start`
//! looks equivalent to doing it with the tokenizer and isn't: sqlparser
//! represents both `--` and `/* */` comments as [`Token::Whitespace`]
//! variants, so on raw text a leading comment *is* the first word and the
//! statement is misrecognized. That was issue #17.
//!
//! Everything here reads the tokenizer's output instead, which skips comments
//! for free — the same reasoning [`strip_semantic_distinct`] relies on. Where
//! a parser is already in hand, [`peek_nth_is_word`] does the same job as
//! lookahead.
//!
//! One invariant throughout: **input that doesn't tokenize is reported as "not
//! ours"** (`None` / an empty list), never as an error. These are recognition
//! gates, and the real parser has to be the one that produces the error
//! message for a malformed statement.
//!
//! [`strip_semantic_distinct`]: crate::sql::distinct::strip_semantic_distinct

use datafusion::sql::sqlparser::parser::Parser;
use datafusion::sql::sqlparser::tokenizer::{Token, Tokenizer};

use crate::sql::SemcastDialect;

/// The first `n` significant words of `sql`, uppercased.
///
/// Comments and whitespace are skipped. The run stops at the first token that
/// is not an unquoted word, so `SELECT 'CREATE SEMANTIC ...'` yields just
/// `["SELECT"]` and a quoted `"create"` stays an identifier.
pub(crate) fn leading_words(sql: &str, n: usize) -> Vec<String> {
    let Ok(tokens) = Tokenizer::new(&SemcastDialect::default(), sql).tokenize() else {
        return Vec::new();
    };
    tokens
        .iter()
        .filter(|token| !matches!(token, Token::Whitespace(_)))
        .take(n)
        .map_while(|token| match token {
            Token::Word(word) if word.quote_style.is_none() => {
                Some(word.value.to_ascii_uppercase())
            }
            _ => None,
        })
        .collect()
}

/// The text after a leading unquoted `keyword`, as a subslice of `sql`.
///
/// The subslice is what makes this usable for `SUBMIT`, which hands its
/// payload to the engine byte-for-byte: comments *before* the keyword are
/// dropped along with it, and everything from the next token onward survives
/// verbatim.
pub(crate) fn strip_leading_keyword<'a>(sql: &'a str, keyword: &str) -> Option<&'a str> {
    let tokens = Tokenizer::new(&SemcastDialect::default(), sql)
        .tokenize_with_location()
        .ok()?;
    let mut significant = tokens
        .iter()
        .filter(|t| !matches!(t.token, Token::Whitespace(_)));

    match &significant.next()?.token {
        Token::Word(word)
            if word.quote_style.is_none() && word.value.eq_ignore_ascii_case(keyword) => {}
        _ => return None,
    }
    // Resume at the next significant token rather than at the end of the
    // keyword, so a comment sitting between the two goes away with it.
    let rest = match significant.next() {
        Some(next) => {
            let offsets = line_offsets(sql);
            byte_offset(sql, &offsets, next.span.start.line, next.span.start.column)?
        }
        None => sql.len(),
    };
    Some(sql[rest..].trim())
}

/// Is the `n`th lookahead token the unquoted word `word` (keyword status
/// aside)?
///
/// [`Parser::peek_nth_token_ref`] skips whitespace, so this sees through
/// comments the way the rest of this module does.
pub(crate) fn peek_nth_is_word(parser: &Parser, n: usize, word: &str) -> bool {
    matches!(
        &parser.peek_nth_token_ref(n).token,
        Token::Word(w) if w.quote_style.is_none() && w.value.eq_ignore_ascii_case(word)
    )
}

/// Is the next token the unquoted word `word`?
pub(crate) fn peek_is_word(parser: &Parser, word: &str) -> bool {
    peek_nth_is_word(parser, 0, word)
}

/// Byte offset of the start of each 1-based line.
pub(crate) fn line_offsets(sql: &str) -> Vec<usize> {
    let mut offsets = vec![0usize];
    for (i, byte) in sql.bytes().enumerate() {
        if byte == b'\n' {
            offsets.push(i + 1);
        }
    }
    offsets
}

/// sqlparser reports 1-based line and column in *characters*; a caller cutting
/// the text needs bytes. Walking the line rather than adding the column keeps
/// a multibyte literal earlier on the same line from shifting the offset.
pub(crate) fn byte_offset(sql: &str, offsets: &[usize], line: u64, column: u64) -> Option<usize> {
    let line_start = *offsets.get(line.checked_sub(1)? as usize)?;
    let column = column.checked_sub(1)? as usize;
    let rest = sql.get(line_start..)?;
    Some(match rest.char_indices().nth(column) {
        Some((offset, _)) => line_start + offset,
        // The location is the end of the text.
        None => sql.len(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn words(sql: &str) -> Vec<String> {
        leading_words(sql, 3)
    }

    #[test]
    fn reads_the_leading_words() {
        assert_eq!(
            words("CREATE SEMANTIC INDEX ON t(c)"),
            ["CREATE", "SEMANTIC", "INDEX"]
        );
        assert_eq!(words("select 1"), ["SELECT"]);
        assert_eq!(words(""), Vec::<String>::new());
    }

    #[test]
    fn a_leading_comment_is_not_a_word() {
        // The bug this module exists for: on raw text, `--` is word one.
        for sql in [
            "-- why we index\nCREATE SEMANTIC INDEX ON t(c)",
            "/* why we index */ CREATE SEMANTIC INDEX ON t(c)",
            "/* nested /* comment */ */\nCREATE SEMANTIC INDEX ON t(c)",
            "CREATE /* here too */ SEMANTIC INDEX ON t(c)",
        ] {
            assert_eq!(words(sql), ["CREATE", "SEMANTIC", "INDEX"], "for `{sql}`");
        }
    }

    #[test]
    fn the_run_stops_at_the_first_non_word() {
        // Nothing inside a string literal can be read as a leading keyword.
        assert_eq!(words("SELECT 'CREATE SEMANTIC INDEX'"), ["SELECT"]);
        assert_eq!(words("SELECT (1)"), ["SELECT"]);
    }

    #[test]
    fn a_quoted_word_is_an_identifier() {
        assert_eq!(words(r#""CREATE" "SEMANTIC" x"#), Vec::<String>::new());
    }

    #[test]
    fn untokenizable_input_is_not_ours() {
        // An unterminated literal must not become this module's error to
        // report — the real parser owns that message.
        assert_eq!(words("SELECT 'unterminated"), Vec::<String>::new());
        assert_eq!(
            strip_leading_keyword("SUBMIT SELECT 'unterminated", "submit"),
            None,
        );
    }

    #[test]
    fn strips_a_leading_keyword() {
        assert_eq!(
            strip_leading_keyword("SUBMIT SELECT 1", "submit"),
            Some("SELECT 1"),
        );
        assert_eq!(
            strip_leading_keyword("submit\n  SELECT 1", "SUBMIT"),
            Some("SELECT 1"),
        );
        assert_eq!(strip_leading_keyword("SUBMIT", "submit"), Some(""));
    }

    #[test]
    fn strips_a_leading_keyword_past_comments() {
        assert_eq!(
            strip_leading_keyword("-- run it detached\nSUBMIT SELECT 1", "submit"),
            Some("SELECT 1"),
        );
        assert_eq!(
            strip_leading_keyword("SUBMIT /* detached */ SELECT 1", "submit"),
            Some("SELECT 1"),
        );
    }

    #[test]
    fn the_remainder_is_the_original_text() {
        // Byte-for-byte: comments and literals inside the payload survive.
        let sql = "SUBMIT SELECT * FROM t WHERE x MEANS 'a; b' -- keep me";
        assert_eq!(
            strip_leading_keyword(sql, "submit"),
            Some("SELECT * FROM t WHERE x MEANS 'a; b' -- keep me"),
        );
    }

    #[test]
    fn only_a_whole_word_matches() {
        assert_eq!(strip_leading_keyword("SUBMITTED SELECT 1", "submit"), None);
        assert_eq!(
            strip_leading_keyword("SELECT * FROM submissions", "submit"),
            None,
        );
        assert_eq!(strip_leading_keyword(r#""submit" x"#, "submit"), None);
    }

    #[test]
    fn a_multibyte_prefix_does_not_shift_the_cut() {
        assert_eq!(
            strip_leading_keyword("-- ☃ snowman\nSUBMIT SELECT 1", "submit"),
            Some("SELECT 1"),
        );
    }
}
