//! Splits a simple-protocol query string into statements and decides, per
//! statement, whether the engine sees it. Clients open with session chatter
//! (`SET`, `BEGIN`, `SHOW transaction_isolation`) that DataFusion would
//! reject — and the context is shared across connections, so `SET` must not
//! reach it anyway.

/// Where a statement goes. `Canned*` variants answer client handshake
/// probes without touching the engine.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Route {
    /// No-op accepted for client compatibility; respond with this command tag.
    NoOp(&'static str),
    /// One-row `SHOW transaction_isolation` answer.
    CannedTransactionIsolation,
    /// One-row `SHOW server_version` answer.
    CannedServerVersion,
    /// Everything real: `semcast::sql` via the query engine.
    Engine,
}

/// Split on top-level `;`, honoring `'...'` (with `''` escapes), `"..."`
/// identifiers, `$tag$...$tag$` dollar quotes, `--` line comments, and
/// nested `/* */` block comments. Statements keep their original text —
/// re-serializing tokens would risk mangling `MEANS '...'` literals.
pub fn split_statements(input: &str) -> Vec<&str> {
    let bytes = input.as_bytes();
    let mut statements = Vec::new();
    let mut start = 0;
    let mut i = 0;
    while i < bytes.len() {
        match bytes[i] {
            b';' => {
                let stmt = input[start..i].trim();
                if !stmt.is_empty() {
                    statements.push(stmt);
                }
                start = i + 1;
                i += 1;
            }
            b'\'' | b'"' => i = skip_quoted(bytes, i),
            b'$' => i = skip_dollar_quoted(input, i),
            b'-' if bytes.get(i + 1) == Some(&b'-') => {
                while i < bytes.len() && bytes[i] != b'\n' {
                    i += 1;
                }
            }
            b'/' if bytes.get(i + 1) == Some(&b'*') => i = skip_block_comment(bytes, i),
            _ => i += 1,
        }
    }
    let stmt = input[start..].trim();
    if !stmt.is_empty() {
        statements.push(stmt);
    }
    statements
}

/// Byte index just past a `'...'` or `"..."` run starting at `open`.
/// Doubling the delimiter (`''`, `""`) escapes it.
fn skip_quoted(bytes: &[u8], open: usize) -> usize {
    let quote = bytes[open];
    let mut i = open + 1;
    while i < bytes.len() {
        if bytes[i] == quote {
            if bytes.get(i + 1) == Some(&quote) {
                i += 2;
                continue;
            }
            return i + 1;
        }
        i += 1;
    }
    i
}

/// Byte index just past `$tag$...$tag$` starting at `open`, or `open + 1`
/// if this `$` doesn't begin a dollar quote (e.g. `$1` placeholders).
fn skip_dollar_quoted(input: &str, open: usize) -> usize {
    let rest = &input[open + 1..];
    let Some(tag_len) = rest
        .find('$')
        .filter(|&n| rest[..n].chars().all(|c| c.is_alphanumeric() || c == '_'))
    else {
        return open + 1;
    };
    let delim = &input[open..open + tag_len + 2]; // "$tag$"
    match input[open + delim.len()..].find(delim) {
        Some(n) => open + delim.len() + n + delim.len(),
        None => input.len(),
    }
}

/// Byte index just past a `/* ... */` run starting at `open`; Postgres block
/// comments nest.
fn skip_block_comment(bytes: &[u8], open: usize) -> usize {
    let mut depth = 0;
    let mut i = open;
    while i < bytes.len() {
        if bytes[i] == b'/' && bytes.get(i + 1) == Some(&b'*') {
            depth += 1;
            i += 2;
        } else if bytes[i] == b'*' && bytes.get(i + 1) == Some(&b'/') {
            depth -= 1;
            i += 2;
            if depth == 0 {
                return i;
            }
        } else {
            i += 1;
        }
    }
    i
}

pub fn classify(statement: &str) -> Route {
    let lower = statement.to_ascii_lowercase();
    let mut words = lower.split_whitespace();
    match words.next() {
        Some("set") => Route::NoOp("SET"),
        Some("begin") | Some("start") => Route::NoOp("BEGIN"),
        Some("commit") | Some("end") => Route::NoOp("COMMIT"),
        Some("rollback") | Some("abort") => Route::NoOp("ROLLBACK"),
        Some("discard") => Route::NoOp("DISCARD"),
        Some("reset") => Route::NoOp("RESET"),
        Some("show") => match words.collect::<Vec<_>>().join(" ").as_str() {
            "transaction_isolation" | "transaction isolation level" => {
                Route::CannedTransactionIsolation
            }
            "server_version" => Route::CannedServerVersion,
            _ => Route::Engine,
        },
        _ => Route::Engine,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splits_on_top_level_semicolons_only() {
        assert_eq!(
            split_statements("SELECT 1; SELECT 2;\n SELECT 3"),
            vec!["SELECT 1", "SELECT 2", "SELECT 3"],
        );
        assert_eq!(
            split_statements("SELECT * FROM t WHERE x MEANS 'a; b'; SELECT 2"),
            vec!["SELECT * FROM t WHERE x MEANS 'a; b'", "SELECT 2"],
        );
        assert_eq!(
            split_statements(r#"SELECT "odd;name" FROM t"#),
            vec![r#"SELECT "odd;name" FROM t"#],
        );
        assert_eq!(
            split_statements("SELECT 'it''s; fine'; SELECT 2"),
            vec!["SELECT 'it''s; fine'", "SELECT 2"],
        );
        assert_eq!(
            split_statements("SELECT $q$a; b$q$; SELECT 2"),
            vec!["SELECT $q$a; b$q$", "SELECT 2"],
        );
        assert_eq!(
            split_statements("SELECT 1 -- trailing; comment\n; SELECT 2"),
            vec!["SELECT 1 -- trailing; comment", "SELECT 2"],
        );
        assert_eq!(
            split_statements("SELECT 1 /* a; /* nested; */ b */; SELECT 2"),
            vec!["SELECT 1 /* a; /* nested; */ b */", "SELECT 2"],
        );
    }

    #[test]
    fn drops_empty_statements() {
        assert_eq!(split_statements("; ;\n;"), Vec::<&str>::new());
        assert_eq!(split_statements("SELECT 1;;"), vec!["SELECT 1"]);
        assert_eq!(split_statements(""), Vec::<&str>::new());
    }

    #[test]
    fn unterminated_quote_swallows_the_rest() {
        assert_eq!(
            split_statements("SELECT 'unterminated; SELECT 2"),
            vec!["SELECT 'unterminated; SELECT 2"],
        );
    }

    #[test]
    fn session_chatter_becomes_noops() {
        assert_eq!(classify("SET application_name = 'x'"), Route::NoOp("SET"));
        assert_eq!(classify("set datestyle to iso"), Route::NoOp("SET"));
        assert_eq!(classify("BEGIN"), Route::NoOp("BEGIN"));
        assert_eq!(classify("START TRANSACTION"), Route::NoOp("BEGIN"));
        assert_eq!(classify("COMMIT"), Route::NoOp("COMMIT"));
        assert_eq!(classify("ROLLBACK"), Route::NoOp("ROLLBACK"));
        assert_eq!(classify("DISCARD ALL"), Route::NoOp("DISCARD"));
        assert_eq!(classify("RESET all"), Route::NoOp("RESET"));
    }

    #[test]
    fn handshake_shows_are_canned_the_rest_hit_the_engine() {
        assert_eq!(
            classify("SHOW transaction_isolation"),
            Route::CannedTransactionIsolation,
        );
        assert_eq!(
            classify("SHOW TRANSACTION ISOLATION LEVEL"),
            Route::CannedTransactionIsolation,
        );
        assert_eq!(classify("SHOW server_version"), Route::CannedServerVersion);
        assert_eq!(classify("SHOW tables"), Route::Engine);
        assert_eq!(
            classify("SHOW datafusion.execution.batch_size"),
            Route::Engine
        );
    }

    #[test]
    fn real_sql_hits_the_engine() {
        for sql in [
            "SELECT 1",
            "CREATE SEMANTIC INDEX ON meetings(transcript)",
            "SELECT id FROM t WHERE x MEANS 'a' WITH RECALL 0.9",
            "EXPLAIN SELECT 1",
            "CREATE EXTERNAL TABLE t STORED AS PARQUET LOCATION '/x'",
            "INSERT INTO t VALUES (1)",
        ] {
            assert_eq!(classify(sql), Route::Engine, "{sql}");
        }
    }
}
