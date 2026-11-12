//! List prices, for estimating what a refresh will cost before it runs.
//!
//! Estimates only: per-million-token list prices for the models semcast is
//! commonly pointed at, as published when this table was written. Batch
//! discounts, prompt caching and partner pricing (Bedrock, Vertex) are not
//! modelled, and a model not listed has no price — an estimate says
//! "unknown" rather than guess.

use super::ModelId;

/// When the prices below were taken from Anthropic's published price list.
pub const PRICES_AS_OF: &str = "2026-11";

/// USD per million input and output tokens for `model`, if known. A local
/// Ollama model costs nothing; a fallback chain is priced by its primary,
/// which answers everything it can.
pub fn price_per_mtok(model: &ModelId) -> Option<(f64, f64)> {
    let primary = model.0.split('+').next().unwrap_or_default();
    if primary.starts_with("ollama/") {
        return Some((0.0, 0.0));
    }
    let name = primary.strip_prefix("anthropic/")?;
    let price = match name {
        "claude-fable-5-2" | "claude-mythos-5-2" => (8.0, 40.0),
        "claude-fable-5-1" | "claude-mythos-5-1" | "claude-fable-5" => (10.0, 50.0),
        "claude-opus-5-5" => (4.0, 20.0),
        "claude-opus-5" | "claude-opus-4-8" | "claude-opus-4-7" | "claude-opus-4-6" => (5.0, 25.0),
        "claude-sonnet-5-5" | "claude-sonnet-5" => (2.0, 10.0),
        "claude-sonnet-4-6" => (3.0, 15.0),
        "claude-haiku-5-5" => (0.10, 0.50),
        "claude-haiku-4-5" => (1.0, 5.0),
        _ => return None,
    };
    Some(price)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn id(s: &str) -> ModelId {
        ModelId(s.to_owned())
    }

    #[test]
    fn known_unknown_and_local() {
        assert_eq!(
            price_per_mtok(&id("anthropic/claude-haiku-4-5")),
            Some((1.0, 5.0))
        );
        assert_eq!(price_per_mtok(&id("ollama/gemma4:e4b")), Some((0.0, 0.0)));
        assert_eq!(price_per_mtok(&id("anthropic/claude-unknown")), None);
        assert_eq!(price_per_mtok(&id("mock")), None);
        assert_eq!(
            price_per_mtok(&id("anthropic/claude-haiku-4-5+ollama/gemma4:e4b")),
            Some((1.0, 5.0)),
            "a chain is priced by its primary"
        );
    }
}
