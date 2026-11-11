//! SemIR — the semantic intermediate representation (redesign phase 2).
//!
//! Every semantic operator is, underneath, one question put to a model about
//! one piece of text: `ASK(input, instruction, type)`. `x MEANS 'p'` is
//! `ASK(x, 'p', BOOL)`; `CONFIDENCE(x MEANS 'p')` is the probability of that
//! same `ASK`; later phases bring `CAST`, classify and rank onto it too.
//!
//! An [`Ask`] has two identities, and keeping them apart is the point:
//!
//! - its **question**, [`Ask::expr_id`]: the normalized instruction, the
//!   output type, the model and the prompt version, *not* the input. Two
//!   queries asking the same question, through any syntax and over any
//!   column, share an `expr_id`.
//! - each **answer**, a [`CacheKey`]: the question plus a digest of the text
//!   it was asked about. That is the row key of the verdict store, so the
//!   store is content-addressed: the same question about the same text is
//!   paid for once, whichever table the text came from.
//!
//! Both are durable — stored answers outlive the process — so, like
//! [`crate::index::doc_hash`], the normalization and the digests must never
//! change; golden tests pin them. The key layout is the one the store has
//! always used, so introducing SemIR invalidates no stored answer.

use datafusion::logical_expr::Expr;

use crate::cache::{CacheKey, stable_digest};
use crate::model::ModelId;

/// What an [`Ask`] answers with. One variant until classify, rank and typed
/// extraction move onto SemIR.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd)]
pub enum AskType {
    /// Yes or no — `MEANS`.
    Bool,
}

impl AskType {
    /// The answer kind as the store names it. Durable: it is part of every
    /// key, and `"means"` is what yes/no verdicts have always been stored as.
    pub fn field(self) -> &'static str {
        match self {
            AskType::Bool => "means",
        }
    }
}

/// The identity of a question, independent of what it is asked about. See
/// the module docs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ExprId(pub u128);

impl std::fmt::Display for ExprId {
    /// The first 8 bytes in hex — enough to tell questions apart in a plan.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:016x}", (self.0 >> 64) as u64)
    }
}

/// One question to a model about one input: `ASK(input, instruction, type)`.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd)]
pub struct Ask {
    /// The text under scrutiny, as an expression over the plan's columns.
    pub input: Expr,
    /// The instruction, normalized ([`normalize_instruction`]).
    pub instruction: String,
    pub output: AskType,
}

impl Ask {
    /// An `ASK`, its instruction normalized on the way in, so every node
    /// built from it agrees on the question.
    pub fn new(input: Expr, instruction: &str, output: AskType) -> Self {
        Self {
            input,
            instruction: normalize_instruction(instruction),
            output,
        }
    }

    /// The question's identity under `model` and `prompt_version`.
    pub fn expr_id(&self, model: &ModelId, prompt_version: &str) -> ExprId {
        question_id(&self.instruction, self.output, model, prompt_version)
    }
}

/// [`Ask::expr_id`] without building an [`Ask`] — for the physical layer,
/// which holds the instruction and type but no longer the input expression.
///
/// A digest of exactly the [`CacheKey`] fields that name the question (every
/// field but the input), so the question id and the store's row keys are two
/// views of one layout.
pub fn question_id(
    instruction: &str,
    output: AskType,
    model: &ModelId,
    prompt_version: &str,
) -> ExprId {
    ExprId(stable_digest(&[
        instruction.as_bytes(),
        output.field().as_bytes(),
        model.0.as_bytes(),
        prompt_version.as_bytes(),
    ]))
}

/// The store key for one answer: the question, plus a digest of the text it
/// was asked about. The same layout the store has always used, so stored
/// answers are found by every build.
pub fn answer_key(
    instruction: &str,
    output: AskType,
    input: &str,
    model: &ModelId,
    prompt_version: &str,
) -> CacheKey {
    CacheKey {
        type_version: instruction.to_owned(),
        field: output.field().to_owned(),
        input_digest: stable_digest(&[input.as_bytes()]),
        model_id: model.clone(),
        prompt_version: prompt_version.to_owned(),
    }
}

/// The instruction as the model is asked it and the store keys it: leading
/// and trailing whitespace removed, and every run of whitespace inside
/// (spaces, tabs, a line break in a long condition) made one space.
///
/// Whitespace only. Case, punctuation and wording can change what a model
/// answers, so normalizing them would merge questions that aren't the same.
/// Durable, like every part of a key: golden tests pin it.
pub fn normalize_instruction(instruction: &str) -> String {
    instruction.split_whitespace().collect::<Vec<_>>().join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn model() -> ModelId {
        ModelId("ollama/gemma4:e4b".to_owned())
    }

    #[test]
    fn whitespace_is_the_only_thing_normalized() {
        assert_eq!(
            normalize_instruction("  the customer\tis\n  angry "),
            "the customer is angry"
        );
        assert_eq!(normalize_instruction("Is Angry?"), "Is Angry?");
        assert_eq!(normalize_instruction(""), "");
    }

    #[test]
    fn syntax_and_input_do_not_change_the_question() {
        let a = Ask::new(datafusion::prelude::col("body"), "is  angry", AskType::Bool);
        let b = Ask::new(datafusion::prelude::col("notes"), "is angry", AskType::Bool);
        assert_eq!(
            a.expr_id(&model(), "means-v2"),
            b.expr_id(&model(), "means-v2")
        );
        assert_ne!(
            a.expr_id(&model(), "means-v2"),
            a.expr_id(&model(), "means-v3"),
            "a new prompt is a new question"
        );
    }

    /// The question id and the answer key must never change: answers are
    /// stored under them and outlive the process.
    #[test]
    fn ids_match_golden_values() {
        let id = question_id("the customer is angry", AskType::Bool, &model(), "means-v2");
        assert_eq!(id, ExprId(GOLDEN_QUESTION));
        assert_eq!(
            id.to_string(),
            format!("{:016x}", (GOLDEN_QUESTION >> 64) as u64)
        );
        let key = answer_key(
            "the customer is angry",
            AskType::Bool,
            "I am furious",
            &model(),
            "means-v2",
        );
        assert_eq!(key.type_version, "the customer is angry");
        assert_eq!(key.field, "means");
        assert_eq!(key.input_digest, stable_digest(&[b"I am furious"]));
        assert_eq!(key.input_digest, GOLDEN_INPUT);
    }

    const GOLDEN_QUESTION: u128 = 25513797799744841332576056727838765421;
    const GOLDEN_INPUT: u128 = 7014386744396058049177024359173512010;
}
