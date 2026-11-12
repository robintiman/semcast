//! Calibration samples: which documents a `WITH RECALL` stage labels, and
//! how it labels them. Shared by every pre-filter that calibrates a floor —
//! the index scan and the scorer — so both draw the same documents and pay
//! for each other's labels through the verdict cache.

use datafusion::arrow::array::{Array, StringArray};
use datafusion::physical_plan::metrics::Count;
use std::collections::HashMap;

use crate::cache::{CachedValue, SemanticCache};
use crate::index::doc_hash;
use crate::model::{CompletionRequest, ModelProvider};
use crate::physical::verify::{
    MEANS_CHUNKED_PROMPT_VERSION, MEANS_PROMPT_VERSION, VERIFY_MAX_TOKENS, cache_verdict,
    means_cache_key, merged_calls, parse_verdict, synthesize_means_prompt,
    synthesize_means_prompt_chunked,
};

/// A sampled document and its label: `Some(matched)`, or `None` when the
/// labeling call failed and its truth is unknown.
pub(crate) type Label = (String, Option<bool>);

/// Seeds the calibration sample's document order. Fixed, so repeat queries
/// draw the same sample and their labels come from the verdict cache.
const SAMPLE_SEED: u64 = 0x5eed_ca1b_0000_0001;

/// A document's place in the calibration sample's draw order: SplitMix64 of
/// its [`doc_hash`] mixed with the seed. Taking the documents with the
/// smallest priorities is a uniform sample without replacement, and it
/// depends on nothing but the set of documents — not on input order, not on
/// partitioning.
fn sample_priority(doc_hash: u64) -> u64 {
    let mut z = (doc_hash ^ SAMPLE_SEED).wrapping_add(0x9e37_79b9_7f4a_7c15);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

/// A uniform random sample of the distinct non-NULL documents in `texts`, at
/// most `sample_size`, in draw order: the documents with the smallest
/// [`sample_priority`]. Any prefix is itself a uniform sample, which is what
/// lets certification label it a tranche at a time.
///
/// Only hashes and row positions are kept while drawing; text is copied for
/// the drawn documents alone, so the sample costs memory in its own size,
/// not the input's.
pub(crate) fn draw_sample(texts: &[&StringArray], sample_size: usize) -> Vec<String> {
    let mut first_seen: HashMap<u64, (usize, usize)> = HashMap::new();
    for (batch, array) in texts.iter().enumerate() {
        for row in 0..array.len() {
            if array.is_valid(row) {
                first_seen
                    .entry(doc_hash(array.value(row)))
                    .or_insert((batch, row));
            }
        }
    }
    let mut drawn: Vec<(u64, (usize, usize))> = first_seen
        .into_iter()
        .map(|(hash, position)| (sample_priority(hash), position))
        .collect();
    drawn.sort_unstable_by_key(|(priority, _)| *priority);
    drawn.truncate(sample_size);
    drawn
        .into_iter()
        .map(|(_, (batch, row))| texts[batch].value(row).to_owned())
        .collect()
}

/// Ground-truth labels: the session model reads each document's full text —
/// the definition of `MEANS`.
pub(crate) struct Labeler<'a> {
    /// The funnel's store evidence, told every label as it lands, so later
    /// stages decide a labelled row by its full-text verdict too.
    pub evidence: Option<&'a crate::physical::store_lookup::StoreEvidence>,
    pub condition: &'a str,
    pub model: &'a dyn ModelProvider,
    pub cache: &'a dyn SemanticCache,
    /// Labels sent to the model.
    pub calls: &'a Count,
    /// Of `calls`, those merged onto an identical call in flight.
    pub calls_merged: &'a Count,
    /// Labels whose call failed or answered unparseably.
    pub errors: &'a Count,
    /// Where full-text labels are recorded as the condition's statistics.
    pub stats_key: Option<u128>,
}

impl Labeler<'_> {
    /// Byte-identical requests and cache keys to the no-index verify path,
    /// so labels land in the verdict cache and repeat calibrations draw from
    /// it. A document whose call fails or answers unparseably stays in the
    /// sample, labelled `None`: rows fail, queries don't, but the certificate
    /// counts the failure as a possible miss rather than pretending the
    /// document was never drawn. Failures are not cached, so the next
    /// calibration retries them.
    pub(crate) async fn label(&self, sample: &[String]) -> Vec<Label> {
        let prompt = synthesize_means_prompt(self.condition);
        let inputs = sample.iter().map(|text| (text.clone(), text.clone()));
        let labels = self.ask(inputs, &prompt, MEANS_PROMPT_VERSION).await;
        self.errors
            .add(labels.iter().filter(|(_, label)| label.is_none()).count());
        if let Some(evidence) = self.evidence {
            evidence.record(
                labels
                    .iter()
                    .filter_map(|(text, label)| Some((text.as_str(), (*label)?))),
            );
        }
        labels
    }

    /// What verify will answer for each document when it reads its chunks:
    /// byte-identical requests and keys to [`VerifyExec`]'s chunked path, so
    /// the answers measure what verify does to rows outside the sample, and
    /// a repeat calibration finds them cached. `docs` pairs a document's hash
    /// with its chunks, best first. `None` is a failed call, which verify
    /// would drop, so the certificate counts it as a miss. Not recorded as
    /// store evidence: a chunked verdict is not the full-text answer.
    ///
    /// [`VerifyExec`]: crate::physical::VerifyExec
    pub(crate) async fn chunked(&self, docs: &[(u64, Vec<String>)]) -> Vec<(u64, Option<bool>)> {
        let prompt = synthesize_means_prompt_chunked(self.condition);
        let inputs = docs
            .iter()
            .map(|(hash, chunks)| (*hash, chunks.join(crate::physical::verify::CHUNK_SEPARATOR)));
        self.ask(inputs, &prompt, MEANS_CHUNKED_PROMPT_VERSION)
            .await
    }

    /// Ask `prompt` of each input, from the cache when it has the answer.
    /// Failures stay in the result as `None` and are not cached, so the next
    /// calibration retries them.
    async fn ask<K: Clone>(
        &self,
        inputs: impl Iterator<Item = (K, String)>,
        prompt: &str,
        prompt_version: &str,
    ) -> Vec<(K, Option<bool>)> {
        let model_id = self.model.id();
        let mut answers = Vec::new();
        let mut misses = Vec::new();
        let mut requests = Vec::new();
        let mut input_chars = Vec::new();
        for (key, input) in inputs {
            let cache_key = means_cache_key(self.condition, &input, &model_id, prompt_version);
            match self.cache.get(&cache_key) {
                Some(CachedValue::Value(verdict)) => answers.push((key, Some(verdict == "yes"))),
                _ => {
                    input_chars.push(input.len());
                    requests.push(CompletionRequest {
                        system: prompt.to_owned(),
                        input,
                        max_tokens: VERIFY_MAX_TOKENS,
                        schema: None,
                        logprobs: true,
                    });
                    misses.push((key, cache_key));
                }
            }
        }
        self.calls.add(requests.len());
        let completions = self.model.complete(requests).await;
        self.calls_merged.add(merged_calls(&completions));
        debug_assert_eq!(completions.len(), misses.len());
        // Full-text labels of a uniform sample: the condition's rate as it
        // is. A chunked check is a second, weaker answer about the same
        // document and is not counted.
        if let Some(key) = self.stats_key
            && prompt_version == MEANS_PROMPT_VERSION
        {
            crate::physical::verify::record_means_stats(
                self.cache,
                key,
                input_chars
                    .iter()
                    .zip(&completions)
                    .filter_map(|(chars, completion)| {
                        let matched = parse_verdict(&completion.as_ref().ok()?.text)?;
                        Some((matched, *chars))
                    }),
            );
        }
        for ((key, cache_key), completion) in misses.into_iter().zip(&completions) {
            match completion
                .as_ref()
                .map(|c| (parse_verdict(&c.text), c.yes_probability))
            {
                Ok((Some(matched), yes_probability)) => {
                    cache_verdict(self.cache, cache_key, matched, yes_probability);
                    answers.push((key, Some(matched)));
                }
                Ok((None, _)) | Err(_) => answers.push((key, None)),
            }
        }
        answers
    }
}
