//! A proxy learned from the calibration labels, instead of raw similarity.
//!
//! The index pre-filter needs a per-document score to threshold. Cosine
//! similarity to the embedded condition is the obvious one and costs nothing,
//! but it is the *same* geometry for every condition: it can only say "this
//! document is near that phrasing". Whether a document *satisfies* a
//! predicate is a different question, and the calibration sample already
//! answers it for a few dozen documents.
//!
//! So fit a logistic regression on those answers, over vectors the index
//! already stores ([`SemanticIndex::doc_vectors`]), and use its probability as
//! the proxy score. Nothing extra is embedded and no extra model call is made
//! — the labels were bought to calibrate a threshold either way. What changes
//! is that the direction being thresholded is fitted to *this* condition
//! rather than fixed by the embedding of its wording.
//!
//! Training examples are held apart from the ones the threshold is calibrated
//! on, or the floor would be chosen on data the model had already seen and the
//! bound in [`crate::optimizer::calibrate`] would be measuring the fit rather
//! than the population.
//!
//! Falls back to cosine whenever the fit would be meaningless — too few
//! labels, one class only, or vectors the index cannot supply. A worse proxy
//! is a cost problem; the recall target is enforced by the calibration either
//! way.
//!
//! [`SemanticIndex::doc_vectors`]: crate::index::SemanticIndex::doc_vectors

use linfa::prelude::*;
use linfa_logistic::{FittedLogisticRegression, LogisticRegression};
use ndarray::{Array1, Array2};

use crate::model::Embedding;

/// Below this many labelled documents, a fit is noise. The threshold search
/// on raw similarity is the better use of a small sample.
const MIN_TRAINING_EXAMPLES: usize = 12;

/// L2 strength. The library default, kept deliberately: an embedding has far
/// more dimensions than a calibration sample has rows, so shrinkage is what
/// stops the fit memorizing its training half.
const L2_ALPHA: f64 = 1.0;

/// A condition-specific classifier over document vectors.
#[derive(Debug)]
pub struct LearnedProxy {
    model: FittedLogisticRegression<f64, usize>,
    /// Vectors of another width belong to another embedder; scoring them
    /// would be reading noise.
    dimensions: usize,
}

impl LearnedProxy {
    /// Fit on `(vector, matched)` examples, or `None` when the data cannot
    /// support a fit that means anything.
    ///
    /// The one-class case is the one worth naming: if every labelled document
    /// matched (or none did), there is no boundary to learn, and a fit would
    /// either fail or return a constant. Cosine still separates them.
    pub fn fit(examples: &[(&Embedding, bool)]) -> Option<Self> {
        if examples.len() < MIN_TRAINING_EXAMPLES {
            return None;
        }
        let dimensions = examples.first()?.0.len();
        if dimensions == 0 || examples.iter().any(|(v, _)| v.len() != dimensions) {
            return None;
        }
        let positives = examples.iter().filter(|(_, matched)| *matched).count();
        if positives == 0 || positives == examples.len() {
            return None;
        }

        let features: Vec<f64> = examples
            .iter()
            .flat_map(|(vector, _)| vector.iter().map(|x| f64::from(*x)))
            .collect();
        let records = Array2::from_shape_vec((examples.len(), dimensions), features).ok()?;
        let targets: Array1<usize> = examples.iter().map(|(_, m)| usize::from(*m)).collect();

        let model = LogisticRegression::default()
            .alpha(L2_ALPHA)
            .fit(&Dataset::new(records, targets))
            .ok()?;
        Some(Self { model, dimensions })
    }

    /// Probability that `vector`'s document satisfies the condition.
    pub fn score(&self, vector: &Embedding) -> Option<f32> {
        if vector.len() != self.dimensions {
            return None;
        }
        let row = Array2::from_shape_vec(
            (1, self.dimensions),
            vector.iter().map(|x| f64::from(*x)).collect(),
        )
        .ok()?;
        // The positive class is 1, the larger of the two labels fitted.
        Some(self.model.predict_probabilities(&row)[0] as f32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `count` vectors either side of a plane, labelled by which side.
    fn separable(count: usize) -> Vec<(Embedding, bool)> {
        (0..count)
            .map(|i| {
                let matched = i % 2 == 0;
                let noise = i as f32 / (count as f32 * 20.0);
                let vector = if matched {
                    vec![1.0 - noise, noise, 0.0, 0.0]
                } else {
                    vec![noise, 1.0 - noise, 0.0, 0.0]
                };
                (vector, matched)
            })
            .collect()
    }

    fn borrow(examples: &[(Embedding, bool)]) -> Vec<(&Embedding, bool)> {
        examples.iter().map(|(v, m)| (v, *m)).collect()
    }

    #[test]
    fn separates_what_the_labels_separate() {
        let examples = separable(40);
        let proxy = LearnedProxy::fit(&borrow(&examples)).expect("separable data fits");

        let matched = proxy.score(&vec![1.0, 0.0, 0.0, 0.0]).unwrap();
        let unmatched = proxy.score(&vec![0.0, 1.0, 0.0, 0.0]).unwrap();
        assert!(
            matched > 0.5 && unmatched < 0.5,
            "matched {matched}, unmatched {unmatched}",
        );
        assert!(matched > unmatched);
    }

    #[test]
    fn scores_are_probabilities() {
        let examples = separable(40);
        let proxy = LearnedProxy::fit(&borrow(&examples)).unwrap();
        for (vector, _) in &examples {
            let score = proxy.score(vector).unwrap();
            assert!((0.0..=1.0).contains(&score), "score {score}");
        }
    }

    /// The failure DocETL documents: nothing to learn from one class.
    #[test]
    fn one_class_does_not_fit() {
        let all_true: Vec<(Embedding, bool)> = (0..40)
            .map(|i| (vec![i as f32, 1.0, 0.0, 0.0], true))
            .collect();
        assert!(LearnedProxy::fit(&borrow(&all_true)).is_none());

        let all_false: Vec<(Embedding, bool)> = (0..40)
            .map(|i| (vec![i as f32, 1.0, 0.0, 0.0], false))
            .collect();
        assert!(LearnedProxy::fit(&borrow(&all_false)).is_none());
    }

    #[test]
    fn too_few_examples_do_not_fit() {
        let examples = separable(MIN_TRAINING_EXAMPLES - 1);
        assert!(LearnedProxy::fit(&borrow(&examples)).is_none());
    }

    #[test]
    fn an_empty_sample_does_not_fit() {
        assert!(LearnedProxy::fit(&[]).is_none());
    }

    /// Ragged input means two embedders got mixed; refuse rather than guess.
    #[test]
    fn inconsistent_dimensions_do_not_fit() {
        let mut examples = separable(40);
        examples[3].0.push(0.5);
        assert!(LearnedProxy::fit(&borrow(&examples)).is_none());
    }

    #[test]
    fn scoring_a_foreign_vector_is_refused() {
        let examples = separable(40);
        let proxy = LearnedProxy::fit(&borrow(&examples)).unwrap();
        assert!(proxy.score(&vec![1.0, 0.0]).is_none());
    }
}
