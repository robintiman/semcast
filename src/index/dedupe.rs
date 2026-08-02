//! Greedy near-duplicate grouping over document vectors.
//!
//! Clustering asks "what are the themes here" and answers with `k` groups.
//! Dedupe asks a narrower question — "which of these are *the same thing*" —
//! and the answer is not a count but a threshold: two documents are the same
//! when they are at least this similar.
//!
//! So this is not k-means. It walks the documents in order and gives each one
//! either an existing representative it is close enough to, or its own. That
//! makes the first document of each group the one that defines it, which is
//! what "keep the first, drop the rest" needs to mean something.

/// Similarity at or above which two documents count as duplicates, when the
/// query didn't say. Deliberately strict: wrongly merging two distinct rows
/// loses data, while wrongly keeping two similar ones only leaves a duplicate
/// the user can see.
pub const DEFAULT_SIMILARITY: f32 = 0.9;

/// For each point, the index of the point that represents its group.
///
/// A representative points at itself, so `groups[i] == i` identifies the rows
/// that survive a dedupe. Order matters and is the caller's: the first point
/// of a group is always its representative.
pub fn greedy_groups(points: &[Vec<f32>], threshold: f32) -> Vec<usize> {
    let mut groups = Vec::with_capacity(points.len());
    let mut representatives: Vec<usize> = Vec::new();

    for (i, point) in points.iter().enumerate() {
        // Nearest representative wins, not merely the first one over the
        // line — otherwise the grouping would depend on representative
        // discovery order rather than on the geometry.
        let best = representatives
            .iter()
            .map(|&rep| (rep, similarity(point, &points[rep])))
            .filter(|(_, score)| *score >= threshold)
            .max_by(|(_, a), (_, b)| a.total_cmp(b));
        match best {
            Some((rep, _)) => groups.push(rep),
            None => {
                representatives.push(i);
                groups.push(i);
            }
        }
    }
    groups
}

/// Cosine similarity. The vectors come from
/// [`SemanticIndex::doc_vectors`](crate::index::SemanticIndex::doc_vectors),
/// which normalizes, so this is a dot product — but it renormalizes anyway
/// rather than trusting a caller it cannot see.
fn similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(a, b)| a * b).sum();
    let norm =
        a.iter().map(|v| v * v).sum::<f32>().sqrt() * b.iter().map(|v| v * v).sum::<f32>().sqrt();
    if norm > 0.0 { dot / norm } else { 0.0 }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_points_share_a_representative() {
        let points = vec![vec![1.0, 0.0], vec![1.0, 0.0], vec![0.0, 1.0]];
        assert_eq!(greedy_groups(&points, 0.9), vec![0, 0, 2]);
    }

    #[test]
    fn the_first_point_of_a_group_represents_it() {
        let points = vec![vec![1.0, 0.0], vec![1.0, 0.0], vec![1.0, 0.0]];
        let groups = greedy_groups(&points, 0.9);
        assert_eq!(groups, vec![0, 0, 0], "the first row is the survivor");
    }

    #[test]
    fn a_strict_threshold_keeps_near_neighbours_apart() {
        // ~0.98 similar: duplicates at 0.9, distinct at 0.99.
        let points = vec![vec![1.0, 0.0], vec![0.98, 0.2]];
        assert_eq!(greedy_groups(&points, 0.9), vec![0, 0]);
        assert_eq!(greedy_groups(&points, 0.99), vec![0, 1]);
    }

    #[test]
    fn a_point_joins_its_nearest_representative() {
        // Two representatives, both over the line for the third point; it
        // must take the closer one rather than whichever was found first.
        let points = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![0.9, 0.44]];
        let groups = greedy_groups(&points, 0.5);
        assert_eq!(groups[2], 0, "closer to the first representative");
    }

    #[test]
    fn a_threshold_of_one_only_merges_exact_matches() {
        let points = vec![vec![1.0, 0.0], vec![0.999, 0.0447], vec![1.0, 0.0]];
        let groups = greedy_groups(&points, 1.0);
        assert_eq!(groups[0], 0);
        assert_eq!(groups[2], 0, "exactly parallel still merges");
        assert_eq!(groups[1], 1, "very close is not the same");
    }

    #[test]
    fn no_points_is_not_a_panic() {
        assert!(greedy_groups(&[], 0.9).is_empty());
    }
}
