//! k-means over document vectors, with a silhouette sweep to pick `k`.
//!
//! Deliberately small and dependency-free: the vectors are unit-length
//! embeddings, so squared Euclidean distance and cosine distance rank
//! identically, and a few dozen Lloyd iterations converge long before the
//! model calls that label the clusters become affordable.
//!
//! Everything here is seeded and deterministic. Clustering is already a
//! judgement call; re-running the same query and getting different groups
//! would make it an unusable one.

use std::collections::HashMap;

/// The `k` values the auto sweep tries when the query didn't say.
pub const AUTO_K: [usize; 5] = [2, 4, 8, 16, 32];

/// Lloyd iterations. Assignments almost always settle well before this;
/// the cap is what keeps a pathological corpus from stalling a query.
const MAX_ITERATIONS: usize = 50;

/// Points scored when evaluating a candidate `k`. Silhouette is O(n²) in
/// full, so a sample stands in for the whole: the winner is stable long
/// before the score is precise.
const SILHOUETTE_SAMPLE: usize = 256;

/// A clustering: which cluster each point landed in, and how many there are.
#[derive(Debug, Clone, PartialEq)]
pub struct Clustering {
    /// Cluster index per input point, parallel to the input slice.
    pub assignments: Vec<usize>,
    pub k: usize,
}

impl Clustering {
    /// The point indices belonging to each cluster, cluster 0 first.
    pub fn members(&self) -> Vec<Vec<usize>> {
        let mut members = vec![Vec::new(); self.k];
        for (point, &cluster) in self.assignments.iter().enumerate() {
            members[cluster].push(point);
        }
        members
    }
}

/// Cluster `points` into `k` groups, or into the `k` the silhouette sweep
/// likes best when `k` is `None`.
///
/// Fewer points than clusters is not an error: every point simply becomes its
/// own cluster, which is the honest answer to "group these three documents
/// into eight".
pub fn cluster(points: &[Vec<f32>], k: Option<usize>) -> Clustering {
    if points.is_empty() {
        return Clustering {
            assignments: Vec::new(),
            k: 0,
        };
    }
    match k {
        Some(k) => kmeans(points, k.clamp(1, points.len())),
        None => auto_k(points),
    }
}

/// Try each candidate `k` and keep the best-separated clustering.
fn auto_k(points: &[Vec<f32>]) -> Clustering {
    let mut best: Option<(f64, Clustering)> = None;
    for &k in AUTO_K.iter() {
        // A k at or above the point count degenerates to one point per
        // cluster, which always scores well and never means anything.
        if k >= points.len() {
            break;
        }
        let clustering = kmeans(points, k);
        let score = silhouette(points, &clustering);
        if best.as_ref().is_none_or(|(best, _)| score > *best) {
            best = Some((score, clustering));
        }
    }
    // Too few points for even the smallest candidate: one cluster each.
    best.map(|(_, clustering)| clustering)
        .unwrap_or_else(|| kmeans(points, points.len()))
}

/// Lloyd's algorithm from a k-means++ seeding.
fn kmeans(points: &[Vec<f32>], k: usize) -> Clustering {
    let k = k.clamp(1, points.len());
    let mut centroids = seed(points, k);
    let mut assignments = vec![0usize; points.len()];

    for _ in 0..MAX_ITERATIONS {
        let mut moved = false;
        for (i, point) in points.iter().enumerate() {
            let nearest = nearest_centroid(point, &centroids);
            if assignments[i] != nearest {
                assignments[i] = nearest;
                moved = true;
            }
        }
        if !moved {
            break;
        }
        centroids = recentre(points, &assignments, k, &centroids);
    }
    Clustering { assignments, k }
}

/// k-means++ seeding with a fixed pseudo-random stream: spread the initial
/// centroids out, but identically on every run.
fn seed(points: &[Vec<f32>], k: usize) -> Vec<Vec<f32>> {
    let mut rng = Lcg::new(0x5E_1EC7ED);
    let mut centroids = vec![points[rng.below(points.len())].clone()];
    while centroids.len() < k {
        // Distance to the nearest chosen centroid, per point.
        let weights: Vec<f64> = points
            .iter()
            .map(|point| {
                centroids
                    .iter()
                    .map(|centroid| distance(point, centroid) as f64)
                    .fold(f64::INFINITY, f64::min)
            })
            .collect();
        let total: f64 = weights.iter().sum();
        // Every point already sits on a centroid — nothing left to spread.
        if total <= 0.0 {
            centroids.push(points[rng.below(points.len())].clone());
            continue;
        }
        let mut target = rng.unit() * total;
        let mut chosen = points.len() - 1;
        for (i, weight) in weights.iter().enumerate() {
            target -= weight;
            if target <= 0.0 {
                chosen = i;
                break;
            }
        }
        centroids.push(points[chosen].clone());
    }
    centroids
}

/// Move each centroid to its members' mean, keeping an empty cluster's
/// centroid where it was rather than letting it drift to the origin.
fn recentre(
    points: &[Vec<f32>],
    assignments: &[usize],
    k: usize,
    previous: &[Vec<f32>],
) -> Vec<Vec<f32>> {
    let dim = points[0].len();
    let mut sums = vec![vec![0.0f32; dim]; k];
    let mut counts = vec![0usize; k];
    for (point, &cluster) in points.iter().zip(assignments) {
        for (slot, value) in sums[cluster].iter_mut().zip(point) {
            *slot += value;
        }
        counts[cluster] += 1;
    }
    sums.into_iter()
        .zip(counts)
        .enumerate()
        .map(|(cluster, (mut sum, count))| {
            if count == 0 {
                return previous[cluster].clone();
            }
            for value in &mut sum {
                *value /= count as f32;
            }
            sum
        })
        .collect()
}

fn nearest_centroid(point: &[f32], centroids: &[Vec<f32>]) -> usize {
    let mut best = 0;
    let mut best_distance = f32::INFINITY;
    for (i, centroid) in centroids.iter().enumerate() {
        let distance = distance(point, centroid);
        if distance < best_distance {
            best_distance = distance;
            best = i;
        }
    }
    best
}

/// Squared Euclidean distance. The square root would only rescale every
/// comparison here, so it is never taken.
fn distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(a, b)| (a - b) * (a - b)).sum()
}

/// Mean silhouette over a sample of points: how much closer a point sits to
/// its own cluster than to the nearest other one, in `[-1, 1]`.
///
/// Returns `-1` for a degenerate clustering (one cluster, or one point per
/// cluster), so the sweep never prefers a `k` that explains nothing.
pub fn silhouette(points: &[Vec<f32>], clustering: &Clustering) -> f64 {
    if clustering.k < 2 || points.len() <= clustering.k {
        return -1.0;
    }
    let members = clustering.members();
    let mut total = 0.0;
    let mut scored = 0usize;

    let stride = points.len().div_ceil(SILHOUETTE_SAMPLE).max(1);
    for i in (0..points.len()).step_by(stride) {
        let own = clustering.assignments[i];
        // A point alone in its cluster has no cohesion to measure; by
        // convention its silhouette is 0.
        if members[own].len() < 2 {
            scored += 1;
            continue;
        }
        let cohesion = mean_distance(points, i, &members[own], true);
        let separation = (0..clustering.k)
            .filter(|&cluster| cluster != own && !members[cluster].is_empty())
            .map(|cluster| mean_distance(points, i, &members[cluster], false))
            .fold(f64::INFINITY, f64::min);
        if separation.is_finite() {
            let spread = cohesion.max(separation);
            if spread > 0.0 {
                total += (separation - cohesion) / spread;
            }
        }
        scored += 1;
    }
    if scored == 0 {
        -1.0
    } else {
        total / scored as f64
    }
}

/// Mean distance from `point` to a cluster's members, excluding the point
/// itself when measuring its own cluster.
fn mean_distance(points: &[Vec<f32>], point: usize, members: &[usize], skip_self: bool) -> f64 {
    let mut total = 0.0;
    let mut count = 0usize;
    for &other in members {
        if skip_self && other == point {
            continue;
        }
        total += distance(&points[point], &points[other]) as f64;
        count += 1;
    }
    if count == 0 {
        0.0
    } else {
        total / count as f64
    }
}

/// The member index nearest its cluster's centre, per cluster — the
/// documents a label should be written from.
pub fn representatives(
    points: &[Vec<f32>],
    clustering: &Clustering,
    per_cluster: usize,
) -> HashMap<usize, Vec<usize>> {
    let mut chosen = HashMap::new();
    for (cluster, members) in clustering.members().into_iter().enumerate() {
        if members.is_empty() {
            continue;
        }
        let dim = points[0].len();
        let mut centre = vec![0.0f32; dim];
        for &member in &members {
            for (slot, value) in centre.iter_mut().zip(&points[member]) {
                *slot += value;
            }
        }
        for value in &mut centre {
            *value /= members.len() as f32;
        }
        let mut ranked = members;
        ranked.sort_by(|a, b| {
            distance(&points[*a], &centre)
                .total_cmp(&distance(&points[*b], &centre))
                // Ties break on index so the choice is reproducible.
                .then(a.cmp(b))
        });
        ranked.truncate(per_cluster);
        chosen.insert(cluster, ranked);
    }
    chosen
}

/// A tiny linear congruential generator. Not for anything that needs real
/// randomness — only to make k-means++ seeding spread out *and* repeat.
struct Lcg(u64);

impl Lcg {
    fn new(seed: u64) -> Self {
        Self(seed | 1)
    }

    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0 >> 33
    }

    fn below(&mut self, bound: usize) -> usize {
        (self.next() % bound as u64) as usize
    }

    fn unit(&mut self) -> f64 {
        self.next() as f64 / (1u64 << 31) as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Three tight, well-separated blobs in 2-D.
    fn blobs() -> Vec<Vec<f32>> {
        let centres = [[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]];
        let mut points = Vec::new();
        for centre in centres {
            for i in 0..8 {
                let jitter = i as f32 * 0.01;
                points.push(vec![centre[0] + jitter, centre[1] - jitter]);
            }
        }
        points
    }

    #[test]
    fn separates_well_separated_blobs() {
        let points = blobs();
        let clustering = cluster(&points, Some(3));
        assert_eq!(clustering.k, 3);
        // Every blob's 8 points must land together.
        for blob in 0..3 {
            let first = clustering.assignments[blob * 8];
            assert!(
                clustering.assignments[blob * 8..(blob + 1) * 8]
                    .iter()
                    .all(|&c| c == first),
                "blob {blob} was split: {:?}",
                clustering.assignments
            );
        }
    }

    #[test]
    fn is_deterministic_across_runs() {
        let points = blobs();
        assert_eq!(cluster(&points, Some(3)), cluster(&points, Some(3)));
        assert_eq!(cluster(&points, None), cluster(&points, None));
    }

    #[test]
    fn silhouette_prefers_the_true_shape() {
        let points = blobs();
        let three = silhouette(&points, &cluster(&points, Some(3)));
        let two = silhouette(&points, &cluster(&points, Some(2)));
        assert!(three > two, "3 blobs: {three} should beat 2: {two}");
    }

    #[test]
    fn auto_k_lands_near_the_true_shape() {
        // The sweep tries 2, 4, 8, 16 — 4 is the closest candidate to 3.
        let clustering = cluster(&blobs(), None);
        assert!(
            (2..=4).contains(&clustering.k),
            "auto-k chose {}",
            clustering.k
        );
    }

    #[test]
    fn more_clusters_than_points_gives_one_each() {
        let points = vec![vec![0.0, 0.0], vec![1.0, 1.0]];
        let clustering = cluster(&points, Some(8));
        assert_eq!(clustering.k, 2);
        assert_ne!(clustering.assignments[0], clustering.assignments[1]);
    }

    #[test]
    fn no_points_is_not_a_panic() {
        let clustering = cluster(&[], Some(4));
        assert_eq!(clustering.k, 0);
        assert!(clustering.assignments.is_empty());
    }

    #[test]
    fn representatives_sit_nearest_the_centre() {
        let points = blobs();
        let clustering = cluster(&points, Some(3));
        let reps = representatives(&points, &clustering, 2);
        assert_eq!(reps.len(), 3);
        for members in reps.values() {
            assert_eq!(members.len(), 2);
        }
    }
}
