use std::collections::{BTreeSet, HashMap};

use itertools::{Itertools, iproduct};

type Coordinate = (usize, usize, usize);
type Pattern = BTreeSet<Coordinate>;

#[derive(Eq, Hash, PartialEq)]
struct NormalizedPattern {
    coordinates: Vec<Coordinate>,
}

impl NormalizedPattern {
    fn from_pattern(pattern: &Pattern, dim: usize) -> Self {
        if pattern.is_empty() {
            return Self {
                coordinates: vec![],
            };
        }

        let indices_per_axis: Vec<usize> = (0..dim).collect();
        let x_perms: Vec<Vec<usize>> = indices_per_axis.iter().cloned().permutations(dim).collect();
        let y_perms: Vec<Vec<usize>> = indices_per_axis.iter().cloned().permutations(dim).collect();
        let z_perms: Vec<Vec<usize>> = indices_per_axis.iter().cloned().permutations(dim).collect();

        // try all combinations of permutations to find the canonical (lex smallest) representation
        let mut canonical_form = None;

        for x_perm in &x_perms {
            for y_perm in &y_perms {
                for z_perm in &z_perms {
                    let mut transformed: Vec<Coordinate> = pattern
                        .iter()
                        .map(|&(x, y, z)| {
                            let new_x = x_perm[x];
                            let new_y = y_perm[y];
                            let new_z = z_perm[z];
                            (new_x, new_y, new_z)
                        })
                        .collect();

                    transformed.sort();

                    if canonical_form.is_none() || transformed < canonical_form.clone().unwrap() {
                        canonical_form = Some(transformed);
                    }
                }
            }
        }

        Self {
            coordinates: canonical_form.unwrap(),
        }
    }
}

fn generate_all_patterns(dim: usize, nonzero_elements: usize) -> Vec<Pattern> {
    if nonzero_elements == 0 {
        return vec![BTreeSet::new()];
    }

    if nonzero_elements > dim * dim * dim {
        return vec![];
    }

    let all_coords: Vec<Coordinate> = iproduct!(0..dim, 0..dim, 0..dim)
        .map(|(i, j, k)| (i, j, k))
        .collect();

    all_coords
        .into_iter()
        .combinations(nonzero_elements)
        .map(|combos| combos.into_iter().collect())
        .collect()
}

fn normalize_and_classify_patterns(patterns: Vec<Pattern>, dim: usize) -> Vec<Vec<Pattern>> {
    let mut classes: HashMap<NormalizedPattern, Vec<Pattern>> = HashMap::new();

    for pattern in patterns {
        let normalized = NormalizedPattern::from_pattern(&pattern, dim);
        classes.entry(normalized).or_default().push(pattern);
    }

    classes.into_values().collect()
}

/// Iterate through all order-3 tensor isomorphism classes for tensors
/// of dimension `dim` x `dim` x `dim` and print them. For each isomorphism
/// class, we print out one representative.
pub fn print_tensor_isomorphism_classes(dim: usize) {
    let mut results: HashMap<usize, Vec<Vec<Pattern>>> = HashMap::new();

    for nonzero_elements in 0..=(dim * dim * dim) {
        let patterns = generate_all_patterns(dim, nonzero_elements);
        let classes = normalize_and_classify_patterns(patterns, dim);
        results.insert(nonzero_elements, classes);
    }

    for (nonzero_elements, classes) in results.iter().sorted_by_key(|&(&n, _)| n) {
        println!("nonzero elements: {}", nonzero_elements);
        println!("nof classes: {}", classes.len());
        for (i, class) in classes.iter().enumerate() {
            if let Some(representative) = class.first() {
                println!("\tclass {}: {:?}", i + 1, representative);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_pattern(coords: &[(usize, usize, usize)]) -> Pattern {
        coords.iter().cloned().collect()
    }

    #[test]
    fn test_generate_all_patterns_small() {
        let patterns = generate_all_patterns(1, 1);
        assert_eq!(patterns.len(), 1);
        assert_eq!(patterns[0], create_pattern(&[(0, 0, 0)]));
    }

    #[test]
    fn test_generate_all_patterns_count() {
        let patterns = generate_all_patterns(2, 2);
        assert_eq!(patterns.len(), 28);
    }

    #[test]
    fn test_classify_isomorphic_patterns() {
        let pattern1 = create_pattern(&[(0, 0, 0), (0, 1, 1)]);
        let pattern2 = create_pattern(&[(1, 1, 1), (1, 2, 2)]);
        let patterns = vec![pattern1, pattern2];

        let classes = normalize_and_classify_patterns(patterns, 3);
        assert_eq!(classes.len(), 1);
        assert_eq!(classes[0].len(), 2);
    }

    #[test]
    fn test_known_counts_d2() {
        let dim = 2;

        let cases = vec![
            (0, 1),
            (1, 1),
            (2, 7),
            (3, 7),
            (4, 14),
            (5, 7),
            (6, 7),
            (7, 1),
            (8, 1),
        ];

        for (nonzero_elements, expected_count) in cases {
            let patterns = generate_all_patterns(dim, nonzero_elements);
            let classes = normalize_and_classify_patterns(patterns, dim);
            assert_eq!(classes.len(), expected_count);
        }
    }
}
