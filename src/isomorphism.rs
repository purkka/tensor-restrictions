use std::collections::{BTreeSet, HashMap};

use itertools::{Itertools, iproduct};

pub type Coordinate = (usize, usize, usize);
pub type Delta = BTreeSet<Coordinate>; // effectively the support of a tensor of order 3

#[derive(Eq, Hash, PartialEq)]
struct NormalizedDelta {
    coordinates: Vec<Coordinate>,
}

impl NormalizedDelta {
    fn from_delta(delta: &Delta, dim: usize) -> Self {
        if delta.is_empty() {
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
                    let mut transformed: Vec<Coordinate> = delta
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

fn generate_all_deltas(dim: usize, nonzero_elements: usize) -> Vec<Delta> {
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

fn normalize_and_classify_deltas(deltas: Vec<Delta>, dim: usize) -> Vec<Vec<Delta>> {
    let mut classes: HashMap<NormalizedDelta, Vec<Delta>> = HashMap::new();

    for delta in deltas {
        let normalized = NormalizedDelta::from_delta(&delta, dim);
        classes.entry(normalized).or_default().push(delta);
    }

    classes.into_values().collect()
}

/// Iterate through all order-3 tensor isomorphism classes for tensors
/// of dimension `dim` x `dim` x `dim`.
pub fn get_isomorphism_classes(dim: usize) -> HashMap<usize, Vec<Vec<Delta>>> {
    let mut results: HashMap<usize, Vec<Vec<Delta>>> = HashMap::new();

    for nonzero_elements in 0..=(dim * dim * dim) {
        let deltas = generate_all_deltas(dim, nonzero_elements);
        let classes = normalize_and_classify_deltas(deltas, dim);
        results.insert(nonzero_elements, classes);
    }

    results
}

/// Iterate through all order-3 tensor isomorphism classes for tensors
/// of dimension `dim` x `dim` x `dim` and print them. For each isomorphism
/// class, we print out one representative.
pub fn print_tensor_isomorphism_classes(dim: usize) {
    let results = get_isomorphism_classes(dim);

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

    fn create_delta(coords: &[(usize, usize, usize)]) -> Delta {
        coords.iter().cloned().collect()
    }

    #[test]
    fn test_generate_all_deltas_small() {
        let deltas = generate_all_deltas(1, 1);
        assert_eq!(deltas.len(), 1);
        assert_eq!(deltas[0], create_delta(&[(0, 0, 0)]));
    }

    #[test]
    fn test_generate_all_deltas_count() {
        let deltas = generate_all_deltas(2, 2);
        assert_eq!(deltas.len(), 28);
    }

    #[test]
    fn test_classify_isomorphic_deltas() {
        let delta1 = create_delta(&[(0, 0, 0), (0, 1, 1)]);
        let delta2 = create_delta(&[(1, 1, 1), (1, 2, 2)]);
        let deltas = vec![delta1, delta2];

        let classes = normalize_and_classify_deltas(deltas, 3);
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
            let deltas = generate_all_deltas(dim, nonzero_elements);
            let classes = normalize_and_classify_deltas(deltas, dim);
            assert_eq!(classes.len(), expected_count);
        }
    }
}
