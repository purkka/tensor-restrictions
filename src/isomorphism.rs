use std::collections::{BTreeSet, HashMap};

use itertools::{Itertools, iproduct};

type Coordinate = (usize, usize, usize);
type Pattern = BTreeSet<Coordinate>;

#[derive(Eq, Hash, PartialEq)]
struct NormalizedPattern {
    coordinates: Vec<Coordinate>,
    dimensions: (usize, usize, usize),
}

impl NormalizedPattern {
    fn from_pattern(pattern: &Pattern) -> Self {
        fn sort_and_get_coordinate_map(coordinates: &mut Vec<usize>) -> HashMap<usize, usize> {
            coordinates.dedup();
            coordinates.sort();

            coordinates
                .iter()
                .enumerate()
                .map(|(i, &x)| (x, i))
                .collect()
        }

        let mut x_coords: Vec<usize> = pattern.iter().map(|&(x, _, _)| x).collect();
        let mut y_coords: Vec<usize> = pattern.iter().map(|&(_, y, _)| y).collect();
        let mut z_coords: Vec<usize> = pattern.iter().map(|&(_, _, z)| z).collect();

        let x_map: HashMap<usize, usize> = sort_and_get_coordinate_map(&mut x_coords);
        let y_map: HashMap<usize, usize> = sort_and_get_coordinate_map(&mut y_coords);
        let z_map: HashMap<usize, usize> = sort_and_get_coordinate_map(&mut z_coords);

        let mut normalized_coords: Vec<Coordinate> = pattern
            .iter()
            .map(|&(x, y, z)| {
                (
                    *x_map.get(&x).unwrap(),
                    *y_map.get(&y).unwrap(),
                    *z_map.get(&z).unwrap(),
                )
            })
            .collect();

        normalized_coords.sort();

        let dimensions = (x_coords.len(), y_coords.len(), z_coords.len());

        Self {
            coordinates: normalized_coords,
            dimensions,
        }
    }
}

struct PatternGenerator {
    d: usize, // dimension d x d x d
    n: usize, // nof nonzero elements
}

impl PatternGenerator {
    fn new(d: usize, n: usize) -> Self {
        Self { d, n }
    }

    fn generate_all_patterns(&self) -> Vec<Pattern> {
        if self.n == 0 {
            return vec![BTreeSet::new()];
        }

        if self.n > self.d * self.d * self.d {
            return vec![];
        }

        let all_coords: Vec<Coordinate> = iproduct!(0..self.d, 0..self.d, 0..self.d)
            .map(|(i, j, k)| (i, j, k))
            .collect();

        all_coords
            .into_iter()
            .combinations(self.n)
            .map(|combos| combos.into_iter().collect())
            .collect()
    }
}

struct IsomorphismClassifier;

impl IsomorphismClassifier {
    fn classify_patterns(patterns: Vec<Pattern>) -> (usize, Vec<Vec<Pattern>>) {
        let mut classes: HashMap<NormalizedPattern, Vec<Pattern>> = HashMap::new();

        for pattern in patterns {
            let normalized = NormalizedPattern::from_pattern(&pattern);
            classes.entry(normalized).or_default().push(pattern);
        }

        let class_count = classes.len();
        let classes_vec: Vec<Vec<Pattern>> = classes.into_values().collect();

        (class_count, classes_vec)
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
        let generator = PatternGenerator::new(1, 1);
        let patterns = generator.generate_all_patterns();
        assert_eq!(patterns.len(), 1);
        assert_eq!(patterns[0], create_pattern(&[(0, 0, 0)]));
    }

    #[test]
    fn test_generate_all_patterns_count() {
        let generator = PatternGenerator::new(2, 2);
        let patterns = generator.generate_all_patterns();
        assert_eq!(patterns.len(), 28);
    }

    #[test]
    fn test_classify_isomorphic_patterns() {
        let pattern1 = create_pattern(&[(0, 0, 0), (0, 1, 1)]);
        let pattern2 = create_pattern(&[(1, 1, 1), (1, 2, 2)]);
        let patterns = vec![pattern1, pattern2];

        let (count, classes) = IsomorphismClassifier::classify_patterns(patterns);
        assert_eq!(count, 1);
        assert_eq!(classes[0].len(), 2);
    }
}
