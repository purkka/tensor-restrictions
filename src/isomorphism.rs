use std::collections::{BTreeSet, HashMap};

use itertools::{Itertools, iproduct};

type Coordinate = (usize, usize, usize);
type Pattern = BTreeSet<Coordinate>;

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
                .map(|(i, &x)| (x, i + 1))
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

        let dimensions = (
            *x_coords.iter().max().unwrap_or(&0),
            *y_coords.iter().max().unwrap_or(&0),
            *z_coords.iter().max().unwrap_or(&0),
        );

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
