use std::collections::{BTreeSet, HashMap, HashSet};

use itertools::{Itertools, iproduct};

pub type Coordinate = (usize, usize, usize);
pub type Delta = BTreeSet<Coordinate>; // effectively the support of a tensor of order 3

/// Helper struct to hold an order-3 tensor
#[derive(Clone, Debug)]
pub struct Tensor {
    delta: Delta,
    dims: (usize, usize, usize),
    nelements: usize,
}

impl Tensor {
    pub fn new(support: &[Coordinate]) -> Self {
        let mut delta = Delta::new();
        for &coordinate in support.iter() {
            delta.insert(coordinate);
        }
        let dims = support.iter().fold(
            (0usize, 0usize, 0usize),
            |(acc_x, acc_y, acc_z), &(x, y, z)| {
                (acc_x.max(x + 1), acc_y.max(y + 1), acc_z.max(z + 1))
            },
        );
        let nelements = delta.len();

        Self {
            delta,
            dims,
            nelements,
        }
    }

    pub fn empty() -> Self {
        Self {
            delta: Delta::new(),
            dims: (0, 0, 0),
            nelements: 0,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.nelements == 0
    }

    pub fn delta(&self) -> &Delta {
        &self.delta
    }

    /// Check if tensor is a subset of a Latin square
    pub fn is_partial_latin_square(&self) -> bool {
        let mut xy: HashSet<(usize, usize)> = HashSet::new(); // x, y -> z uniqueness
        let mut yz: HashSet<(usize, usize)> = HashSet::new(); // y, z -> x uniqueness
        let mut zx: HashSet<(usize, usize)> = HashSet::new(); // z, x -> y uniqueness

        for &(x, y, z) in self.delta() {
            if !xy.insert((x, y)) {
                return false;
            }
            if !yz.insert((y, z)) {
                return false;
            }
            if !zx.insert((z, x)) {
                return false;
            }
        }

        true
    }
}

#[derive(Eq, Hash, PartialEq)]
struct NormalizedDelta {
    coordinates: Vec<Coordinate>,
}

impl NormalizedDelta {
    fn from_tensor(tensor: &Tensor) -> Self {
        if tensor.is_empty() {
            return Self {
                coordinates: vec![],
            };
        }

        let (dim_x, dim_y, dim_z) = tensor.dims;
        let x_perms: Vec<Vec<usize>> = (0..dim_x).permutations(dim_x).collect();
        let y_perms: Vec<Vec<usize>> = (0..dim_y).permutations(dim_y).collect();
        let z_perms: Vec<Vec<usize>> = (0..dim_z).permutations(dim_z).collect();

        // try all combinations of permutations to find the canonical (lex smallest) representation
        let mut canonical_form = None;

        for x_perm in &x_perms {
            for y_perm in &y_perms {
                for z_perm in &z_perms {
                    let mut transformed: Vec<Coordinate> = tensor
                        .delta
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

fn generate_all_tensors(dims: (usize, usize, usize), nonzero_elements: usize) -> Vec<Tensor> {
    if nonzero_elements == 0 {
        return vec![Tensor::empty()];
    }

    let (dim_x, dim_y, dim_z) = dims;

    if nonzero_elements > dim_x * dim_y * dim_z {
        return vec![];
    }

    let all_coords: Vec<Coordinate> = iproduct!(0..dim_x, 0..dim_y, 0..dim_z)
        .map(|(i, j, k)| (i, j, k))
        .collect();

    all_coords
        .into_iter()
        .combinations(nonzero_elements)
        .map(|combos| Tensor::new(&combos))
        .collect()
}

fn normalize_and_classify_tensors(tensors: Vec<Tensor>) -> Vec<Vec<Tensor>> {
    let mut classes: HashMap<NormalizedDelta, Vec<Tensor>> = HashMap::new();

    for tensor in tensors {
        let normalized = NormalizedDelta::from_tensor(&tensor);
        classes.entry(normalized).or_default().push(tensor);
    }

    classes.into_values().collect()
}

/// Struct holding all order-3 tensor isomorphism classes for tensors
pub struct TensorIsomorphisms {
    dims: (usize, usize, usize),
    isomorphism_classes: HashMap<usize, Vec<Vec<Tensor>>>,
}

impl TensorIsomorphisms {
    pub fn new_square(dim: usize) -> Self {
        Self::new((dim, dim, dim))
    }

    pub fn new(dims: (usize, usize, usize)) -> Self {
        let mut isomorphism_classes: HashMap<usize, Vec<Vec<Tensor>>> = HashMap::new();

        for nonzero_elements in 0..=(dims.0 * dims.1 * dims.2) {
            let tensors = generate_all_tensors(dims, nonzero_elements);
            let classes = normalize_and_classify_tensors(tensors);
            isomorphism_classes.insert(nonzero_elements, classes);
        }

        Self {
            dims,
            isomorphism_classes,
        }
    }

    pub fn get_isomorphism_classes(&self) -> HashMap<usize, Vec<Vec<Tensor>>> {
        self.isomorphism_classes.clone()
    }

    /// Get only the tensor isomorphisms of tensors that are (partial) latin squares
    pub fn get_partial_latin_squares(tensor_isomorphisms: &Self) -> Self {
        let latin_square_tensor_isomorphisms = tensor_isomorphisms
            .get_isomorphism_classes()
            .iter()
            .map(|(&nelems, classes)| {
                let filtered_tensors: Vec<Vec<Tensor>> = classes
                    .iter()
                    .map(|tensors| {
                        tensors
                            .iter()
                            .filter(|&tensor| tensor.is_partial_latin_square())
                            .cloned()
                            .collect::<Vec<Tensor>>()
                    })
                    .filter(|filtered_vec| !filtered_vec.is_empty())
                    .collect();

                (nelems, filtered_tensors)
            })
            .filter(|(_, classes)| !classes.is_empty())
            .collect();
        Self {
            dims: tensor_isomorphisms.dims,
            isomorphism_classes: latin_square_tensor_isomorphisms,
        }
    }

    fn filter_max_nonzero_elements(tensor_isomorphisms: &Self, max_elements: usize) -> Self {
        let filtered = tensor_isomorphisms
            .get_isomorphism_classes()
            .iter()
            .filter(|&(nelems, _)| *nelems <= max_elements)
            .map(|(&nelems, classes)| (nelems, classes.clone()))
            .collect();
        Self {
            dims: tensor_isomorphisms.dims,
            isomorphism_classes: filtered,
        }
    }

    /// Get only the tensor isomorphisms that have at most as many elements as the max
    /// dimension
    pub fn get_d_sparse(tensor_isomorphisms: &Self) -> Self {
        let dims = tensor_isomorphisms.dims;
        let d = dims.0.max(dims.1).max(dims.2);
        Self::filter_max_nonzero_elements(tensor_isomorphisms, d)
    }

    /// Get only the tensor isomorphisms that have at most as many elements as the
    /// largest dimension times the second largest dimension
    pub fn get_d2_sparse(tensor_isomorphisms: &Self) -> Self {
        // Calculate d2 by multiplying the two largest dimensions. This is equal
        // to the number of elements that fit into a Latin square
        let (dx, dy, dz) = tensor_isomorphisms.dims;
        let dim1 = dx.max(dy).max(dz);
        let dim2 = match (dx, dy, dz) {
            _ if dx == dim1 => dy.max(dz),
            _ if dy == dim1 => dx.max(dz),
            _ => dx.max(dy),
        };

        Self::filter_max_nonzero_elements(tensor_isomorphisms, dim1 * dim2)
    }

    /// Iterate through all order-3 tensor isomorphism classes for tensors and print them.
    /// For each isomorphism class, we print out one representative.
    pub fn print_tensor_isomorphism_classes(tensor_isomorphisms: &Self) {
        for (nonzero_elements, classes) in tensor_isomorphisms
            .get_isomorphism_classes()
            .iter()
            .sorted_by_key(|&(&n, _)| n)
        {
            println!("nonzero elements: {}", nonzero_elements);
            println!("nof classes: {}", classes.len());
            for (i, class) in classes.iter().enumerate() {
                if let Some(representative) = class.first() {
                    println!("\tclass {}: {:?}", i + 1, representative);
                }
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
    fn test_generate_all_tensors_small() {
        let tensors = generate_all_tensors((1, 1, 1), 1);
        assert_eq!(tensors.len(), 1);
        assert_eq!(tensors[0].delta, create_delta(&[(0, 0, 0)]));
    }

    #[test]
    fn test_generate_all_tensors_count() {
        let tensors = generate_all_tensors((2, 2, 2), 2);
        assert_eq!(tensors.len(), 28);
    }

    #[test]
    fn test_classify_isomorphic_deltas() {
        let tensor1 = Tensor::new(&[(0, 0, 0), (0, 1, 1)]);
        let tensor2 = Tensor::new(&[(1, 1, 1), (1, 2, 2)]);
        let tensors = vec![tensor1, tensor2];

        let classes = normalize_and_classify_tensors(tensors);
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
            let tensors = generate_all_tensors((dim, dim, dim), nonzero_elements);
            let classes = normalize_and_classify_tensors(tensors);
            assert_eq!(classes.len(), expected_count);
        }
    }

    #[test]
    fn test_is_partial_latin_square_valid() {
        let tensor = Tensor::new(&[(0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 0)]);
        assert!(tensor.is_partial_latin_square());

        // both y=0 and y=1 have duplicates
        let tensor2 = Tensor::new(&[(0, 0, 0), (0, 1, 1), (1, 0, 0), (1, 1, 1)]);
        assert!(!tensor2.is_partial_latin_square());

        // duplicate for y=0
        let tensor3 = Tensor::new(&[(0, 0, 0), (0, 1, 1), (1, 0, 0)]);
        assert!(!tensor3.is_partial_latin_square());

        // duplicate for y=0
        let tensor4 = Tensor::new(&[(0, 0, 0), (1, 0, 0)]);
        assert!(!tensor4.is_partial_latin_square());
    }

    #[test]
    fn test_getting_latin_squares() {
        let dim = 2;
        let cases = HashMap::from([
            (0, vec![1]),
            (1, vec![8]),
            (2, vec![4, 4, 4, 4]),
            (3, vec![8]),
            (4, vec![2]),
        ]);
        let latin_square_classes =
            TensorIsomorphisms::get_partial_latin_squares(&TensorIsomorphisms::new_square(dim));

        assert_eq!(
            cases.len(),
            latin_square_classes.get_isomorphism_classes().len()
        );

        for (&nelems, classes) in latin_square_classes.get_isomorphism_classes().iter() {
            assert!(nelems <= dim * dim);
            assert!(cases.contains_key(&nelems));
            let reference = cases.get(&nelems).unwrap();
            let tensors_per_class = &classes.iter().map(|t| t.len()).collect::<Vec<usize>>();
            assert_eq!(reference, tensors_per_class);
        }
    }

    #[test]
    fn test_getting_sparsity_filtering() {
        let dim = 2;
        let tensor_isomorphisms = TensorIsomorphisms::new_square(dim);

        let d_sparse = TensorIsomorphisms::get_d_sparse(&tensor_isomorphisms);
        assert_eq!(dim + 1, d_sparse.get_isomorphism_classes().len());

        let d2_sparse = TensorIsomorphisms::get_d2_sparse(&tensor_isomorphisms);
        assert_eq!(dim * dim + 1, d2_sparse.get_isomorphism_classes().len());
    }
}
