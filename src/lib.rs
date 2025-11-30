pub mod isomorphism;
pub mod restriction;

use ndarray::{ArrayD, IxDyn};

use crate::isomorphism::Delta;

/// Returns the `r`-th unit tensor of order `n`
pub fn order_n_unit_tensor(n: usize, r: usize) -> ArrayD<u32> {
    let shape = vec![r; n];
    let mut tensor = ArrayD::<u32>::zeros(IxDyn(&shape));

    for i in 0..r {
        let index = vec![i; n];
        tensor[&index[..]] = 1;
    }

    tensor
}

/// Returns `r`-th unit tensor of order 3
pub fn unit_tensor(r: usize) -> ArrayD<u32> {
    order_n_unit_tensor(3, r)
}

/// Returns matrix multiplication tensor for multiplying a `n` times `m` matrix
/// by a `m` times `p` matrix
pub fn matrix_multiplication_tensor(n: usize, m: usize, p: usize) -> ArrayD<u32> {
    let mut tensor = ArrayD::<u32>::zeros(IxDyn(&[n * m, m * p, p * n]));

    for i in 0..n {
        for j in 0..m {
            for k in 0..p {
                tensor[[i * m + j, j * p + k, k * n + i]] = 1;
            }
        }
    }

    tensor
}

/// Returns matrix multiplication tensor for multiplying two `n` times `n` matrices
pub fn square_matrix_multiplication_tensor(n: usize) -> ArrayD<u32> {
    matrix_multiplication_tensor(n, n, n)
}

/// Returns delta (support) of `r`-th unit tensor of order 3
pub fn unit_tensor_delta(r: usize) -> Delta {
    let mut delta = Delta::new();

    for i in 0..r {
        delta.insert((i, i, i));
    }

    delta
}
