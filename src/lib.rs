use itertools::Itertools;
use ndarray::ArrayD;
use symbolica::{atom::Atom, symbol};

/// Converts an order-n `tensor` into a multivariate polynomial with the given index labels `e`.
///
/// The function returns the sum over all valid tensor index tuples `(i_1, i_2, ..., i_n)`
///
/// ```text
/// ∑_{i_1, i_2, ..., i_n} T[i_1, i_2, ..., i_n] * ∏_{k=1}^{n} v_<k>_<e_k>_<i_k>
/// ```
///
/// Returns an error if the length of `e` does not equal the order of `tensor`.
pub fn tensor_as_polynomial(tensor: &ArrayD<i64>, e: &[i64]) -> anyhow::Result<Atom> {
    let dimensions = tensor.shape().to_vec();

    let e_len = e.len();
    let tensor_order = dimensions.len();

    if e_len != tensor_order {
        anyhow::bail!(
            "Index label vector e has length {e_len}, but tensor has order {tensor_order}",
        );
    }

    let mut sum: Atom = Atom::Zero;

    for tensor_index_set in dimensions.iter().map(|&n| 0..n).multi_cartesian_product() {
        let tensor_element = tensor[&tensor_index_set[..]];

        if tensor_element != 0 {
            let mut product = Atom::num(tensor_element);

            for (k, (&e_k, &tensor_index)) in e.iter().zip(tensor_index_set.iter()).enumerate() {
                // TODO Check if we want 0-indexing instead
                let variable =
                    Atom::from(symbol!(&format!("v_{}_{e_k}_{}", k + 1, tensor_index + 1)));

                product *= variable;
            }

            sum += product;
        }
    }

    Ok(sum)
}

// TODO Add unit tests
