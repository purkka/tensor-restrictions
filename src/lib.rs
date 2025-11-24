use std::fmt::Display;

use itertools::Itertools;
use ndarray::{ArrayD, IxDyn};
use symbolica::{
    atom::{Atom, AtomCore},
    domains::finite_field::Zp,
    poly::{groebner::GroebnerBasis, polynomial::MultivariatePolynomial},
    symbol,
};

/// Returns `r`-th unit tensor of order `order`
pub fn unit_tensor(order: usize, r: usize) -> ArrayD<u32> {
    let shape = vec![r; order];
    let mut tensor = ArrayD::<u32>::zeros(IxDyn(&shape));

    for i in 0..r {
        let index = vec![i; order];
        tensor[&index[..]] = 1;
    }

    tensor
}

fn create_variable<A, B, C>(a: A, b: B, c: C) -> Atom
where
    A: Display,
    B: Display,
    C: Display,
{
    Atom::from(symbol!(&format!("v_{a}_{b}_{c}")))
}

/// Converts an order-n `tensor` into a multivariate polynomial with the given index labels `e`.
///
/// The function returns the sum over all valid tensor index tuples `(i_1, i_2, ..., i_n)`
///
/// ```text
/// ∑_{i_1, i_2, ..., i_n} T[i_1, i_2, ..., i_n] * ∏_{k=1}^{n} v_<k>_<e_k>_<i_k>
/// ```
///
/// Returns an error if the length of `e` does not equal the order of `tensor`.
pub fn tensor_as_polynomial(tensor: &ArrayD<u32>, e: &[u32]) -> anyhow::Result<Atom> {
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
                let variable = create_variable(k + 1, e_k, tensor_index + 1);

                product *= variable;
            }

            sum += product;
        }
    }

    Ok(sum)
}

/// Checks whether a tensor `tensor_s` reduces to another tensor `tensor_t`
/// via a Groebner basis computation.
///
/// Rteurns an error if the orders of the tensors do not match.
pub fn tensor_reduces_to(tensor_s: &ArrayD<u32>, tensor_t: &ArrayD<u32>) -> anyhow::Result<bool> {
    let dimensions_s = tensor_s.shape().to_vec();
    let dimensions_t = tensor_t.shape().to_vec();

    if dimensions_s.len() != dimensions_t.len() {
        anyhow::bail!("Tensor orders do not match")
    }

    // create all variables that can appear
    let mut variables: Vec<Atom> = Vec::new();
    for (tensor_axis, (&dim_s, &dim_t)) in dimensions_s.iter().zip(dimensions_t.iter()).enumerate()
    {
        for (idx_s, idx_t) in (0..dim_s).cartesian_product(0..dim_t) {
            variables.push(create_variable(tensor_axis + 1, idx_s + 1, idx_t + 1));
        }
    }

    // build polynomial system
    let mut poly: Vec<MultivariatePolynomial<_, u32>> = Vec::new();
    let field = Zp::new(1_000_000_007); // big prime that fits into u32

    for index_set in dimensions_s.iter().map(|&d| 0..d).multi_cartesian_product() {
        // for now we use 1-indexing
        let e: Vec<u32> = index_set.iter().map(|&ei| (ei as u32) + 1).collect();
        let te_poly = tensor_as_polynomial(tensor_t, &e)?;

        let tensor_s_element = tensor_s[&index_set[..]];
        let tensor_s_atom = Atom::num(tensor_s_element);

        let diff = te_poly - tensor_s_atom;

        poly.push(diff.to_polynomial(&field, None));
    }

    let groebner_basis = GroebnerBasis::new(&poly, false);
    let has_one = groebner_basis.system.iter().any(|p| p.is_one());

    Ok(!has_one)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    // everything is in a single unit test, as an unlicensed symbolica restricts to running on one thread in one core (https://symbolica.io/docs/get_started.html#license)
    #[test]
    fn test_tensor_reduces() {
        let m1 = array![[1, 0], [0, 0]].into_dyn();
        let m2 = array![[1, 0], [0, 1]].into_dyn();
        let m3 = array![[1, 0, 0], [0, 1, 0], [0, 0, 1]].into_dyn();

        assert!(matches!(tensor_reduces_to(&m1, &m2), Ok(true)));
        assert!(matches!(tensor_reduces_to(&m2, &m1), Ok(false)));
        assert!(matches!(tensor_reduces_to(&m2, &m3), Ok(true)));
        assert!(matches!(tensor_reduces_to(&m3, &m1), Ok(false)));
        assert!(matches!(tensor_reduces_to(&m3, &m2), Ok(false)));
        assert!(matches!(tensor_reduces_to(&m1, &m3), Ok(true)));

        let p1 = array![[[1, 0], [0, 0]], [[0, 1], [1, 0]]].into_dyn();
        let r1 = unit_tensor(3, 1);
        let r2 = unit_tensor(3, 2);
        let r3 = unit_tensor(3, 3);

        assert!(matches!(tensor_reduces_to(&p1, &r1), Ok(false)));
        assert!(matches!(tensor_reduces_to(&p1, &r2), Ok(false)));
        assert!(matches!(tensor_reduces_to(&p1, &r3), Ok(true)));
    }
}
