use std::{fmt::Display, sync::Arc};

use itertools::Itertools;
use ndarray::{ArrayD, IxDyn};
use symbolica::{
    domains::{
        RingOps,
        finite_field::{FiniteFieldCore, Zp},
    },
    poly::{PolyVariable, groebner::GroebnerBasis, polynomial::MultivariatePolynomial},
    symbol,
};

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

pub fn square_matrix_multiplication_tensor(n: usize) -> ArrayD<u32> {
    matrix_multiplication_tensor(n, n, n)
}

fn create_variable<T: Display>(a: T, b: T, c: T) -> PolyVariable {
    PolyVariable::Symbol(symbol!(&format!("v_{a}_{b}_{c}")))
}

fn make_poly_vars(dims_s: &[usize], dims_t: &[usize]) -> Arc<Vec<PolyVariable>> {
    let mut vars = Vec::new();
    for (axis, (&ds, &dt)) in dims_s.iter().zip(dims_t.iter()).enumerate() {
        for (idx_s, idx_t) in (0..ds).cartesian_product(0..dt) {
            vars.push(create_variable(axis, idx_s, idx_t));
        }
    }
    Arc::new(vars)
}

fn get_flat_indices(dims_s: &[usize], dims_t: &[usize]) -> Vec<Vec<Vec<usize>>> {
    let order = dims_s.len();
    let mut axis_offsets = vec![0; order];
    for a in 1..order {
        axis_offsets[a] = axis_offsets[a - 1] + dims_s[a - 1] * dims_t[a - 1];
    }
    let mut flat_index_lut = Vec::with_capacity(order);
    for axis in 0..order {
        let mut table_axis = Vec::with_capacity(dims_s[axis]);
        for i_s in 0..dims_s[axis] {
            let mut row = Vec::with_capacity(dims_t[axis]);
            for i_t in 0..dims_t[axis] {
                let index = axis_offsets[axis] + i_s * dims_t[axis] + i_t;
                row.push(index);
            }
            table_axis.push(row);
        }
        flat_index_lut.push(table_axis);
    }

    flat_index_lut
}

/// Checks whether a tensor `tensor_s` reduces to another tensor `tensor_t`
/// via a Groebner basis computation.
///
/// Returns an error if the orders of the tensors do not match.
pub fn tensor_restriction_of(
    tensor_s: &ArrayD<u32>,
    tensor_t: &ArrayD<u32>,
) -> anyhow::Result<bool> {
    let dimensions_s = tensor_s.shape().to_vec();
    let dimensions_t = tensor_t.shape().to_vec();

    if dimensions_s.len() != dimensions_t.len() {
        anyhow::bail!("Tensor orders do not match")
    }

    // precompute indices
    let flat_index_lut = get_flat_indices(&dimensions_s, &dimensions_t);

    let s_ranges: Vec<std::ops::Range<usize>> = dimensions_s.iter().map(|&d| 0..d).collect();
    let t_ranges: Vec<std::ops::Range<usize>> = dimensions_t.iter().map(|&d| 0..d).collect();

    let variable_map = Arc::new(make_poly_vars(&dimensions_s, &dimensions_t));
    let var_count = variable_map.len();

    let field = Zp::new(1_000_000_007);

    let mut polynomial_system: Vec<MultivariatePolynomial<_, u32>> = Vec::new();

    // loop through elements of `tensor_s`
    for index_s in s_ranges.iter().cloned().multi_cartesian_product() {
        let mut polynomial =
            MultivariatePolynomial::new(&field, Some(var_count), Arc::clone(&variable_map));

        // populate with terms from `tensor_t`
        for index_t in t_ranges.iter().cloned().multi_cartesian_product() {
            let value_t = tensor_t[&index_t[..]];
            // only handle nonzero elements
            if value_t == 0 {
                continue;
            }

            let mut exponents = vec![0u32; variable_map.len()];

            for (axis, (&idx_s, &idx_t)) in index_s.iter().zip(index_t.iter()).enumerate() {
                // let var_index = get_flat_index(axis, idx_s, idx_t, &dimensions_s, &dimensions_t);
                let var_index = flat_index_lut[axis][idx_s][idx_t];
                exponents[var_index] = 1;
            }

            polynomial.append_monomial(field.to_element(value_t), &exponents);
        }

        let value_s = tensor_s[&index_s[..]];

        // polynomial equals zero iff a valid mapping exists from `tensor_t` to `tensor_s`
        let zero_exps = vec![0u32; variable_map.len()];
        polynomial.append_monomial(
            field.mul(field.neg(&field.to_element(1)), field.to_element(value_s)),
            &zero_exps,
        );

        polynomial_system.push(polynomial);
    }

    // if the system contains 1, no mapping exists
    let groebner_basis = GroebnerBasis::new(&polynomial_system, false);
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

        assert!(matches!(tensor_restriction_of(&m1, &m2), Ok(true)));
        assert!(matches!(tensor_restriction_of(&m2, &m1), Ok(false)));
        assert!(matches!(tensor_restriction_of(&m2, &m3), Ok(true)));
        assert!(matches!(tensor_restriction_of(&m3, &m1), Ok(false)));
        assert!(matches!(tensor_restriction_of(&m3, &m2), Ok(false)));
        assert!(matches!(tensor_restriction_of(&m1, &m3), Ok(true)));

        let p1 = array![[[1, 0], [0, 0]], [[0, 1], [1, 0]]].into_dyn();
        let r1 = order_n_unit_tensor(3, 1);
        let r2 = order_n_unit_tensor(3, 2);
        let r3 = order_n_unit_tensor(3, 3);

        assert!(matches!(tensor_restriction_of(&p1, &r1), Ok(false)));
        assert!(matches!(tensor_restriction_of(&p1, &r2), Ok(false)));
        assert!(matches!(tensor_restriction_of(&p1, &r3), Ok(true)));
    }
}
