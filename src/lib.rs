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

fn get_flat_index(
    axis: usize,
    idx_s: usize,
    idx_t: usize,
    dims_s: &[usize],
    dims_t: &[usize],
) -> usize {
    let mut offset = 0;
    for a in 0..axis {
        offset += dims_s[a] * dims_t[a];
    }
    offset + idx_s * dims_t[axis] + idx_t
}

/// Checks whether a tensor `tensor_s` reduces to another tensor `tensor_t`
/// via a Groebner basis computation.
///
/// Returns an error if the orders of the tensors do not match.
pub fn tensor_reduces_to(tensor_s: &ArrayD<u32>, tensor_t: &ArrayD<u32>) -> anyhow::Result<bool> {
    let dimensions_s = tensor_s.shape().to_vec();
    let dimensions_t = tensor_t.shape().to_vec();

    if dimensions_s.len() != dimensions_t.len() {
        anyhow::bail!("Tensor orders do not match")
    }

    let variable_map = make_poly_vars(&dimensions_s, &dimensions_t);

    // big prime that fits into u32
    let field = Zp::new(1_000_000_007);

    let mut polynomial_system: Vec<MultivariatePolynomial<_, u32>> = Vec::new();

    // loop through elements of `tensor_s`
    for index_s in dimensions_s.iter().map(|&d| 0..d).multi_cartesian_product() {
        let mut polynomial =
            MultivariatePolynomial::new(&field, Some(variable_map.len()), variable_map.clone());

        // populate with terms from `tensor_t`
        for index_t in tensor_t
            .shape()
            .iter()
            .map(|&d| 0..d)
            .multi_cartesian_product()
        {
            let value_t = tensor_t[&index_t[..]];
            // only handle nonzero elements
            if value_t == 0 {
                continue;
            }

            let mut exponents = vec![0u32; variable_map.len()];
            for (axis, (&idx_s, &idx_t)) in index_s.iter().zip(index_t.iter()).enumerate() {
                let var_index = get_flat_index(axis, idx_s, idx_t, &dimensions_s, &dimensions_t);
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
