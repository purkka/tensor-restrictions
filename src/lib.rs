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

fn create_variable<A, B, C>(a: A, b: B, c: C) -> PolyVariable
where
    A: Display,
    B: Display,
    C: Display,
{
    PolyVariable::Symbol(symbol!(&format!("v_{a}_{b}_{c}")))
}

fn make_poly_vars(dim_s: &[usize], dim_t: &[usize]) -> Arc<Vec<PolyVariable>> {
    let mut vars = Vec::new();
    for (axis, (&ds, &dt)) in dim_s.iter().zip(dim_t.iter()).enumerate() {
        for (s_idx, t_idx) in (0..ds).cartesian_product(0..dt) {
            vars.push(create_variable(axis, s_idx, t_idx));
        }
    }
    Arc::new(vars)
}

fn get_flat_index(
    axis: usize,
    s_idx: usize,
    t_idx: usize,
    dims_s: &[usize],
    dims_t: &[usize],
) -> usize {
    let mut offset = 0;
    for a in 0..axis {
        offset += dims_s[a] * dims_t[a];
    }
    offset + s_idx * dims_t[axis] + t_idx
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

    let var_map = make_poly_vars(&dimensions_s, &dimensions_t);

    // big prime that fits into u32
    let field = Zp::new(1_000_000_007);

    // build polynomial system
    let mut poly_system: Vec<MultivariatePolynomial<_, u32>> = Vec::new();

    for index_set in dimensions_s.iter().map(|&d| 0..d).multi_cartesian_product() {
        let mut p = MultivariatePolynomial::new(&field, Some(var_map.len()), var_map.clone());

        for idxs in tensor_t
            .shape()
            .iter()
            .map(|&d| 0..d)
            .multi_cartesian_product()
        {
            let val = tensor_t[&idxs[..]];
            // only handle nonzero elements
            if val == 0 {
                continue;
            }

            let mut exps = vec![0u32; var_map.len()];
            for (axis, (&s_i, &t_i)) in index_set.iter().zip(idxs.iter()).enumerate() {
                let var_index = get_flat_index(axis, s_i, t_i, &dimensions_s, &dimensions_t);
                exps[var_index] = 1;
            }

            p.append_monomial(field.to_element(val), &exps);
        }

        let s_val = tensor_s[&index_set[..]];

        let zero_exps = vec![0u32; var_map.len()];
        p.append_monomial(
            field.mul(field.neg(&field.to_element(1)), field.to_element(s_val)),
            &zero_exps,
        );

        poly_system.push(p);
    }

    let groebner_basis = GroebnerBasis::new(&poly_system, false);
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
