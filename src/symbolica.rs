use std::{collections::HashMap, sync::Arc};

use symbolica::{
    domains::{
        integer::{Integer, IntegerRing},
        rational::Q,
    },
    poly::{PolyVariable, groebner::GroebnerBasis, polynomial::MultivariatePolynomial},
    symbol,
};

use crate::{
    isomorphism::{Delta, Tensor},
    restriction::delta_dimensions_3d,
    unit_tensor_delta,
};

pub fn int(integer: i64, field: &Q) -> symbolica::domains::rational::Fraction<IntegerRing> {
    field.to_element(Integer::new(integer), Integer::one(), false)
}

fn create_variable(var_name: String) -> PolyVariable {
    PolyVariable::Symbol(symbol!(&var_name))
}

#[derive(Default)]
struct VariableFactory {
    mapping: HashMap<String, usize>,
}

impl VariableFactory {
    pub fn get_or_create(&mut self, name: &str) -> usize {
        let var_count = self.variable_count();
        *self.mapping.entry(name.to_string()).or_insert(var_count)
    }

    pub fn variable_count(&self) -> usize {
        self.mapping.len()
    }

    pub fn variables(&self) -> Arc<Vec<PolyVariable>> {
        let mut mapping_tuples: Vec<(String, usize)> =
            self.mapping.iter().map(|(k, v)| (k.clone(), *v)).collect();
        mapping_tuples.sort_by_key(|(_, v)| *v);

        let mut vars: Vec<PolyVariable> = Vec::new();
        for (name, _) in mapping_tuples.iter() {
            vars.push(create_variable(name.to_string()));
        }
        Arc::new(vars)
    }

    pub fn zero_exponents(&self) -> Vec<u16> {
        vec![0u16; self.variable_count()]
    }
}

pub struct RestrictionSystem {
    source_tensor: Delta,
    target_tensor: Delta,
    variable_factory: VariableFactory,
}

impl RestrictionSystem {
    pub fn new(source_tensor: Delta, target_tensor: Delta) -> Self {
        Self {
            source_tensor,
            target_tensor,
            variable_factory: VariableFactory::default(),
        }
    }

    pub fn generate_system_of_polynomials(&mut self) -> Vec<MultivariatePolynomial<Q, u16>> {
        fn a_variable(i: usize, a: usize) -> String {
            format!("A_{i}_{a}")
        }

        fn b_variable(j: usize, b: usize) -> String {
            format!("B_{j}_{b}")
        }

        fn c_variable(k: usize, c: usize) -> String {
            format!("C_{k}_{c}")
        }

        let (source_dim, source_dim2, source_dim3) = delta_dimensions_3d(&self.source_tensor);
        let (target_dim, target_dim2, target_dim3) = delta_dimensions_3d(&self.target_tensor);

        // we assume that the tensors are cube (for now)
        let d_s = source_dim.max(source_dim2).max(source_dim3);
        let d_t = target_dim.max(target_dim2).max(target_dim3);

        let field = Q::new(IntegerRing::new());

        // create all variables
        for ijk in 0..d_s {
            for abc in 0..d_t {
                self.variable_factory.get_or_create(&a_variable(ijk, abc));
                self.variable_factory.get_or_create(&b_variable(ijk, abc));
                self.variable_factory.get_or_create(&c_variable(ijk, abc));
            }
        }

        let mut polynomials: Vec<MultivariatePolynomial<Q, u16>> = Vec::new();

        for i in 0..d_s {
            for j in 0..d_s {
                for k in 0..d_s {
                    // we ssume the tensors to be 0-1 valued (for now)
                    let s_value = if self.source_tensor.contains(&(i, j, k)) {
                        int(1, &field)
                    } else {
                        int(0, &field)
                    };

                    let mut terms = MultivariatePolynomial::new(
                        &field,
                        Some(self.variable_factory.variable_count()),
                        Arc::clone(&self.variable_factory.variables()),
                    );

                    // -S_{i,j,k}
                    if !s_value.is_zero() {
                        terms.append_monomial(-s_value, &self.variable_factory.zero_exponents());
                    }

                    for a in 0..d_t {
                        for b in 0..d_t {
                            for c in 0..d_t {
                                // we ssume the tensors to be 0-1 valued (for now)
                                let t_value = if self.target_tensor.contains(&(a, b, c)) {
                                    int(1, &field)
                                } else {
                                    int(0, &field)
                                };

                                if !t_value.is_zero() {
                                    let a_idx =
                                        self.variable_factory.get_or_create(&a_variable(i, a));
                                    let b_idx =
                                        self.variable_factory.get_or_create(&b_variable(j, b));
                                    let c_idx =
                                        self.variable_factory.get_or_create(&c_variable(k, c));

                                    let mut exponents: Vec<u16> =
                                        self.variable_factory.zero_exponents();
                                    exponents[a_idx] += 1;
                                    exponents[b_idx] += 1;
                                    exponents[c_idx] += 1;

                                    // A_{a,i} B_{b,j} C_{c,k} T_{a,b,c}
                                    terms.append_monomial(t_value, &exponents);
                                }
                            }
                        }
                    }

                    if terms.nterms() != 0 {
                        polynomials.push(terms);
                    }
                }
            }
        }

        polynomials
    }
}

pub struct GroebnerSolver;

impl GroebnerSolver {
    pub fn is_restriction_of(source_tensor: &Delta, target_tensor: &Delta) -> bool {
        let mut system = RestrictionSystem::new(source_tensor.clone(), target_tensor.clone());
        let polynomials = system.generate_system_of_polynomials();
        let groebner_basis = GroebnerBasis::new(&polynomials, false);
        let has_one = groebner_basis.system.iter().any(|p| p.is_one());
        !has_one
    }
}

pub struct TensorRankFinder;

impl TensorRankFinder {
    pub fn find_tensor_rank(tensor: &Tensor, max_rank: Option<usize>) -> usize {
        let mut rank = 1;
        loop {
            let unit_tensor = unit_tensor_delta(rank);

            if Some(rank) == max_rank {
                break rank;
            }

            if GroebnerSolver::is_restriction_of(tensor.delta(), &unit_tensor) {
                break rank;
            }

            rank += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::symbolica::GroebnerSolver;

    #[test]
    fn test_polynomial_system_generation() {
        let mut source_tensor = Delta::new();
        source_tensor.insert((0, 0, 0));
        source_tensor.insert((1, 1, 1));

        let mut target_tensor = Delta::new();
        target_tensor.insert((0, 0, 0));
        target_tensor.insert((1, 0, 0));
        target_tensor.insert((0, 1, 0));
        target_tensor.insert((0, 0, 1));
        target_tensor.insert((1, 1, 1));

        let (sd1, sd2, sd3) = delta_dimensions_3d(&source_tensor);
        let expected_nof_polynomials = sd1 * sd2 * sd3;

        let mut system = RestrictionSystem::new(source_tensor.clone(), target_tensor.clone());
        let polynomials = system.generate_system_of_polynomials();

        assert!(!polynomials.is_empty());
        assert!(polynomials.len() == expected_nof_polynomials);

        // Every polynomial should have as many terms as nonzero elements in the taget tensor.
        // Additionally, there should be +1 terms for as many polynomials as there are nonzero elements in the source tensor.
        let polynomials_nterms: Vec<usize> = polynomials.iter().map(|p| p.nterms()).collect();
        let s_len = source_tensor.len();
        let t_len = target_tensor.len();

        let count_t_len = polynomials_nterms.iter().filter(|&&x| x == t_len).count();
        let count_t_len_plus_one = polynomials_nterms
            .iter()
            .filter(|&&x| x == t_len + 1)
            .count();

        // t_len and t_len_plus_one should include all terms
        assert_eq!(count_t_len + count_t_len_plus_one, polynomials.len());

        assert_eq!(count_t_len, expected_nof_polynomials - s_len);
        assert_eq!(count_t_len_plus_one, s_len);

        // debug prints
        // for (i, poly) in polynomials.iter().enumerate() {
        //     println!("g{i}: {poly}");
        // }

        let restriction = GroebnerSolver::is_restriction_of(&source_tensor, &target_tensor);
        assert!(restriction);
    }

    #[test]
    fn test_tensor_reduces() {
        let mut m1 = Delta::new();
        m1.insert((0, 0, 0));
        let mut m2 = Delta::new();
        m2.insert((0, 0, 0));
        m2.insert((1, 1, 0));
        let mut m3 = Delta::new();
        m3.insert((0, 0, 0));
        m3.insert((1, 1, 0));
        m3.insert((2, 2, 0));

        assert!(GroebnerSolver::is_restriction_of(&m1, &m2));
        assert!(!GroebnerSolver::is_restriction_of(&m2, &m1));
        assert!(GroebnerSolver::is_restriction_of(&m2, &m3));
        assert!(!GroebnerSolver::is_restriction_of(&m3, &m1));
        assert!(!GroebnerSolver::is_restriction_of(&m3, &m2));
        assert!(GroebnerSolver::is_restriction_of(&m1, &m3));

        let mut p1 = Delta::new();
        p1.insert((0, 0, 0));
        p1.insert((1, 0, 1));
        p1.insert((1, 1, 0));

        let r1 = unit_tensor_delta(1);
        let r2 = unit_tensor_delta(2);
        let r3 = unit_tensor_delta(3);

        assert!(!GroebnerSolver::is_restriction_of(&p1, &r1));
        assert!(!GroebnerSolver::is_restriction_of(&p1, &r2));
        assert!(GroebnerSolver::is_restriction_of(&p1, &r3));
    }
}
