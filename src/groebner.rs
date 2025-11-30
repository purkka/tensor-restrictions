use std::collections::HashMap;

use groebner::{Field, Monomial, MonomialOrder, Term, groebner_basis, polynomial::Polynomial};
use num_rational::BigRational;

use crate::{isomorphism::Delta, restriction::delta_dimensions_3d};

// custom type and constructors for field elements
pub type FieldElement = BigRational;

pub fn int(integer: i64) -> FieldElement {
    FieldElement::from_integer(integer.into())
}

pub fn rational(numerator: i64, denominator: i64) -> FieldElement {
    FieldElement::new(numerator.into(), denominator.into())
}

fn constant_term(constant: FieldElement, nvars: usize) -> Term<FieldElement> {
    Term::new(constant, Monomial::new(vec![0; nvars]))
}

fn term(coefficient: FieldElement, exponents: Vec<u32>) -> Term<FieldElement> {
    Term::new(coefficient, Monomial::new(exponents))
}

fn a_variable(i: usize, a: usize) -> String {
    format!("A_{i}_{a}")
}

fn b_variable(j: usize, b: usize) -> String {
    format!("B_{j}_{b}")
}

fn c_variable(k: usize, c: usize) -> String {
    format!("C_{k}_{c}")
}

#[derive(Default)]
pub struct VariableFactory {
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
}

pub struct RestrictionSystem {
    source_tensor: Delta,
    target_tensor: Delta,
    variable_factory: VariableFactory,
    order: MonomialOrder,
}

impl RestrictionSystem {
    pub fn new(source_tensor: Delta, target_tensor: Delta) -> Self {
        Self {
            source_tensor,
            target_tensor,
            variable_factory: VariableFactory::default(),
            order: MonomialOrder::GRevLex,
        }
    }

    pub fn generate_system_of_polynomials(&mut self) -> Vec<Polynomial<FieldElement>> {
        let (source_dim, source_dim2, source_dim3) = delta_dimensions_3d(&self.target_tensor);
        let (target_dim, target_dim2, target_dim3) = delta_dimensions_3d(&self.target_tensor);

        // we assume that the tensors are cube (for now)
        let d_s = source_dim.max(source_dim2).max(source_dim3);
        let d_t = target_dim.max(target_dim2).max(target_dim3);

        let mut polynomials: Vec<Polynomial<FieldElement>> = Vec::new();

        // create all variables
        for ijk in 0..source_dim {
            for abc in 0..target_dim {
                self.variable_factory.get_or_create(&a_variable(ijk, abc));
                self.variable_factory.get_or_create(&b_variable(ijk, abc));
                self.variable_factory.get_or_create(&c_variable(ijk, abc));
            }
        }

        let nvars = self.variable_factory.variable_count();

        for i in 0..d_s {
            for j in 0..d_s {
                for k in 0..d_s {
                    // we ssume the tensors to be 0-1 valued (for now)
                    let s_value = if self.source_tensor.contains(&(i, j, k)) {
                        int(1)
                    } else {
                        int(0)
                    };

                    let mut terms: Vec<Term<FieldElement>> = Vec::new();

                    // -S_{i,j,k}
                    if !s_value.is_zero() {
                        terms.push(constant_term(-s_value, nvars));
                    }

                    for a in 0..d_t {
                        for b in 0..d_t {
                            for c in 0..d_t {
                                // we ssume the tensors to be 0-1 valued (for now)
                                let t_value = if self.target_tensor.contains(&(i, j, k)) {
                                    int(1)
                                } else {
                                    int(0)
                                };

                                if !t_value.is_zero() {
                                    let a_idx =
                                        self.variable_factory.get_or_create(&a_variable(i, a));
                                    let b_idx =
                                        self.variable_factory.get_or_create(&b_variable(j, b));
                                    let c_idx =
                                        self.variable_factory.get_or_create(&c_variable(k, c));

                                    let mut exponents: Vec<u32> = vec![0; nvars];
                                    exponents[a_idx] += 1;
                                    exponents[b_idx] += 1;
                                    exponents[c_idx] += 1;

                                    // A_{a,i} B_{b,j} C_{c,k} T_{a,b,c}
                                    terms.push(term(t_value, exponents));
                                }
                            }
                        }
                    }

                    if !terms.is_empty() {
                        polynomials.push(Polynomial {
                            terms,
                            nvars,
                            order: self.order,
                        });
                    }
                }
            }
        }

        polynomials
    }
}

pub struct GroebnerSolver;

impl GroebnerSolver {
    pub fn compute_groebner_basis(
        polynomials: Vec<Polynomial<FieldElement>>,
        monomial_order: MonomialOrder,
    ) -> anyhow::Result<Vec<Polynomial<FieldElement>>> {
        groebner_basis(polynomials, monomial_order, true).map_err(anyhow::Error::from)
    }

    pub fn has_one(basis: &[Polynomial<FieldElement>]) -> bool {
        fn is_constant_one(polynomial: &Polynomial<FieldElement>) -> bool {
            polynomial.terms.len() == 1 && {
                let term = &polynomial.terms[0];
                term.coefficient == FieldElement::new(1.into(), 1.into())
                    && term.monomial.degree() == 0
            }
        }

        basis.iter().any(is_constant_one)
    }
}

#[cfg(test)]
mod tests {
    use groebner::{Monomial, Term};

    use super::*;

    #[test]
    fn test_compute_groebner_basis() {
        // this is from the example of the `groebner` crate
        let nvars = 2;
        let order = MonomialOrder::Lex;

        // x0^2 + x1^2 - 1
        let f = Polynomial::new(
            vec![
                Term::new(int(1), Monomial::new(vec![2, 0])),
                Term::new(int(1), Monomial::new(vec![0, 2])),
                Term::new(int(-1), Monomial::new(vec![0, 0])),
            ],
            nvars,
            order,
        );

        // x0 - x1
        let g = Polynomial::new(
            vec![
                Term::new(int(1), Monomial::new(vec![1, 0])),
                Term::new(int(-1), Monomial::new(vec![0, 1])),
            ],
            nvars,
            order,
        );

        // x0 - x1, x1^2 - 1/2
        let expected = [
            Polynomial::new(
                vec![
                    Term::new(int(1), Monomial::new(vec![1, 0])),
                    Term::new(int(-1), Monomial::new(vec![0, 1])),
                ],
                nvars,
                order,
            ),
            Polynomial::new(
                vec![
                    Term::new(int(1), Monomial::new(vec![0, 2])),
                    Term::new(rational(-1, 2), Monomial::new(vec![0, 0])),
                ],
                nvars,
                order,
            ),
        ];

        let result = GroebnerSolver::compute_groebner_basis(vec![f, g], order);

        assert!(result.is_ok());
        assert!(!GroebnerSolver::has_one(result.as_ref().unwrap()));
        assert_eq!(result.unwrap(), expected);
    }

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

        let expected_nof_polynomials = target_tensor.len();

        let mut system = RestrictionSystem::new(source_tensor, target_tensor);
        let polynomials = system.generate_system_of_polynomials();

        assert!(!polynomials.is_empty());
        assert!(polynomials.len() == expected_nof_polynomials);
    }
}
