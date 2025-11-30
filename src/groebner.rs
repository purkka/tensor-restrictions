use groebner::{MonomialOrder, groebner_basis, polynomial::Polynomial};
use num_rational::BigRational;

// custom type and constructors for field elements
pub type FieldElement = BigRational;

pub fn int(integer: i64) -> FieldElement {
    FieldElement::from_integer(integer.into())
}

pub fn rational(numerator: i64, denominator: i64) -> FieldElement {
    FieldElement::new(numerator.into(), denominator.into())
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
}
