use groebner::{MonomialOrder, groebner_basis, polynomial::Polynomial};
use num_rational::BigRational;

pub struct GroebnerSolver;

impl GroebnerSolver {
    pub fn compute_groebner_basis(
        polynomials: Vec<Polynomial<BigRational>>,
        monomial_order: MonomialOrder,
    ) -> anyhow::Result<Vec<Polynomial<BigRational>>> {
        groebner_basis(polynomials, monomial_order, true).map_err(anyhow::Error::from)
    }

    pub fn has_one(basis: &[Polynomial<BigRational>]) -> bool {
        fn is_constant_one(polynomial: &Polynomial<BigRational>) -> bool {
            polynomial.terms.len() == 1 && {
                let term = &polynomial.terms[0];
                term.coefficient == BigRational::new(1.into(), 1.into())
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
                Term::new(
                    BigRational::new(1.into(), 1.into()),
                    Monomial::new(vec![2, 0]),
                ),
                Term::new(
                    BigRational::new(1.into(), 1.into()),
                    Monomial::new(vec![0, 2]),
                ),
                Term::new(
                    BigRational::new((-1).into(), 1.into()),
                    Monomial::new(vec![0, 0]),
                ),
            ],
            nvars,
            order,
        );

        // x0 - x1
        let g = Polynomial::new(
            vec![
                Term::new(
                    BigRational::new(1.into(), 1.into()),
                    Monomial::new(vec![1, 0]),
                ),
                Term::new(
                    BigRational::new((-1).into(), 1.into()),
                    Monomial::new(vec![0, 1]),
                ),
            ],
            nvars,
            order,
        );

        // x0 - x1, x1^2 - 1/2
        let expected = [
            Polynomial::new(
                vec![
                    Term::new(
                        BigRational::new(1.into(), 1.into()),
                        Monomial::new(vec![1, 0]),
                    ),
                    Term::new(
                        BigRational::new((-1).into(), 1.into()),
                        Monomial::new(vec![0, 1]),
                    ),
                ],
                nvars,
                order,
            ),
            Polynomial::new(
                vec![
                    Term::new(
                        BigRational::new(1.into(), 1.into()),
                        Monomial::new(vec![0, 2]),
                    ),
                    Term::new(
                        BigRational::new((-1).into(), 2.into()),
                        Monomial::new(vec![0, 0]),
                    ),
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
