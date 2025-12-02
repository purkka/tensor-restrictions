use itertools::Itertools;
use tensor_restrictions::{
    isomorphism::{Delta, get_isomorphism_classes, print_tensor_isomorphism_classes},
    symbolica::{GroebnerSolver, TensorRankFinder},
    unit_tensor_delta,
};

fn main() {
    let mut p1 = Delta::new();
    p1.insert((0, 0, 0));
    p1.insert((1, 0, 1));
    p1.insert((1, 1, 0));

    let r1 = unit_tensor_delta(1);
    let r2 = unit_tensor_delta(2);
    let r3 = unit_tensor_delta(3);

    println!("{:?}", GroebnerSolver::is_restriction_of(&p1, &r1));
    println!("{:?}", GroebnerSolver::is_restriction_of(&p1, &r2));
    println!("{:?}", GroebnerSolver::is_restriction_of(&p1, &r3));

    print_tensor_isomorphism_classes(2);

    let dim = 2;
    let r2 = unit_tensor_delta(dim);

    for (&n, classes) in get_isomorphism_classes(dim)
        .iter()
        .sorted_by_key(|&(&n, _)| n)
    {
        if n == 0 {
            continue;
        }

        println!("nonzero elements: {n}");
        for (i, class) in classes.iter().enumerate() {
            if let Some(representative) = class.first() {
                let rank = TensorRankFinder::find_tensor_rank(representative, 3);
                println!("\tclass {}: rank <= {}", i + 1, rank);
                println!("\t\t{:?}", representative);
            }
        }
    }
}
