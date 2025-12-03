use itertools::Itertools;
use tensor_restrictions::{isomorphism::TensorIsomorphisms, symbolica::TensorRankFinder};

fn main() {
    let dim = 2;
    let tensor_isomorphisms = TensorIsomorphisms::new(dim);

    tensor_isomorphisms.print_tensor_isomorphism_classes();

    for (&n, classes) in tensor_isomorphisms
        .get_isomorphism_classes()
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
