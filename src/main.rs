use itertools::Itertools;
use tensor_restrictions::{isomorphism::TensorIsomorphisms, symbolica::TensorRankFinder};

fn main() {
    let dim = 3;
    let tensor_isomorphisms = TensorIsomorphisms::new_square_sparse(dim);
    let sparse_latin_squares = TensorIsomorphisms::get_partial_latin_squares(&tensor_isomorphisms);

    println!("--- Sparse Latin square tensors ---");
    TensorIsomorphisms::print_tensor_isomorphism_classes(&sparse_latin_squares);

    println!();
    println!("--- Their ranks ---");
    for (&n, classes) in sparse_latin_squares
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
                let rank = TensorRankFinder::find_tensor_rank(representative, None);
                println!("\tclass {}: rank <= {}", i + 1, rank);
                println!("\t\t{:?}", representative);
            }
        }
    }
}
