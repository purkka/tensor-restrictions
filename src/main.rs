use itertools::Itertools;
use ndarray::array;
use tensor_restrictions::{
    isomorphism::{get_isomorphism_classes, print_tensor_isomorphism_classes},
    order_n_unit_tensor,
    restriction::{tensor_restriction_of, tensor_restriction_supports},
    unit_tensor_delta,
};

fn main() {
    let m1 = array![[1, 0], [0, 0]].into_dyn();
    let m2 = array![[1, 0], [0, 1]].into_dyn();
    let m3 = array![[1, 0, 0], [0, 1, 0], [0, 0, 1]].into_dyn();

    println!("{:?}", tensor_restriction_of(&m1, &m2));
    println!("{:?}", tensor_restriction_of(&m2, &m1));
    println!("{:?}", tensor_restriction_of(&m2, &m3));
    println!("{:?}", tensor_restriction_of(&m3, &m1));
    println!("{:?}", tensor_restriction_of(&m3, &m2));
    println!("{:?}", tensor_restriction_of(&m1, &m3));

    let p1 = array![[[1, 0], [0, 0]], [[0, 1], [1, 0]]].into_dyn();
    let r1 = order_n_unit_tensor(3, 1);
    let r2 = order_n_unit_tensor(3, 2);
    let r3 = order_n_unit_tensor(3, 3);

    println!("{:?}", tensor_restriction_of(&p1, &r1));
    println!("{:?}", tensor_restriction_of(&p1, &r2));
    println!("{:?}", tensor_restriction_of(&p1, &r3));

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
                let restricts = tensor_restriction_supports(representative, &r2).unwrap();
                println!("\tclass {}: restricts {}", i + 1, restricts);
                println!("\t\t{:?}", representative);
            }
        }
    }
}
