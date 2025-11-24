use ndarray::array;
use tensor_restrictions::tensor_as_polynomial;

fn main() {
    let t_poly = tensor_as_polynomial(&array![[2, 0, 0], [0, 2, 0], [0, 0, 2]].into_dyn(), &[1, 2]);

    println!("{t_poly:?}");
}
