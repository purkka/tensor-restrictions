use ndarray::array;
use tensor_restrictions::tensor_reduces_to;

fn main() {
    let m1 = array![[1, 0], [0, 0]].into_dyn();
    let m2 = array![[1, 0], [0, 1]].into_dyn();
    let m3 = array![[1, 0, 0], [0, 1, 0], [0, 0, 1]].into_dyn();

    println!("{:?}", tensor_reduces_to(&m1, &m2));
    println!("{:?}", tensor_reduces_to(&m2, &m1));
    println!("{:?}", tensor_reduces_to(&m2, &m3));
    println!("{:?}", tensor_reduces_to(&m3, &m1));
    println!("{:?}", tensor_reduces_to(&m3, &m2));
    println!("{:?}", tensor_reduces_to(&m1, &m3));
}
