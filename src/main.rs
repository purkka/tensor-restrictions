use symbolica::{atom::Atom, symbol};

fn make_symbolic_variable(v: &str, exp: usize, z: usize) -> Atom {
    Atom::from(symbol!(&format!("{v}_p{exp}_{z}")))
}

fn main() {
    let var = make_symbolic_variable("v", 3, 2);

    println!("{var:?}");
}
