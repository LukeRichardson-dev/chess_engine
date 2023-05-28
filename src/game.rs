use std::hash::Hash;

use ndarray::Array1;

pub trait Game: Hash {
    type Children;

    fn branch(&self) -> Vec<Self::Children>;
    fn state(&self) -> Array1<f64>;
}