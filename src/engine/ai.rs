use std::{fs::File, io::{Read, Write}};

use ndarray::{arr1, Array1};
use serde::{Deserialize, Serialize};
use anyhow::Result;

use crate::neural_net::{Layer, Activation};

use super::tools::Tools;

#[derive(Debug, Deserialize, Serialize)]
pub struct Thod {
    policy: Layer,
    value: Layer,
}

impl Thod {
    pub fn from_shape(pol: Vec<usize>, val: Vec<usize>) -> Self {
        let mut policy = Layer::random(1089, pol[0], Activation::LeakyReLU);
        let mut value  = Layer::random(1089, val[0], Activation::LeakyReLU);

        for i in pol.iter().skip(1) {
            policy.add_random_layer(*i, Activation::LeakyReLU);
        }

        for i in val.iter().skip(1) {
            value.add_random_layer(*i, Activation::LeakyReLU);
        }

        policy.add_random_layer(2, Activation::Softmax);
        value.add_random_layer(2, Activation::Softmax);

        value.scale(0.4);
        policy.scale(0.3);

        Self { policy, value }
    }

    pub fn from_file(path: &str) -> Result<Self> {
        let mut file = File::open(path)?;
        let mut buf = String::default();
        file.read_to_string(&mut buf)?;
        Ok(serde_json::from_str(&buf)?)
    }

    pub fn save(&self, path: &str) -> Result<()> {
        let data = serde_json::to_vec_pretty(self)?;
        let mut file = File::create(path)?;
        file.write(&data)?;

        Ok(())
    }

    pub fn train_policy(&mut self, state: &Array1<f64>, outcome: f64, lr: f64) {
        self.policy.train(state, &arr1(&[outcome, 1.0 - outcome]), &crate::neural_net::Cost::CrossEntropy, lr);
    }

    pub fn train_value(&mut self, state: &Array1<f64>, outcome: f64, lr: f64) {
        self.value.train(state, &arr1(&[outcome, 1.0 - outcome]), &crate::neural_net::Cost::CrossEntropy, lr);
    }
}

impl Default for Thod {
    fn default() -> Self {
        Self::from_shape(vec![500, 250, 100], vec![750, 500, 250, 100, 10])
    }
}

impl Tools for Thod {
    fn policy(&self, state: &ndarray::Array1<f64>) -> f64 {
        let r = self.policy.predict(state);
        return r[0];
    }

    fn value(&self, state: &ndarray::Array1<f64>) -> f64 {
        let r = self.value.predict(state);
        return r[0];
    }
}