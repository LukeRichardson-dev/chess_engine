use chess::ChessState;
use chester::neural_net::Cost;
use database::{init, load_to_memory, get_batch};
use engine::tools::Tools;
use ndarray::arr1;
use crate::{database::Instance, engine::{tools::Analysis, ai::Thod}};
use game::Game;

mod engine;
mod database;
mod chess;
mod game;
mod neural_net;

pub fn main() {
    let broken = "AkBAQEBABkABAUADwwFAAUBAQAUBQAFAQEBAQEBAQEBAQEDBQEBAwUDBQMVAQEBAQMFAQEDBwUBAQEDCwkDGQA==";
    Instance::from_str(broken, 0, 0);

    let conn = init(&"chess.db".to_owned());
    let conn = load_to_memory(&conn).unwrap();

    let mut thod = Thod::from_file("test.json").unwrap();
    // let mut thod = Thod::default();
    let test = get_batch(&conn, 5, 64);

    loop {
        let batch = get_batch(&conn, 0, 2000);

        for (idx, i) in batch.iter().enumerate() {
            thod.train_policy(&i.state(), i.winrate().0, 0.008);
            thod.train_value (&i.state(), i.winrate().0, 0.008);
            
            if idx % 100 == 0 {
                println!("{idx}/2000")
            }
        }
    
        thod.save("test.json").unwrap();

        let mut ploss = 0.0;
        let mut vloss = 0.0;
        for i in &test {
            let (p0, p1) = i.winrate();
            let pol = thod.policy(&i.state());
            let val = thod.value(&i.state());
            // println!("{pol} {val} {p0}");
            ploss += Cost::CrossEntropy.apply(&arr1(&[pol, 1.0 - pol]), &arr1(&[p0, p1])).sum();
            vloss += Cost::CrossEntropy.apply(&arr1(&[val, 1.0 - val]), &arr1(&[p0, p1])).sum();
        }
        ploss /= 64.0;
        vloss /= 64.0;
        println!("Policy loss -> {ploss}");
        println!("Value  loss -> {vloss}");
    }

    // println!("{:?}", batch);
}