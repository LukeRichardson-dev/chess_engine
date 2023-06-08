// use chester::engine::tools::AccumulativeAnalysis;
use engine::tools::{Analysis, Tools};
use rand::random;

use crate::{engine::{ai::Thod, tools::AccumulativeAnalysis}};
use chess::ChessState;

mod neural_net;
mod engine;
mod game;
mod chess;

struct RandTool;
impl Tools for RandTool {
    fn policy(&self, _: &ndarray::Array1<f64>) -> f64 {
        random()
    }

    fn value(&self, _: &ndarray::Array1<f64>) -> f64 {
        random()
    }
}

fn main() {

    let thod = Thod::from_file("test.json").unwrap();
    // let mut thod = Thod::default();
    // thod.save("test.json").unwrap();

    println!("STARTING MCTS");

    let mut analysis = AccumulativeAnalysis::from_position(ChessState::default()).unwrap();
    analysis.mcts(ChessState::default().board.hash(), &thod, 2.0, 3);

    // let mut analysis = &mut Analysis::from_state(ChessState::default());
    // println!("{:?}", analysis.probabilities(&thod));
    // loop {
    //     for _ in 0..20 {
    //         analysis.simulate(&thod, 3);
    //     }
    //     analysis.eval();
    //     analysis.advance();

    //     analysis.show_board();
    //     println!()
    // }
    // analysis.eval();
    
    // let mut state = EngineState::from_state(BoardBuilder::default().build().unwrap());
    // println!("{:?}", state.rollout(1, (100, 100)));
    // loop {
    //     for i in 0..200000 {
    //         let (w, s) = state.ucb_search_rollout(2.0, 5, None);
    //     }

    //     let (w, s) = state.ucb_search_rollout(2.0, 5, None);
    //     println!("{w:?} {s:?}");
        
    //     if let Some((data, ns)) = state.follow() {
    //         println!("{} {}", data.action, ns.state());
    //         state = ns;
    //     } else {
    //         break;
    //     }
        
    // }

}