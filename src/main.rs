use std::default;

// use chester::engine::tools::AccumulativeAnalysis;
use engine::tools::{Analysis, Tools};
use rand::{seq::{IteratorRandom}, random};

use crate::{engine::{ai::Thod, tools::AccumulativeAnalysis}};
use chess::ChessState;

mod neural_net;
mod engine;
mod game;
mod chess;

struct RandTool;
impl Tools for RandTool {
    fn policy(&self, _: &ndarray::Array1<f32>) -> f32 {
        random()
    }

    fn value(&self, _: &ndarray::Array1<f32>) -> f32 {
        random()
    }
}

fn main() {

    let mut thod = Thod::from_file("network.json").unwrap();
    // let mut thod = Thod::default();
    // thod.save("test.json").unwrap();

    println!("STARTING MCTS");

    let mut analysis = AccumulativeAnalysis::from_position(ChessState::default()).unwrap();
    let default = ChessState::default().board.hash();
    let mut def = default;
    let mut cycle = 0;

    loop {
        cycle += 1;

        if let Some(()) = analysis.mcts(def, &thod, 2.0, 30) {
            for i in 0..500 {
                println!("Training 1000/{}", i * 10);
                for _ in 0..10 {
                    analysis.mcts(def, &thod, 2.0, 30);
                }
            }
        } else {
            def = analysis.random_hash();
            continue;
        }


    
        let a2 = analysis.try_get_analysis(&def).unwrap();
        
        let moves = a2.borrow_mut().moves();
        for (mv, p) in moves.iter().zip(a2.borrow_mut().p(&mut analysis)) {
            println!("{:?} -> {:?}, {p}", mv.from, mv.to);
        }
    
        for (i, (s, pos)) in analysis.training_data(50).enumerate() {
            println!("Training step: {i}");
    
            thod.train_policy(&pos, s, 0.08);
            thod.train_value(&pos, s, 0.06);
        }

        let (idx, _) = a2.borrow_mut().p(&mut analysis).iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap();
        let mv = moves[idx];
        let mut state = a2.borrow().state();
        state.board.play_unchecked(mv);

        println!("{}", state.board.to_string());

        thod.save(&format!("network_{}.json", cycle % 5)).unwrap();
        thod.save("network.json").unwrap();

        let r = random::<usize>() % 7;
        if 0 == r {
            analysis = AccumulativeAnalysis::from_position(state).unwrap();
        } else if r == 1 {
            def = analysis.random_hash();
        } else {
            def = a2.borrow().children()[idx];
        }
        // analysis = AccumulativeAnalysis::from_position(ChessState::default()).unwrap();

    }


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