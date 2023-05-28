use cozy_chess::{BoardBuilder};
use cozy_chess_types::bitboard;
use engine::EngineState;

use crate::{chess::{ChessState}, game::Game};

mod neural_net;
mod engine;
mod game;
mod chess;

fn main() {

    let mut state = EngineState::from_state(BoardBuilder::default().build().unwrap());
    println!("{:?}", state.rollout(1, (100, 100)));
    loop {
        for i in 0..200000 {
            let (w, s) = state.ucb_search_rollout(2.0, 5, None);
        }

        let (w, s) = state.ucb_search_rollout(2.0, 5, None);
        println!("{w:?} {s:?}");
        
        if let Some((data, ns)) = state.follow() {
            println!("{} {}", data.action, ns.state());
            state = ns;
        } else {
            break;
        }
        
    }

}