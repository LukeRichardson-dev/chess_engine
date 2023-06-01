use std::cmp::Ordering;

use crate::neural_net::Layer;
use cozy_chess::{Board, GameStatus};
use cozy_chess_types::{Color, Piece, Move};
use rand::prelude::*;

pub mod ai;
pub mod tools;

#[derive(Debug, Clone)]
pub struct Metadata {
    pub action: Move,
}

#[derive(Debug, Clone)]
enum Outcomes {
    Pending,
    Children(Vec<(Metadata, EngineState)>)
}

pub struct Engine {
    policy: Layer,
    value: Layer,
    top_state: EngineState,
}

#[derive(Debug, Clone)]
pub struct EngineState {
    state: Board,
    n: usize,
    b: usize,
    w: usize,
    outcome: GameStatus,
    outcomes: Outcomes,
}

impl EngineState {
    pub fn from_state(state: Board) -> Self {
        Self {
            outcome: state.status(),
            state,
            n: 0,
            b: 0,
            w: 0,
            outcomes: Outcomes::Pending,
        }
    }

    pub fn ucb(&self, n: usize, k: f64) -> f64 {
        if self.n == 0 { f64::INFINITY }
        else { self.score() + ((k * (n as f64).ln()) / self.n as f64).sqrt() }
    }

    pub fn score(&self) -> f64 {
        (match self.state.side_to_move() {
            Color::Black => self.w,
            Color::White => self.b,
        }  as f64) / self.n as f64
    }

    pub fn max_ucb(&mut self, k: f64) -> Option<&mut (Metadata, EngineState)> {
        match (&mut self.outcomes, self.n) {
            (Outcomes::Pending, _) | (_, 0) => None,
            (Outcomes::Children(c), _) => {
                Some(c.iter_mut()
                    .max_by(
                        |(_, a), (_, b)| a.ucb(self.n, k)
                            .partial_cmp(&b.ucb(self.n, k))
                            .unwrap()
                    )   
                    .unwrap()
                )
            }
        }
    }

    pub fn ucb_search_rollout(&mut self, k: f64, depth: usize, mut rel: Option<(u32, u32)>) -> (Color, GameStatus) {
        if let None = rel {
            rel = Some(self.simple_score());
        }
        
        match &self.outcome {
            GameStatus::Ongoing => (),
            o => return (self.state.side_to_move(), o.clone()),
        }

        self.n += 1;
        let o = if let Some(child) = self.max_ucb(k) {
            child.1.ucb_search_rollout(k, depth, rel)
        } else {
            self.rollout(depth, rel.unwrap())
        };
        
        match &o {
            (_, GameStatus::Ongoing) => panic!(),
            (Color::White, GameStatus::Won) => {
                self.w += 1;
            } 
            (Color::Black, GameStatus::Won) => {
                self.b += 1;
            }
            _ => (),
        }

        o
    }

    pub fn rollout(&mut self, depth: usize, rel: (u32, u32)) -> (Color, GameStatus) {
        match &self.outcome {
            GameStatus::Ongoing => (),
            o => return (self.state.side_to_move(), o.clone()),
        }

        if depth == 0 {
            return match self.predict(rel) {
                Some(c) => (c, GameStatus::Won),
                None => (Color::White, GameStatus::Drawn),
            }
        }

        let children = self.obtain_the_children();
        let (_, child): &mut (_, EngineState) = children.choose_mut(&mut rand::rngs::OsRng).unwrap();

        child.rollout(depth - 1, rel)
    }

    pub fn obtain_the_children(&mut self) -> &mut Vec<(Metadata, EngineState)> {
        let out = &mut self.outcomes;
        let children: &mut Vec<(Metadata, EngineState)> = match out {
            Outcomes::Pending => {
                let mut c: Vec<(Metadata, EngineState)> = Vec::default();
                self.state
                    .generate_moves(
                        |moves| {
                            c.append(&mut moves.into_iter()
                                .map(|x| (   
                                    Metadata {action: x},
                                    EngineState::from_state({
                                        let mut state = self.state.clone();
                                        state.play(x);
                                        state
                                    })
                                ))
                                .collect());
                            false
                        }
                    );
                
                *out = Outcomes::Children(c);
                    
                if let Outcomes::Children(c) = out {
                    c
                } else {
                    panic!("This is literally unreachable. Why panic?");
                }
            },
            Outcomes::Children(c) => c,
        };

        children
    }

    pub fn follow(&mut self) -> Option<(Metadata, EngineState)> {
        match self.max_ucb(0.0) {
            Some(x) => Some(x.clone()),
            _ => None,
        }
    }

    pub fn state(&self) -> &Board {
        &self.state
    }


    fn predict(&self, rel: (u32, u32)) -> Option<Color> {
        let (mut b, mut w) = (0, 0);
        if self.state.colored_pieces(Color::White, Piece::Queen).0 != 0 {
            w += 9;
        }
        w += self.state.colored_pieces(Color::White, Piece::Rook).len() * 5;
        w += self.state.colored_pieces(Color::White, Piece::Bishop).len() * 4;
        w += self.state.colored_pieces(Color::White, Piece::Knight).len() * 4;
        w += self.state.colored_pieces(Color::White, Piece::Pawn).len();

        if self.state.colored_pieces(Color::Black, Piece::Queen).0 != 0 {
            b += 9;
        }
        b += self.state.colored_pieces(Color::Black, Piece::Rook).len() * 5;
        b += self.state.colored_pieces(Color::Black, Piece::Bishop).len() * 4;
        b += self.state.colored_pieces(Color::Black, Piece::Knight).len() * 4;
        b += self.state.colored_pieces(Color::Black, Piece::Pawn).len();
        
        match (rel.1 - b).cmp(&(rel.0 - w)) {
            Ordering::Less => Some(Color::Black),
            Ordering::Greater => Some(Color::White),
            Ordering::Equal => None,
        }
    }

    fn simple_score(&self) -> (u32, u32) {
        let (mut b, mut w) = (0, 0);
        if self.state.colored_pieces(Color::White, Piece::Queen).0 != 0 {
            w += 9;
        }
        w += self.state.colored_pieces(Color::White, Piece::Rook).len() * 5;
        w += self.state.colored_pieces(Color::White, Piece::Bishop).len() * 4;
        w += self.state.colored_pieces(Color::White, Piece::Knight).len() * 4;
        w += self.state.colored_pieces(Color::White, Piece::Pawn).len();

        if self.state.colored_pieces(Color::Black, Piece::Queen).0 != 0 {
            b += 9;
        }
        b += self.state.colored_pieces(Color::Black, Piece::Rook).len() * 5;
        b += self.state.colored_pieces(Color::Black, Piece::Bishop).len() * 4;
        b += self.state.colored_pieces(Color::Black, Piece::Knight).len() * 4;
        b += self.state.colored_pieces(Color::Black, Piece::Pawn).len();

        (w, b)
    }
}