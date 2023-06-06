use std::cell::{RefCell, RefMut};

use cozy_chess::Board;
use cozy_chess_types::{Color, Square};
use ndarray::Array1;

use crate::{chess::ChessState, game::Game};

pub trait Tools {
    fn policy(&self, state: &Array1<f64>) -> f64;
    fn value(&self, state: &Array1<f64>) -> f64;
}

enum GamePrediction {
    Win,
    Draw,
    Loss,
}

#[derive(Debug)]
struct Candidate(CandidateState);

#[derive(Debug)]
enum CandidateState {
    Waiting(ChessState),
    Analysis(Box<RefCell<Analysis>>),
}

impl Candidate {
    pub fn from_board(board: Board) -> Self {
        Self(CandidateState::Waiting(ChessState { board }))
    }

    pub fn analysis<'a>(&'a mut self) -> RefMut<'a, Analysis> {
        if let Candidate(CandidateState::Analysis(a)) = self {
            return a.borrow_mut();
        }

        let state: ChessState;
        if let Candidate(CandidateState::Waiting(s)) = self {
            state = s.clone();
        } else { panic!("ewfhiufe") }

        self.0 = CandidateState::Analysis(Box::new(RefCell::new(Analysis::from_state(state))));
        self.analysis()
    }

    pub fn take(self) -> Analysis {
        if let Candidate(CandidateState::Analysis(a)) = self {
            a.into_inner()
        } else { panic!("ewfhiufe") }
    }
}

#[derive(Debug)]
pub struct Analysis {
    pub state: ChessState,
    encoding: Array1<f64>,
    visits: usize,
    wins: f64,
    children: Vec<Candidate>,
}

impl Analysis {
    pub fn from_state(state: ChessState) -> Self {
        Self {
            encoding: state.state(),
            children: state.branch().into_iter().map(|x| Candidate::from_board(x.board)).collect(),
            visits: 0,
            wins: 0.0,
            state,
        }
    }

    pub fn ucb(&self, n: usize, c: f64) -> f64 {
        if self.visits == 0 { return f64::INFINITY }
        self.exploit() + self.explore(n, c)
    }

    fn exploit(&self) -> f64 {
        match self.state.board.side_to_move() {
            Color::Black => 1.0 - (self.wins as f64 / self.visits as f64),
            Color::White => self.wins as f64 / self.visits as f64,
        }
    }

    fn explore(&self, n: usize, c: f64) -> f64 {
        (c * ((n as f64).ln() / self.visits as f64)).sqrt()
    }

    pub fn argmax<'a>(&'a mut self, c: f64) -> RefMut<'a, Analysis> {
        self.children.iter_mut()
            .map(|x| x.analysis())
            .map(|x| (x.ucb(self.visits, c), x))
            .max_by(|a, b| a.0.partial_cmp(&b.0).or(Some(std::cmp::Ordering::Equal)).unwrap())
            .unwrap().1
    }

    pub fn advance(&mut self) {
        let (idx, _) = self.children.iter_mut()
            .map(|x| x.analysis())
            .map(|x| (x.ucb(self.visits, 0.0), x))
            .enumerate()
            .max_by(|(_, a), (_, b)| a.0.partial_cmp(&b.0).or(Some(std::cmp::Ordering::Equal)).unwrap())
            .unwrap();

        *self = self.children.remove(idx).take();
    }



    pub fn eval(&mut self) {
        for (child, mv) in self.children.iter_mut().zip(self.state.moves()) {
            let analysis = child.analysis();
            let score = analysis.ucb(self.visits, 0.0);
            println!("{:?} -> {:?}| score: {score}", mv.from, mv.to);
        }
    }

    pub fn show_board(&self) {
        for (i, square) in Square::ALL.into_iter().enumerate() {
            if let Some(peice) =  self.state.board.piece_on(square) {
                print!(" {}{} ", peice.to_string(), match self.state.board.color_on(square).unwrap() {
                    Color::White => 'w',
                    Color::Black => 'b',
                });
            } else { print!("    ") }
            if i & 0x7 == 7 { println!() }
        }

        println!("{}", self.state.board.to_string());
    }

    pub fn simulate<T: Tools>(&mut self, tools: &T, depth: usize) -> f64 {
        let res = if self.visits == 0 {
            self.rollout(tools, depth)
        } else {
            self.argmax(2.0).simulate(tools, depth)
        };

        self.visits += 1;
        self.wins += res;

        res
    }

    pub fn rollout<T: Tools>(&mut self, tools: &T, depth: usize) -> f64 {
        match self.state.board.status() {
            cozy_chess::GameStatus::Won => return match self.state.board.side_to_move() {
                Color::White => 0.0,
                Color::Black => 1.0,
            },
            cozy_chess::GameStatus::Drawn => return 0.0,
            _ => (),
        }
        if depth == 0 { 
            let v = tools.value(&self.encoding);
            return v;
        }

        let binding = self.probabilities(tools);
        let (index, _p) = match self.state.board.side_to_move() {
            Color::White => binding.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap(),
            Color::Black => binding.iter().enumerate().min_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap(),
        };
        self.children[index].analysis().rollout(tools, depth - 1)
    }

    pub fn probabilities<T: Tools>(&mut self, tools: &T) -> Vec<f64> {
        self.children.iter_mut().map(|x| tools.policy(&x.analysis().encoding)).collect()
    }
}

pub struct AccumulativeAnalysis {
    positions: HashMap<Board, PositionAnalysis>,
}

impl AccumulativeAnalysis {
    pub fn try_get_game(&mut self, gt)
}

pub struct PositionAnalysis {
    state: ChessState,
    encoding: Array1<f64>,
    children: Vec<Board>,
    visits: usize,
    wins: f64,
    policy: Option<f64>,
    value: Option<f64>,
}

impl PositionAnalysis {
    fn from_state(state: ChessState) -> Self {
        Self {
            encoding: state.state(),
            visits: 0,
            wins: 0.0,
            policy: None,
            value: None,
            state,
        }
    }

    fn exploit(&self) -> f64 {
        match self.state.board.side_to_move() {
            Color::Black => 1.0 - (self.wins as f64 / self.visits as f64),
            Color::White => self.wins as f64 / self.visits as f64,
        }
    }

    fn explore(&self, n: usize, c: f64) -> f64 {
        (c * ((n as f64).ln() / self.visits as f64)).sqrt()
    }

    pub fn ucb(&self, n usize, c: f64) -> f64 {
        self.exploit() + self.explore(n, c)
    }

    pub fn policy<T: Tools>(&mut self, tools: &T) -> f64 {
        if let Some(p) = self.policy {
            p
        } else {
            let p = tools.policy(self.encoding);
            self.policy = Some(p);
            p
        }
    }

    pub fn value<T: Tools>(&mut self, tools: &T) -> f64 {
        if let Some(p) = self.policy {
            p
        } else {
            let p = tools.policy(self.encoding);
            self.policy = Some(p);
            p
        }
    }

    pub fn search<'a>(&mut self, cache: &mut AccumulativeAnalysis) -> &'a mut PositionAnalysis {
        
    }
}