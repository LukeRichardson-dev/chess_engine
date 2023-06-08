use std::{cell::{RefCell, RefMut}, collections::{HashMap, VecDeque}, rc::Rc};

use anyhow::Result;
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
    positions: HashMap<u64, Rc<RefCell<PositionAnalysis>>>, // pain
    temp_pool: HashMap<u64, ChessState>
}

impl AccumulativeAnalysis {
    fn with_positions(positions: HashMap<u64, Rc<RefCell<PositionAnalysis>>>) -> Self {
        Self { positions, temp_pool: HashMap::new() }
    }

    pub fn from_position(state: ChessState) -> Result<Self> {
        let mut positions = HashMap::with_capacity(1);
        
        let hash = state.board.hash();
        let (analysis, children) = PositionAnalysis::from_state(state);

        positions.insert(hash, Rc::new(RefCell::new(analysis)));

        let mut buf = Self::with_positions(positions);
        children.into_iter().for_each(|x| buf.add_to_pool(x));

        Ok(buf)
    }

    fn add_to_pool(&mut self, (hash, state): (u64, ChessState)) {
        if self.positions.contains_key(&hash) { return; }
        if self.temp_pool.contains_key(&hash) { return; }

        self.temp_pool.insert(hash, state);
    }

    fn try_get_analysis<'a>(&'a mut self, hash: &u64) -> Option<Rc<RefCell<PositionAnalysis>>> {
        if let Some(analysis) = self.positions.get(hash) { return Some(analysis.clone()); }
        else if let Some(state) = self.temp_pool.remove(hash) { 
            let (analysis, children) = PositionAnalysis::from_state(state);
            children.into_iter().for_each(|x| self.add_to_pool(x));

            let aref = Rc::new(RefCell::new(analysis));
            self.positions.insert(*hash, aref.clone());
            // .expect("State should not have a pre-existing analysis."); // TODO: More elegant error / more flexible architecture

            return Some(aref);
        }

        None
    } 

    pub fn mcts<T: Tools>(&mut self, hash: u64, tools: &T, c: f64, depth: usize) {
        self.try_get_analysis(&hash).unwrap().borrow_mut().mcts(self, tools, c, depth);
    }

}

pub struct PositionAnalysis {
    state: ChessState,
    encoding: Array1<f64>,
    visits: usize,
    wins: f64,
    hash: u64,
    policy: Option<f64>,
    children: Vec<u64>,
    value: Option<f64>,
}

impl PositionAnalysis {
    fn from_state(state: ChessState) -> (Self, Vec<(u64, ChessState)>) {

        let children: Vec<_> = state.branch().into_iter().map(|x| (x.board.hash(), x)).collect();

        (Self {
            encoding: state.state(),
            visits: 0,
            wins: 0.0,
            hash: state.board.hash(),
            policy: None,
            value: None,
            children: children.iter().map(|x| x.0).collect(),
            state,
        }, children)
    }

    pub fn exploit(&self) -> f64 {
        match self.state.board.side_to_move() {
            Color::Black => 1.0 - (self.wins as f64 / self.visits as f64),
            Color::White => self.wins as f64 / self.visits as f64,
        }
    }

    fn explore(&self, n: usize, c: f64) -> f64 {
        (c * ((n as f64).ln() / self.visits as f64)).sqrt()
    }

    pub fn ucb(&self, n: usize, c: f64) -> f64 {
        if !self.visited() { return 100.0 }
        self.exploit() + self.explore(n, c)
    }

    pub fn policy<T: Tools>(&mut self, tools: &T) -> f64 {
        if let Some(p) = self.policy {
            p
        } else {
            let p = tools.policy(&self.encoding);
            self.policy = Some(p);
            p
        }
    }

    pub fn value<T: Tools>(&mut self, tools: &T) -> f64 {
        if let Some(p) = self.policy {
            p
        } else {
            let p = tools.policy(&self.encoding);
            self.policy = Some(p);
            p
        }
    }

    pub fn visited(&self) -> bool {
        self.visits != 0
    }

    pub fn search<T: Tools>(&self, cache: &mut AccumulativeAnalysis, tools: &T, c: f64, q: &mut VecDeque<Rc<RefCell<PositionAnalysis>>>) -> Rc<RefCell<PositionAnalysis>> {
        
        let analysis = self.children.iter()
            .map(|x| cache.try_get_analysis(x).unwrap())
            .map(|x| ({x.borrow().ucb(self.visits, c)}, x.clone()))
            .max_by(|x, y| x.0.partial_cmp(&y.0).unwrap());

        if let Some(a) = analysis {
            q.push_back(a.1.clone());
            if !a.1.borrow().visited() {
                return a.1;
            } else {
                return a.1.borrow().search(cache, tools, c, q);
            }
        } 
        
        let a = cache.try_get_analysis(&self.hash).unwrap();
        q.push_back(a.clone());
        a
    }

    pub fn rollout<T: Tools>(&mut self, cache: &mut AccumulativeAnalysis, tools: &T, depth: usize) -> f64 {
        if depth == 0 {
            let score = self.value(tools);
            self.increment(score);
            return score;
        }

        if self.children.len() == 0 {
            match self.state.board.status() {
                cozy_chess::GameStatus::Won => return match self.state.board.side_to_move() {
                    Color::White => 0.0,
                    Color::Black => 1.0,
                },
                cozy_chess::GameStatus::Drawn => return 0.0,
                _ => panic!(),
            }
        }

        if self.visits != 0 {
            let analysis = self.children.iter()
                .map(|x| cache.try_get_analysis(x).unwrap())
                .filter(|x| x.borrow().visited())
                .map(|x| (x.borrow_mut().exploit(), x.clone()))
                .max_by(|x, y| x.0.partial_cmp(&y.0).unwrap())
                .unwrap().1;

            return analysis.borrow_mut().rollout(cache, tools, depth - 1);
        }

        let analysis = self.children.iter()
            .map(|x| cache.try_get_analysis(x).unwrap())
            .map(|x| (x.borrow_mut().policy(tools), x.clone()))
            .max_by(|x, y| x.0.partial_cmp(&y.0).unwrap())
            .unwrap().1;

        let x = analysis.borrow_mut().rollout(cache, tools, depth - 1); 
        x
    }

    fn side(&self) -> Color {
        self.state.board.side_to_move()
    }

    pub fn increment(&mut self, score: f64) {
        self.visits += 1;
        self.wins += match self.side() {
            Color::White => score,
            Color::Black => 1.0 - score,
        }
    }

    pub fn mcts<T: Tools>(&mut self, cache: &mut AccumulativeAnalysis, tools: &T, c: f64, depth: usize) {
        let mut q = VecDeque::default();
        let analysis = self.search(cache, tools, c, &mut q);
        let score = analysis.borrow_mut().rollout(cache, tools, depth);
        
        q.drain(..).for_each(|x| x.borrow_mut().increment(score));
    }

}