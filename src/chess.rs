use cozy_chess::Board;
use cozy_chess_types::{BitBoard, Color, Piece, Move};
use ndarray::Array1;
use std::{hash::Hash};
use crate::game::Game;


#[derive(Debug, Default, Clone)]
pub struct ChessState { // TODO: Clone maybe not needed
    pub board: Board,
}

impl ChessState {
    pub fn moves(&self) -> Vec<Move> {
        let mut moves = vec![];
        self.board.generate_moves(|mvs| {
            mvs.into_iter().for_each(|x| moves.push(x));
            false
        });
        moves
    }
}

impl Hash for ChessState {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.board.hash().hash(state);
    }
}

impl Game for ChessState {
    type Children = Self;

    fn branch(&self) -> Vec<Self> {
        let mut children = vec![];
        self.board.generate_moves(|moves| {
            moves.into_iter().for_each(
                |x| {
                    let mut board = self.board.clone();
                    board.play_unchecked(x);
                    children.push(ChessState { board });
                }
            );
            false
        });
        children
    }

    fn state(&self) -> ndarray::Array1<f32> {
        let side = self.board.side_to_move();

        let mut state = vec![match side {
            Color::White => 0.0,
            Color::Black => 1.0,
        }];

        state.append(&mut bitboard_to_array(&self.board.checkers()).to_vec());
        state.append(&mut bitboard_to_array(&self.board.occupied()).to_vec());
        state.append(&mut bitboard_to_array(&self.board.pinned()).to_vec());

        state.append(&mut bitboard_to_array(&self.board.colors(Color::White)).to_vec());
        
        state.append(&mut bitboard_to_array(&self.board.colored_pieces(Color::White, Piece::King)).to_vec());
        state.append(&mut bitboard_to_array(&self.board.colored_pieces(Color::White, Piece::Queen)).to_vec());
        state.append(&mut bitboard_to_array(&self.board.colored_pieces(Color::White, Piece::Rook)).to_vec());
        state.append(&mut bitboard_to_array(&self.board.colored_pieces(Color::White, Piece::Bishop)).to_vec());
        state.append(&mut bitboard_to_array(&self.board.colored_pieces(Color::White, Piece::Knight)).to_vec());
        state.append(&mut bitboard_to_array(&self.board.colored_pieces(Color::White, Piece::Pawn)).to_vec());
        
        state.append(&mut bitboard_to_array(&self.board.colors(Color::Black)).to_vec());

        state.append(&mut bitboard_to_array(&self.board.colored_pieces(Color::Black, Piece::King)).to_vec());
        state.append(&mut bitboard_to_array(&self.board.colored_pieces(Color::Black, Piece::Queen)).to_vec());
        state.append(&mut bitboard_to_array(&self.board.colored_pieces(Color::Black, Piece::Rook)).to_vec());
        state.append(&mut bitboard_to_array(&self.board.colored_pieces(Color::Black, Piece::Bishop)).to_vec());
        state.append(&mut bitboard_to_array(&self.board.colored_pieces(Color::Black, Piece::Knight)).to_vec());
        state.append(&mut bitboard_to_array(&self.board.colored_pieces(Color::Black, Piece::Pawn)).to_vec());

        Array1::from_vec(state)
    }
}

pub fn bitboard_to_array(board: &BitBoard) -> [f32; 64] {
    let mut state = [0.0; 64];

    for i in 0..64 {
        if board.0 >> i & 1 == 1 {
            state[i] = 1.0;
        }
    }

    state
}