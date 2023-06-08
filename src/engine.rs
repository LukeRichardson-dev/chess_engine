use std::cmp::Ordering;

use crate::neural_net::Layer;
use cozy_chess::{Board, GameStatus};
use cozy_chess_types::{Color, Piece, Move};
use rand::prelude::*;

pub mod ai;
pub mod tools;
