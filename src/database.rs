use base64::{engine::general_purpose, Engine};
use cozy_chess::BoardBuilder;
use cozy_chess_types::{Square, Piece, Color, CastleRights};
use crate::{chess::ChessState, game::Game};
use ndarray::Array1;
use rusqlite::{Connection, backup::Backup, params};
use anyhow::Result;


pub fn load_to_memory(conn: &Connection) -> Result<Connection> {
    let mut to = Connection::open_in_memory()?;
    {
        let backup = Backup::new(conn, &mut to)?;
        backup.step(-1)?;
    }
    Ok(to)
}


pub fn init(uri: &String) -> Connection {
    Connection::open(uri).unwrap()
}

#[derive(Debug)]
pub struct Instance {
    pub board: ChessState,
    wins: u64,
    losses: u64,
}

impl Instance {
    pub fn from_str(text: &str, wins: u64, losses: u64) -> Self {
        let buf = general_purpose::STANDARD.decode(text).unwrap();
        Self::from_bytes(buf, wins, losses)
    }

    pub fn from_bytes(buf: Vec<u8>, wins: u64, losses: u64) -> Self {
        let mut player = Color::White;
        let mut builder = BoardBuilder::empty();

        *builder.castle_rights_mut(Color::White) = CastleRights::EMPTY;
        *builder.castle_rights_mut(Color::Black) = CastleRights::EMPTY;

        buf.iter().rev().enumerate().for_each(|(idx, c)| {
            let c1 = c & 0b1111;
            let sq = builder.square_mut(Square::ALL[7 - (idx % 8) + 8 * (idx / 8)]);

            if c1 != 0 {
                if c & 0b10000000 == 0 && c & 0b1000000 == 0 {
                    player = Color::Black;
                }
            }
            *sq = match (c1, c & 0b10000000 == 0) {
                (1, true ) => Some((Piece::Pawn,   Color::Black)),
                (1, false) => Some((Piece::Pawn,   Color::White)),
                (2, true ) => Some((Piece::Rook,   Color::Black)),
                (2, false) => Some((Piece::Rook,   Color::White)),
                (3, true ) => Some((Piece::Knight, Color::Black)),
                (3, false) => Some((Piece::Knight, Color::White)),
                (4, true ) => Some((Piece::Bishop, Color::Black)),
                (4, false) => Some((Piece::Bishop, Color::White)),
                (5, true ) => Some((Piece::Queen,  Color::Black)),
                (5, false) => Some((Piece::Queen,  Color::White)),
                (6, true ) => Some((Piece::King,   Color::Black)),
                (6, false) => Some((Piece::King,   Color::White)),
                _ => None,
            };
        });

        builder.side_to_move = player;

        return Self {
            board: ChessState { board: builder.build().unwrap() },
            wins,
            losses,
        };
    }


    pub fn winrate(&self) -> (f64, f64) {
        let t = (self.wins + self.losses) as f64;
        let w = self.wins as f64 / t;
        (w, 1.0 - w)
    }

    pub fn state(&self) -> Array1<f64> {
        self.board.state()
    }
}

pub fn get_batch(conn: &Connection, min_occurences: usize, size: usize) -> Vec<Instance> {
    let mut stmnt = conn.prepare(
        "SELECT * FROM chess_moves 
        WHERE wins + losses > ?1 
        ORDER BY RANDOM() LIMIT ?2", 
    ).unwrap();

    stmnt.query_map(
        params![
            min_occurences, 
            size
        ],
        |row| Ok({
            // let hash: i64 = row.get(0).unwrap();
            let board = &row.get::<_, String>(1).unwrap();
            let wins: u64 = row.get(2).unwrap();
            let losses: u64 = row.get(3).unwrap();
            
            Instance::from_str(&board, wins, losses)
        })
    )   
        .unwrap()
        .filter_map(|x| x.ok())
        .collect()
    
}