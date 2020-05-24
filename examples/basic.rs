use json_parser::*;
use std::fs;

fn main() {
    let contents = fs::read_to_string("examples/sample.json").unwrap();
    let contents = contents.trim();
    let json = match parse_json(&contents) {
        Ok(x) => x,
        Err(ParseError::Msg(msg)) => {
            eprintln!("{}", msg);
            std::process::exit(1)
        }
        Err(ParseError::UnexpectedEndOfInput) => {
            eprintln!("UnexpectedEndOfInput");
            std::process::exit(1)
        }
    };
    dbg!(json);
}
