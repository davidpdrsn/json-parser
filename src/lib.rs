use std::collections::HashMap;

#[derive(Debug, Eq, PartialEq)]
pub enum Json {
    Number(i32),
    String(String),
    Boolean(bool),
    Array(Vec<Json>),
    Object(HashMap<String, Json>),
}

pub fn parse_json(input: &str) -> Result<Json, ParseError> {
    parse(json(), input)
}

fn parse<T>(mut parser: T, input: &str) -> Result<T::Output, ParseError>
where
    T: Parse,
{
    match parser.parse(input) {
        Ok((parsed, remaining_input)) => {
            if remaining_input.is_empty() {
                Ok(parsed)
            } else {
                Err(ParseError::UnexpectedEndOfInput)
            }
        }
        Err(parse_error) => Err(parse_error),
    }
}

trait Parse: Sized {
    type Output;

    fn parse<'a>(&mut self, input: &'a str) -> Result<(Self::Output, &'a str), ParseError>;

    fn map<F, B>(self, f: F) -> Map<Self, F>
    where
        F: FnMut(Self::Output) -> B,
    {
        Map { parser: self, f }
    }

    fn flat_map<F, P2>(self, f: F) -> FlatMap<Self, F>
    where
        F: FnMut(Self::Output) -> P2,
        P2: Parse,
    {
        FlatMap { parser: self, f }
    }

    fn until<P>(self, until_parser: P) -> Until<Self, P>
    where
        P: Parse,
    {
        Until {
            parser: self,
            until_parser,
        }
    }

    fn zip_left<P>(self, other: P) -> ZipLeft<Self, P> {
        ZipLeft {
            parser: self,
            other,
        }
    }

    fn zip_right<P>(self, other: P) -> ZipRight<Self, P> {
        ZipRight {
            parser: self,
            other,
        }
    }

    fn or<P>(self, other: P) -> Or<Self, P> {
        Or {
            parser: self,
            other,
        }
    }

    fn sep_by<P>(self, sep: P) -> SepBy<Self, P> {
        SepBy { parser: self, sep }
    }
}

fn many<P>(parser: P) -> impl Parse<Output = Vec<P::Output>>
where
    P: Parse,
{
    Many { parser }
}

impl<P> Parse for Many<P>
where
    P: Parse,
{
    type Output = Vec<P::Output>;

    fn parse<'a>(&mut self, input: &'a str) -> Result<(Self::Output, &'a str), ParseError> {
        let mut output: Vec<P::Output> = vec![];

        let mut remaining = input;
        loop {
            match self.parser.parse(remaining) {
                Ok((parsed, next_remaining)) => {
                    output.push(parsed);
                    remaining = next_remaining;
                }
                Err(_) => return Ok((output, remaining)),
            }
        }
    }
}

#[derive(Debug)]
struct Many<P> {
    parser: P,
}

#[derive(Debug)]
struct SepBy<P, P2> {
    parser: P,
    sep: P2,
}

impl<P, P2> Parse for SepBy<P, P2>
where
    P: Parse,
    P2: Parse,
{
    type Output = Vec<P::Output>;

    fn parse<'a>(&mut self, input: &'a str) -> Result<(Self::Output, &'a str), ParseError> {
        let mut output: Vec<P::Output> = vec![];

        let mut remaining = input;
        loop {
            let (parsed, next_remaining) = self.parser.parse(remaining)?;
            output.push(parsed);
            remaining = next_remaining;

            match self.sep.parse(remaining) {
                Ok((_, next_remaining)) => {
                    remaining = next_remaining;
                }
                Err(_) => return Ok((output, remaining)),
            }
        }
    }
}

#[derive(Debug)]
struct Or<P, P2> {
    parser: P,
    other: P2,
}

impl<P, P2> Parse for Or<P, P2>
where
    P: Parse,
    P2: Parse,
{
    type Output = Either<P::Output, P2::Output>;

    fn parse<'a>(&mut self, input: &'a str) -> Result<(Self::Output, &'a str), ParseError> {
        if let Ok((parsed, remaining)) = self.parser.parse(input) {
            return Ok((Either::A(parsed), remaining));
        }

        let (parsed, remaining) = self.other.parse(input)?;
        Ok((Either::B(parsed), remaining))
    }
}

#[derive(Debug, Eq, PartialEq)]
enum Either<A, B> {
    A(A),
    B(B),
}

#[derive(Debug)]
struct ZipRight<P, P2> {
    parser: P,
    other: P2,
}

impl<P, P2> Parse for ZipRight<P, P2>
where
    P: Parse,
    P2: Parse,
{
    type Output = P2::Output;

    fn parse<'a>(&mut self, input: &'a str) -> Result<(Self::Output, &'a str), ParseError> {
        let (_, remaining) = self.parser.parse(input)?;
        let (output_2, remaining_2) = self.other.parse(remaining)?;
        Ok((output_2, remaining_2))
    }
}

#[derive(Debug)]
struct ZipLeft<P, P2> {
    parser: P,
    other: P2,
}

impl<P, P2> Parse for ZipLeft<P, P2>
where
    P: Parse,
    P2: Parse,
{
    type Output = P::Output;

    fn parse<'a>(&mut self, input: &'a str) -> Result<(Self::Output, &'a str), ParseError> {
        let (output_1, remaining) = self.parser.parse(input)?;
        let (_, remaining_2) = self.other.parse(remaining)?;
        Ok((output_1, remaining_2))
    }
}

#[derive(Debug)]
struct FlatMap<P, F> {
    parser: P,
    f: F,
}

impl<P, F, P2> Parse for FlatMap<P, F>
where
    P: Parse,
    F: FnMut(P::Output) -> P2,
    P2: Parse,
{
    type Output = P2::Output;

    fn parse<'a>(&mut self, input: &'a str) -> Result<(Self::Output, &'a str), ParseError> {
        let (parsed, remaining) = self.parser.parse(input)?;
        (self.f)(parsed).parse(remaining)
    }
}

#[derive(Debug)]
struct Until<P, P2> {
    parser: P,
    until_parser: P2,
}

impl<P, P2> Parse for Until<P, P2>
where
    P: Parse,
    P2: Parse,
{
    type Output = Vec<P::Output>;

    fn parse<'a>(&mut self, input: &'a str) -> Result<(Self::Output, &'a str), ParseError> {
        let mut output: Vec<P::Output> = vec![];

        let mut remaining = input;
        loop {
            match self.until_parser.parse(remaining) {
                Ok(_) => return Ok((output, remaining)),
                Err(_) => {
                    let (parsed, next_remaining) = self.parser.parse(remaining)?;
                    remaining = next_remaining;
                    output.push(parsed);
                }
            }
        }
    }
}

#[derive(Debug)]
struct Map<P, F> {
    parser: P,
    f: F,
}

impl<P, F, B> Parse for Map<P, F>
where
    P: Parse,
    F: FnMut(P::Output) -> B,
{
    type Output = B;

    fn parse<'a>(&mut self, input: &'a str) -> Result<(Self::Output, &'a str), ParseError> {
        let (parsed, remaining) = self.parser.parse(input)?;
        Ok(((self.f)(parsed), remaining))
    }
}

#[derive(Debug, Eq, PartialEq)]
pub enum ParseError {
    Msg(String),
    UnexpectedEndOfInput,
}

fn number() -> NumberParser {
    NumberParser
}

#[derive(Debug)]
struct NumberParser;

impl Parse for NumberParser {
    type Output = i32;

    fn parse<'a>(&mut self, input: &'a str) -> Result<(Self::Output, &'a str), ParseError> {
        let numbers = input
            .chars()
            .take_while(|c| c.is_digit(10))
            .collect::<String>();

        let len = numbers.len();

        if len == 0 {
            Err(ParseError::Msg(format!(
                "Expected number but got {:?}",
                input
            )))
        } else {
            let number = numbers.parse::<i32>().unwrap();
            Ok((number, &input[len..]))
        }
    }
}

fn character(c: char) -> CharParser {
    CharParser(c)
}

#[derive(Debug)]
struct CharParser(char);

impl Parse for CharParser {
    type Output = char;

    fn parse<'a>(&mut self, input: &'a str) -> Result<(Self::Output, &'a str), ParseError> {
        if let Some(c) = input.chars().next() {
            if c == self.0 {
                Ok((c, &input[1..]))
            } else {
                Err(ParseError::Msg(format!(
                    "Expected '{}' but got {:?}",
                    self.0, input
                )))
            }
        } else {
            Err(ParseError::UnexpectedEndOfInput)
        }
    }
}

fn any_char() -> AnyChar {
    AnyChar
}

#[derive(Debug)]
struct AnyChar;

impl Parse for AnyChar {
    type Output = char;

    fn parse<'a>(&mut self, input: &'a str) -> Result<(Self::Output, &'a str), ParseError> {
        if let Some(c) = input.chars().next() {
            Ok((c, &input[1..]))
        } else {
            Err(ParseError::UnexpectedEndOfInput)
        }
    }
}

fn json() -> JsonParser {
    JsonParser
}

#[derive(Debug)]
struct JsonParser;

impl Parse for JsonParser {
    type Output = Json;

    fn parse<'a>(&mut self, input: &'a str) -> Result<(Self::Output, &'a str), ParseError> {
        let mut parser = json_number()
            .or(json_boolean())
            .or(json_string())
            .or(json_array())
            .or(json_object())
            .map(|x| match x {
                Either::A(Either::A(Either::A(Either::A(x)))) => x,
                Either::A(Either::A(Either::A(Either::B(x)))) => x,
                Either::A(Either::A(Either::B(x))) => x,
                Either::A(Either::B(x)) => x,
                Either::B(x) => x,
            });

        parser.parse(input)
    }
}

fn json_object() -> impl Parse<Output = Json> {
    character('{')
        .zip_right(many(whitespace()))
        .flat_map(|_| {
            let key_parser = string()
                .zip_left(many(whitespace()))
                .zip_left(character(':'))
                .zip_left(many(whitespace()));

            let pair = key_parser.flat_map(move |key| {
                let value_parser = json().zip_left(many(whitespace()));
                value_parser.map(move |value| (key.clone(), value))
            });

            pair.sep_by(comma()).map(move |pairs| {
                let map = pairs.into_iter().collect::<HashMap<String, Json>>();
                Json::Object(map)
            })
        })
        .zip_left(many(whitespace()))
        .zip_left(character('}'))
        .zip_left(many(whitespace()))
}

fn whitespace() -> impl Parse<Output = ()> {
    character(' ')
        .or(character('\n'))
        .or(character('\t'))
        .map(|_| ())
}

fn comma() -> impl Parse<Output = ()> {
    character(',').zip_right(many(whitespace())).map(|_| ())
}

fn json_array() -> impl Parse<Output = Json> {
    character('[')
        .zip_left(many(whitespace()))
        .zip_right(json().sep_by(comma()))
        .zip_left(many(whitespace()))
        .zip_left(character(']'))
        .zip_left(many(whitespace()))
        .map(|values| Json::Array(values))
}

fn json_number() -> impl Parse<Output = Json> {
    number().map(|n| Json::Number(n))
}

fn string() -> impl Parse<Output = String> {
    character('"')
        .flat_map(|_quote| any_char().until(character('"')))
        .map(|chars| (chars.into_iter().collect::<String>()))
        .zip_left(character('"'))
}

fn json_string() -> impl Parse<Output = Json> {
    string().map(Json::String)
}

fn json_boolean() -> impl Parse<Output = Json> {
    let true_parser = character('t')
        .flat_map(|_| character('r'))
        .flat_map(|_| character('u'))
        .flat_map(|_| character('e'))
        .map(|_| true);

    let false_parser = character('f')
        .flat_map(|_| character('a'))
        .flat_map(|_| character('l'))
        .flat_map(|_| character('s'))
        .flat_map(|_| character('e'))
        .map(|_| false);

    true_parser.or(false_parser).map(|either| match either {
        Either::A(x) => Json::Boolean(x),
        Either::B(x) => Json::Boolean(x),
    })
}

#[cfg(test)]
mod test {
    #[allow(unused_imports)]
    use super::*;

    #[test]
    fn parse_number() {
        let (number, remaining) = number().parse("123").unwrap();

        assert_eq!(number, 123);
        assert_eq!(remaining, "");
    }

    #[test]
    fn parse_number_with_remainder() {
        let (number, remaining) = number().parse("123a").unwrap();

        assert_eq!(number, 123);
        assert_eq!(remaining, "a");
    }

    #[test]
    fn parse_all_input_to_number() {
        let parsed = parse(number(), "1").unwrap();
        assert_eq!(parsed, 1);
    }

    #[test]
    fn parse_number_unexpected_eoi() {
        assert_eq!(parse(number(), "1a"), Err(ParseError::UnexpectedEndOfInput));
    }

    #[test]
    fn parse_number_from_not_a_number() {
        assert_eq!(
            parse(number(), "a"),
            Err(ParseError::Msg("Expected number but got \"a\"".to_string()))
        );
    }

    #[test]
    fn parse_number_persian_digit() {
        let parsed = parse(number(), "۳");
        assert_eq!(parsed, Err(ParseError::Msg("Expected number but got \"۳\"".to_string())));
    }

    #[test]
    fn map() {
        let mut parser = number().map(|n| n + 1);
        let (number, remaining) = parser.parse("123").unwrap();

        assert_eq!(number, 124);
        assert_eq!(remaining, "");
    }

    #[test]
    fn parse_char() {
        assert_eq!(parse(character('a'), "a"), Ok('a'),);
    }

    #[test]
    fn parse_char_no_match() {
        assert_eq!(
            parse(character('b'), "a"),
            Err(ParseError::Msg("Expected 'b' but got \"a\"".to_string()))
        );
    }

    #[test]
    fn parse_char_with_no_input() {
        assert_eq!(
            parse(character('a'), ""),
            Err(ParseError::UnexpectedEndOfInput),
        );
    }

    #[test]
    fn until() {
        let mut parser = character('a').until(character('b'));
        let (number, remaining) = parser.parse("aaab").unwrap();

        assert_eq!(number, vec!['a', 'a', 'a']);
        assert_eq!(remaining, "b");
    }

    #[test]
    fn until_no_match() {
        let mut parser = character('a').until(character('b'));
        let (number, remaining) = parser.parse("b").unwrap();

        assert_eq!(number, vec![]);
        assert_eq!(remaining, "b");
    }

    #[test]
    fn until_no_match_on_other_parser() {
        let mut parser = character('a').until(character('b'));

        assert_eq!(
            parser.parse("1"),
            Err(ParseError::Msg("Expected 'a' but got \"1\"".to_string()))
        );
    }

    #[test]
    fn flat_map() {
        let mut parser = character('a').flat_map(|c| {
            assert_eq!(c, 'a');
            character('b')
        });
        let (number, remaining) = parser.parse("ab").unwrap();

        assert_eq!(number, 'b');
        assert_eq!(remaining, "");
    }

    #[test]
    fn test_any_char() {
        let mut parser = any_char();
        let (number, remaining) = parser.parse("ab").unwrap();

        assert_eq!(number, 'a');
        assert_eq!(remaining, "b");
    }

    #[test]
    fn zip_left() {
        let mut parser = character('x').zip_left(character('a'));

        let (number, remaining) = parser.parse("xa").unwrap();

        assert_eq!(number, 'x');
        assert_eq!(remaining, "");
    }

    #[test]
    fn zip_right() {
        let mut parser = character('x').zip_right(character('a'));

        let (number, remaining) = parser.parse("xa").unwrap();

        assert_eq!(number, 'a');
        assert_eq!(remaining, "");
    }

    #[test]
    fn parse_string() {
        let mut parser = json_string();

        let (number, remaining) = parser.parse("\"aab\"").unwrap();

        assert_eq!(number, Json::String("aab".to_string()));
        assert_eq!(remaining, "");
    }

    #[test]
    fn parse_true() {
        let mut parser = character('t')
            .flat_map(|_| character('r'))
            .flat_map(|_| character('u'))
            .flat_map(|_| character('e'))
            .map(|_| true);

        let (number, remaining) = parser.parse("true").unwrap();

        assert_eq!(number, true);
        assert_eq!(remaining, "");
    }

    #[test]
    fn parse_false() {
        let mut parser = character('f')
            .flat_map(|_| character('a'))
            .flat_map(|_| character('l'))
            .flat_map(|_| character('s'))
            .flat_map(|_| character('e'))
            .map(|_| false);

        let (number, remaining) = parser.parse("false").unwrap();

        assert_eq!(number, false);
        assert_eq!(remaining, "");
    }

    #[test]
    fn or() {
        let mut parser = character('a').or(character('b'));

        let (number, remaining) = parser.parse("a").unwrap();
        assert_eq!(number, Either::A('a'));
        assert_eq!(remaining, "");

        let (number, remaining) = parser.parse("b").unwrap();
        assert_eq!(number, Either::B('b'));
        assert_eq!(remaining, "");
    }

    #[test]
    fn parse_boolean() {
        let mut parser = json_boolean();

        let (number, remaining) = parser.parse("true").unwrap();
        assert_eq!(number, Json::Boolean(true));
        assert_eq!(remaining, "");

        let (number, remaining) = parser.parse("false").unwrap();
        assert_eq!(number, Json::Boolean(false));
        assert_eq!(remaining, "");
    }

    #[test]
    fn parse_sep_by() {
        let mut parser = character('a').sep_by(character(','));

        let (number, remaining) = parser.parse("a,a,a,a").unwrap();
        assert_eq!(number, vec!['a', 'a', 'a', 'a']);
        assert_eq!(remaining, "");
    }

    #[test]
    fn parse_array_of_numbers() {
        let whitespace = character(' ').or(character('\n')).or(character('\t'));

        let sep = character(',').zip_right(many(whitespace));

        let mut parser = character('[')
            .zip_right(number().sep_by(sep))
            .zip_left(character(']'));

        let (number, remaining) = parser.parse("[1, \n 1,   \t\n  1,1]").unwrap();
        assert_eq!(number, vec![1, 1, 1, 1]);
        assert_eq!(remaining, "");
    }

    #[test]
    fn parse_json_array() {
        let mut parser = json();

        let (number, remaining) = parser.parse("[1, false, \"foo\"]").unwrap();
        assert_eq!(
            number,
            Json::Array(vec![
                Json::Number(1),
                Json::Boolean(false),
                Json::String("foo".to_string())
            ])
        );
        assert_eq!(remaining, "");
    }

    #[test]
    fn parse_json_object() {
        let (object, remaining) = json().parse("{ \"foo\": 123 }").unwrap();
        assert_eq!(
            object,
            Json::Object({
                let mut map = HashMap::new();
                map.insert("foo".to_string(), Json::Number(123));
                map
            })
        );
        assert_eq!(remaining, "");
    }

    #[test]
    fn array_as_object_value() {
        let (object, remaining) = json().parse("{ \"foo\": [\n1,2,3] }").unwrap();
        assert_eq!(
            object,
            Json::Object({
                let mut map = HashMap::new();
                map.insert(
                    "foo".to_string(),
                    Json::Array(vec![Json::Number(1), Json::Number(2), Json::Number(3)]),
                );
                map
            })
        );
        assert_eq!(remaining, "");
    }
}
