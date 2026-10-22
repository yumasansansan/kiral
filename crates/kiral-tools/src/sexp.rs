// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later
//
// This file is part of KIRAL. The SPDX lines name its copyright holder and its
// license, in the machine-readable form of the REUSE specification: the GNU
// General Public License, version 3 or any later version
// (LICENSES/GPL-3.0-or-later.txt).

//! The S-expressions that `mozc_emacs_helper` answers in: lists, dotted pairs,
//! strings, symbols and integers, as Emacs Lisp prints them.
//!
//! Its answers are association lists: `((emacs-event-id . 3)(output . (...)))`,
//! a key and its value as a dotted pair, or a key and the values after it, as
//! in `(segment (...) (...))`; [`Value::get`] takes either. The reader takes
//! any bytes, and nests at most [`MAX_DEPTH`] lists deep.

/// The deepest that lists may nest.
pub const MAX_DEPTH: usize = 64;

/// An S-expression.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Value {
    /// A list, and the value after its dot, where it has one.
    List(Vec<Value>, Option<Box<Value>>),
    /// A symbol.
    Symbol(String),
    /// A string.
    String(String),
    /// An integer.
    Integer(i64),
}

impl Value {
    /// The value of a key in an association list: the value of a dotted pair
    /// `(key . value)`, or the list of the values after the key in `(key a b)`.
    #[must_use]
    pub fn get(&self, key: &str) -> Option<Value> {
        let Self::List(entries, _) = self else {
            return None;
        };
        entries.iter().find_map(|entry| {
            let Self::List(items, tail) = entry else {
                return None;
            };
            let (Self::Symbol(name), rest) = items.split_first()? else {
                return None;
            };
            if name != key {
                return None;
            }
            match (rest, tail) {
                ([], Some(tail)) => Some((**tail).clone()),
                (rest, None) => Some(Self::List(rest.to_vec(), None)),
                _ => None,
            }
        })
    }

    /// The items of a list.
    #[must_use]
    pub fn items(&self) -> &[Value] {
        match self {
            Self::List(items, _) => items,
            _ => &[],
        }
    }

    /// The string, where the value is one.
    #[must_use]
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::String(text) => Some(text),
            _ => None,
        }
    }

    /// The symbol's name, where the value is one.
    #[must_use]
    pub fn as_symbol(&self) -> Option<&str> {
        match self {
            Self::Symbol(name) => Some(name),
            _ => None,
        }
    }

    /// The integer, where the value is one.
    #[must_use]
    pub fn as_integer(&self) -> Option<i64> {
        match self {
            Self::Integer(value) => Some(*value),
            _ => None,
        }
    }
}

/// The value of a text: one S-expression, with white space around it.
///
/// # Errors
///
/// Where the text is not one, or nests too deep, with what was wrong.
pub fn parse(text: &str) -> Result<Value, String> {
    let mut reader = Reader {
        characters: text.chars().collect(),
        at: 0,
    };
    let value = reader.value(0)?;
    reader.skip_space();
    if reader.at != reader.characters.len() {
        return Err(format!("something after the expression, at {}", reader.at));
    }
    Ok(value)
}

struct Reader {
    characters: Vec<char>,
    at: usize,
}

impl Reader {
    fn peek(&self) -> Option<char> {
        self.characters.get(self.at).copied()
    }

    fn skip_space(&mut self) {
        while self.peek().is_some_and(char::is_whitespace) {
            self.at += 1;
        }
    }

    fn value(&mut self, depth: usize) -> Result<Value, String> {
        self.skip_space();
        match self.peek() {
            None => Err("the end where an expression was expected".to_owned()),
            Some('(') => self.list(depth + 1),
            Some(')') => Err(format!(
                "a closing parenthesis where an expression was expected, at {}",
                self.at
            )),
            Some('"') => self.string(),
            Some(_) => Ok(self.atom()),
        }
    }

    fn list(&mut self, depth: usize) -> Result<Value, String> {
        if depth > MAX_DEPTH {
            return Err("lists nested too deep".to_owned());
        }
        self.at += 1;
        let mut items = Vec::new();
        loop {
            self.skip_space();
            match self.peek() {
                None => return Err("a list without its closing parenthesis".to_owned()),
                Some(')') => {
                    self.at += 1;
                    return Ok(Value::List(items, None));
                }
                Some('.')
                    if !items.is_empty() && self.characters.get(self.at + 1).is_some_and(|c| c.is_whitespace()) =>
                {
                    self.at += 1;
                    let tail = self.value(depth)?;
                    self.skip_space();
                    if self.peek() != Some(')') {
                        return Err(format!(
                            "a dotted pair with more than one value after its dot, at {}",
                            self.at
                        ));
                    }
                    self.at += 1;
                    return Ok(Value::List(items, Some(Box::new(tail))));
                }
                Some(_) => items.push(self.value(depth)?),
            }
        }
    }

    fn string(&mut self) -> Result<Value, String> {
        self.at += 1;
        let mut text = String::new();
        loop {
            match self.peek() {
                None => return Err("a string without its closing quote".to_owned()),
                Some('"') => {
                    self.at += 1;
                    return Ok(Value::String(text));
                }
                Some('\\') => {
                    let escaped = self.characters.get(self.at + 1).copied();
                    match escaped {
                        Some('n') => text.push('\n'),
                        Some('t') => text.push('\t'),
                        Some(character) => text.push(character),
                        None => return Err("a string ending in a backslash".to_owned()),
                    }
                    self.at += 2;
                }
                Some(character) => {
                    text.push(character);
                    self.at += 1;
                }
            }
        }
    }

    fn atom(&mut self) -> Value {
        let start = self.at;
        while self
            .peek()
            .is_some_and(|character| !character.is_whitespace() && !matches!(character, '(' | ')' | '"'))
        {
            self.at += 1;
        }
        let text: String = self.characters[start..self.at].iter().collect();
        text.parse().map_or_else(|_| Value::Symbol(text), Value::Integer)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_helpers_answers_are_read() {
        let answer = r#"((emacs-event-id . 43)(emacs-session-id . 1)(output . ((id . "951")(mode . hiragana)(preedit . ((cursor . 17)(segment ((annotation . highlight)(value . "家計を")(value-length . 3)(key . "かけいを"))((annotation . underline)(value . "助ける\"ため")(key . "たすける")))(highlighted-position . 0))))))"#;
        let value = parse(answer).expect("an answer");
        assert_eq!(value.get("emacs-event-id").and_then(|id| id.as_integer()), Some(43));
        let output = value.get("output").expect("an output");
        assert_eq!(output.get("mode").as_ref().and_then(Value::as_symbol), Some("hiragana"));
        let preedit = output.get("preedit").expect("a preedit");
        let segments = preedit.get("segment").expect("segments");
        let values: Vec<String> = segments
            .items()
            .iter()
            .filter_map(|segment| segment.get("value").and_then(|value| value.as_str().map(str::to_owned)))
            .collect();
        assert_eq!(values, ["家計を", "助ける\"ため"]);
        assert_eq!(output.get("candidates"), None);
        assert_eq!(
            parse("((output . ()))").expect("an answer").get("output"),
            Some(Value::List(Vec::new(), None))
        );
    }

    /// Answers changed at random -- characters flipped, put in, taken out, the
    /// end cut off -- are read or refused, never a panic.
    #[test]
    fn any_text_is_read_or_refused() {
        let seed: Vec<char> = r#"((emacs-event-id . 7)(output . ((preedit . ((cursor . 2)(segment ((value . "か\"な")(key . "かな"))))))))"#
            .chars()
            .collect();
        assert!(
            parse(&seed.iter().collect::<String>()).is_ok(),
            "a seed that is not an answer"
        );
        let mut state = 0x2545_f491_4f6c_dd1d_u64;
        let mut next = |bound: usize| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            usize::try_from(state % u64::try_from(bound).expect("a small bound")).expect("below the bound")
        };
        let pieces = ['(', ')', '"', '\\', '.', ' ', 'a', '1', '-'];
        for _ in 0..5000 {
            let mut text = seed.clone();
            for _ in 0..=next(4) {
                let at = next(text.len() + 1);
                match next(3) {
                    0 if at < text.len() => {
                        text.remove(at);
                    }
                    1 => text.insert(at, pieces[next(pieces.len())]),
                    _ => text.truncate(at),
                }
            }
            let _ = parse(&text.iter().collect::<String>());
        }
    }

    #[test]
    fn what_is_not_an_expression_is_refused() {
        for text in ["", "(", ")", "(a . b c)", "(\"open", "(a))", "(\"\\", "(((("] {
            assert!(parse(text).is_err(), "{text:?} was taken");
        }
        let deep = "(".repeat(MAX_DEPTH + 1) + &")".repeat(MAX_DEPTH + 1);
        assert!(parse(&deep).is_err());
        assert_eq!(parse(" -12 "), Ok(Value::Integer(-12)));
        assert_eq!(parse("-"), Ok(Value::Symbol("-".to_owned())));
    }
}
