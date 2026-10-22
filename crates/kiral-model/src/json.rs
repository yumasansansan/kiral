// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later
//
// This file is part of KIRAL. The SPDX lines name its copyright holder and its
// license, in the machine-readable form of the REUSE specification: the GNU
// General Public License, version 3 or any later version
// (LICENSES/GPL-3.0-or-later.txt).

//! JSON (RFC 8259), read by hand: the headers of safetensors files, the
//! tokenizers' vocabularies and settings, the benchmark's items.
//!
//! The parser takes any bytes. What is not JSON is an error, never a panic, and
//! so is what nests deeper than [`MAX_DEPTH`], which would otherwise take the
//! stack. An object keeps its members in their order and refuses a name given
//! twice, which two readers could take two ways. A number is kept as its text,
//! read as an integer or a double when it is asked for.

use std::fmt;

/// The deepest that arrays and objects may nest.
pub const MAX_DEPTH: usize = 128;

/// A JSON value.
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    /// `null`.
    Null,
    /// `true` or `false`.
    Bool(bool),
    /// A number, as its text.
    Number(Number),
    /// A string.
    String(String),
    /// An array.
    Array(Vec<Value>),
    /// An object.
    Object(Object),
}

/// A number as it is written: `-? (0 | [1-9][0-9]*) (. [0-9]+)? ([eE] [+-]? [0-9]+)?`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Number(String);

impl Number {
    /// The number's text.
    #[must_use]
    pub fn as_text(&self) -> &str {
        &self.0
    }

    /// The number as an unsigned integer: none where it has a fraction or an
    /// exponent, is negative, or is above `u64::MAX`.
    #[must_use]
    pub fn as_u64(&self) -> Option<u64> {
        self.0.parse().ok()
    }

    /// The number as a signed integer: none where it has a fraction or an
    /// exponent, or is out of the range of `i64`.
    #[must_use]
    pub fn as_i64(&self) -> Option<i64> {
        self.0.parse().ok()
    }

    /// The number as the nearest double; infinite beyond the doubles' range.
    ///
    /// # Panics
    ///
    /// Never: the text of a JSON number always reads as a double.
    #[must_use]
    pub fn as_f64(&self) -> f64 {
        self.0.parse().expect("the text of a JSON number reads as a double")
    }
}

/// An object: its members in their order, each name once.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Object(Vec<(String, Value)>);

impl Object {
    /// The value of a member.
    #[must_use]
    pub fn get(&self, name: &str) -> Option<&Value> {
        self.0.iter().find(|(member, _)| member == name).map(|(_, value)| value)
    }

    /// The members, in their order.
    pub fn iter(&self) -> impl Iterator<Item = (&str, &Value)> {
        self.0.iter().map(|(name, value)| (name.as_str(), value))
    }

    /// The number of members.
    #[must_use]
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Whether the object has no members.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

impl Value {
    /// The string, where the value is one.
    #[must_use]
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::String(text) => Some(text),
            _ => None,
        }
    }

    /// The object, where the value is one.
    #[must_use]
    pub fn as_object(&self) -> Option<&Object> {
        match self {
            Self::Object(object) => Some(object),
            _ => None,
        }
    }

    /// The array, where the value is one.
    #[must_use]
    pub fn as_array(&self) -> Option<&[Value]> {
        match self {
            Self::Array(values) => Some(values),
            _ => None,
        }
    }

    /// The boolean, where the value is one.
    #[must_use]
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Self::Bool(value) => Some(*value),
            _ => None,
        }
    }

    /// The number, where the value is one.
    #[must_use]
    pub fn as_number(&self) -> Option<&Number> {
        match self {
            Self::Number(number) => Some(number),
            _ => None,
        }
    }

    /// The number as an unsigned integer, where the value is one that is.
    #[must_use]
    pub fn as_u64(&self) -> Option<u64> {
        self.as_number().and_then(Number::as_u64)
    }

    /// The number as a double, where the value is a number.
    #[must_use]
    pub fn as_f64(&self) -> Option<f64> {
        self.as_number().map(Number::as_f64)
    }
}

/// Why bytes are not JSON, and where.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Error {
    /// The offset of the byte where reading stopped.
    pub offset: usize,
    /// What was wrong there.
    pub message: &'static str,
}

impl fmt::Display for Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{} at byte {}", self.message, self.offset)
    }
}

impl std::error::Error for Error {}

/// The value of a JSON text: one value, with white space around it and nothing
/// else.
///
/// # Errors
///
/// Where the bytes are not UTF-8 and JSON, or nest deeper than [`MAX_DEPTH`].
pub fn parse(bytes: &[u8]) -> Result<Value, Error> {
    if let Err(error) = std::str::from_utf8(bytes) {
        return Err(Error {
            offset: error.valid_up_to(),
            message: "bytes that are not UTF-8",
        });
    }
    let mut reader = Reader { bytes, at: 0 };
    reader.skip_space();
    let value = reader.value(0)?;
    reader.skip_space();
    if reader.at != bytes.len() {
        return Err(reader.error("something after the value"));
    }
    Ok(value)
}

struct Reader<'a> {
    bytes: &'a [u8],
    at: usize,
}

impl Reader<'_> {
    fn error(&self, message: &'static str) -> Error {
        Error {
            offset: self.at,
            message,
        }
    }

    fn peek(&self) -> Option<u8> {
        self.bytes.get(self.at).copied()
    }

    fn skip_space(&mut self) {
        while let Some(b' ' | b'\t' | b'\n' | b'\r') = self.peek() {
            self.at += 1;
        }
    }

    fn expect_word(&mut self, word: &[u8], value: Value) -> Result<Value, Error> {
        if self.bytes[self.at..].starts_with(word) {
            self.at += word.len();
            Ok(value)
        } else {
            Err(self.error("a word that is not true, false or null"))
        }
    }

    fn value(&mut self, depth: usize) -> Result<Value, Error> {
        match self.peek() {
            None => Err(self.error("the end where a value was expected")),
            Some(b'{') => self.object(depth + 1),
            Some(b'[') => self.array(depth + 1),
            Some(b'"') => self.string().map(Value::String),
            Some(b't') => self.expect_word(b"true", Value::Bool(true)),
            Some(b'f') => self.expect_word(b"false", Value::Bool(false)),
            Some(b'n') => self.expect_word(b"null", Value::Null),
            Some(b'-' | b'0'..=b'9') => self.number().map(Value::Number),
            Some(_) => Err(self.error("a character that begins no value")),
        }
    }

    fn object(&mut self, depth: usize) -> Result<Value, Error> {
        if depth > MAX_DEPTH {
            return Err(self.error("arrays and objects nested too deep"));
        }
        self.at += 1;
        let mut members: Vec<(String, Value)> = Vec::new();
        self.skip_space();
        if self.peek() == Some(b'}') {
            self.at += 1;
            return Ok(Value::Object(Object(members)));
        }
        loop {
            self.skip_space();
            if self.peek() != Some(b'"') {
                return Err(self.error("a member without a name in quotes"));
            }
            let start = self.at;
            let name = self.string()?;
            if members.iter().any(|(member, _)| *member == name) {
                return Err(Error {
                    offset: start,
                    message: "a name given twice in one object",
                });
            }
            self.skip_space();
            if self.peek() != Some(b':') {
                return Err(self.error("a member's name without a colon after it"));
            }
            self.at += 1;
            self.skip_space();
            let value = self.value(depth)?;
            members.push((name, value));
            self.skip_space();
            match self.peek() {
                Some(b',') => self.at += 1,
                Some(b'}') => {
                    self.at += 1;
                    return Ok(Value::Object(Object(members)));
                }
                _ => return Err(self.error("an object's member followed by neither a comma nor its end")),
            }
        }
    }

    fn array(&mut self, depth: usize) -> Result<Value, Error> {
        if depth > MAX_DEPTH {
            return Err(self.error("arrays and objects nested too deep"));
        }
        self.at += 1;
        let mut values = Vec::new();
        self.skip_space();
        if self.peek() == Some(b']') {
            self.at += 1;
            return Ok(Value::Array(values));
        }
        loop {
            self.skip_space();
            values.push(self.value(depth)?);
            self.skip_space();
            match self.peek() {
                Some(b',') => self.at += 1,
                Some(b']') => {
                    self.at += 1;
                    return Ok(Value::Array(values));
                }
                _ => return Err(self.error("an array's value followed by neither a comma nor its end")),
            }
        }
    }

    fn digits(&mut self) -> usize {
        let start = self.at;
        while let Some(b'0'..=b'9') = self.peek() {
            self.at += 1;
        }
        self.at - start
    }

    fn number(&mut self) -> Result<Number, Error> {
        let start = self.at;
        if self.peek() == Some(b'-') {
            self.at += 1;
        }
        match self.peek() {
            Some(b'0') => self.at += 1,
            Some(b'1'..=b'9') => {
                self.digits();
            }
            _ => return Err(self.error("a minus sign without digits")),
        }
        if self.peek() == Some(b'.') {
            self.at += 1;
            if self.digits() == 0 {
                return Err(self.error("a decimal point without digits after it"));
            }
        }
        if let Some(b'e' | b'E') = self.peek() {
            self.at += 1;
            if let Some(b'+' | b'-') = self.peek() {
                self.at += 1;
            }
            if self.digits() == 0 {
                return Err(self.error("an exponent without digits"));
            }
        }
        let text = std::str::from_utf8(&self.bytes[start..self.at]).expect("ASCII digits and signs");
        Ok(Number(text.to_owned()))
    }

    fn hex4(&mut self) -> Result<u32, Error> {
        let Some(digits) = self.bytes.get(self.at..self.at + 4) else {
            return Err(self.error("an escape of fewer than four hexadecimal digits"));
        };
        let mut value = 0;
        for &digit in digits {
            let nibble = char::from(digit)
                .to_digit(16)
                .ok_or_else(|| self.error("an escape of fewer than four hexadecimal digits"))?;
            value = value * 16 + nibble;
        }
        self.at += 4;
        Ok(value)
    }

    fn string(&mut self) -> Result<String, Error> {
        self.at += 1;
        let mut text = String::new();
        loop {
            let start = self.at;
            while let Some(byte) = self.peek() {
                if byte == b'"' || byte == b'\\' || byte < 0x20 {
                    break;
                }
                self.at += 1;
            }
            // The run of bytes is whole characters: the input is UTF-8, and it stops
            // only at ASCII bytes.
            text.push_str(std::str::from_utf8(&self.bytes[start..self.at]).expect("UTF-8 checked before reading"));
            match self.peek() {
                None => return Err(self.error("a string without its closing quote")),
                Some(b'"') => {
                    self.at += 1;
                    return Ok(text);
                }
                Some(b'\\') => {
                    self.at += 1;
                    let escape = self.peek().ok_or_else(|| self.error("an escape at the end"))?;
                    self.at += 1;
                    match escape {
                        b'"' => text.push('"'),
                        b'\\' => text.push('\\'),
                        b'/' => text.push('/'),
                        b'b' => text.push('\u{8}'),
                        b'f' => text.push('\u{c}'),
                        b'n' => text.push('\n'),
                        b'r' => text.push('\r'),
                        b't' => text.push('\t'),
                        b'u' => text.push(self.unicode_escape()?),
                        _ => return Err(self.error("an escape that JSON does not have")),
                    }
                }
                Some(_) => return Err(self.error("a control character in a string")),
            }
        }
    }

    /// The character of a `\u` escape, a surrogate pair's two escapes together.
    fn unicode_escape(&mut self) -> Result<char, Error> {
        let first = self.hex4()?;
        let code = match first {
            0xd800..=0xdbff => {
                if !self.bytes[self.at..].starts_with(b"\\u") {
                    return Err(self.error("a high surrogate without a low one after it"));
                }
                self.at += 2;
                let second = self.hex4()?;
                if !(0xdc00..=0xdfff).contains(&second) {
                    return Err(self.error("a high surrogate without a low one after it"));
                }
                0x10000 + ((first - 0xd800) << 10) + (second - 0xdc00)
            }
            0xdc00..=0xdfff => return Err(self.error("a low surrogate without a high one before it")),
            _ => first,
        };
        char::from_u32(code).ok_or_else(|| self.error("an escape of no character"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn number(text: &str) -> Value {
        Value::Number(Number(text.to_owned()))
    }

    #[test]
    fn values_are_read() {
        assert_eq!(parse(b" null "), Ok(Value::Null));
        assert_eq!(parse(b"true"), Ok(Value::Bool(true)));
        assert_eq!(
            parse(b"[1, -0.5e+3, 0]"),
            Ok(Value::Array(vec![number("1"), number("-0.5e+3"), number("0")]))
        );
        let value = parse(br#"{"a": {"b": []}, "c": "d"}"#).expect("JSON");
        let object = value.as_object().expect("an object");
        assert_eq!(object.len(), 2);
        assert_eq!(object.get("c").and_then(Value::as_str), Some("d"));
        assert_eq!(object.iter().map(|(name, _)| name).collect::<Vec<_>>(), ["a", "c"]);
    }

    #[test]
    fn strings_take_their_escapes() {
        let value = parse(br#""a\"\\\/\b\f\n\r\t\u00e9\u3042\ud83d\ude00""#).expect("JSON");
        assert_eq!(value.as_str(), Some("a\"\\/\u{8}\u{c}\n\r\té\u{3042}\u{1f600}"));
        assert_eq!(parse("\"かな\"".as_bytes()).expect("JSON").as_str(), Some("かな"));
    }

    #[test]
    fn numbers_are_read_as_they_are_asked_for() {
        let value = parse(b"18446744073709551615").expect("JSON");
        assert_eq!(value.as_u64(), Some(u64::MAX));
        assert_eq!(parse(b"18446744073709551616").expect("JSON").as_u64(), None);
        assert_eq!(parse(b"1.5").expect("JSON").as_u64(), None);
        assert_eq!(parse(b"1e2").expect("JSON").as_u64(), None);
        assert_eq!(parse(b"1e400").expect("JSON").as_f64(), Some(f64::INFINITY));
        assert_eq!(
            parse(b"-12").expect("JSON").as_number().and_then(Number::as_i64),
            Some(-12)
        );
        assert_eq!(parse(b"0.1").expect("JSON").as_f64(), Some(0.1));
    }

    #[test]
    fn what_is_not_json_is_refused() {
        for text in [
            &b""[..],
            b"nul",
            b"[1,]",
            b"{\"a\" 1}",
            b"{\"a\":1,}",
            b"{1:2}",
            b"01",
            b"1.",
            b".5",
            b"-",
            b"1e",
            b"+1",
            b"\"\x01\"",
            b"\"\\x\"",
            b"\"\\u12\"",
            b"\"\\ud800\"",
            b"\"\\udc00\"",
            b"\"\\ud800\\u0041\"",
            b"\"unterminated",
            b"[1] 2",
            b"{\"a\":1,\"a\":2}",
            b"\xff",
            b"\"\xc3\"",
        ] {
            assert!(parse(text).is_err(), "{:?} was taken", String::from_utf8_lossy(text));
        }
    }

    #[test]
    fn nesting_is_limited() {
        let deep = |depth: usize| [vec![b'['; depth], vec![b']'; depth]].concat();
        assert!(parse(&deep(MAX_DEPTH)).is_ok());
        let error = parse(&deep(MAX_DEPTH + 1)).expect_err("too deep");
        assert_eq!(error.message, "arrays and objects nested too deep");
        assert!(parse(&deep(100_000)).is_err());
    }
}
