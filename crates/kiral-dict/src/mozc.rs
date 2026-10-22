// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later
//
// This file is part of KIRAL. The SPDX lines name its copyright holder and its
// license, in the machine-readable form of the REUSE specification: the GNU
// General Public License, version 3 or any later version
// (LICENSES/GPL-3.0-or-later.txt).

//! Mozc's open-source dictionary, read from its files.
//!
//! `dictionary00.txt` to `dictionary09.txt` hold a word a line: its reading,
//! the ids of its classes on its left and on its right, its cost and its
//! surface, separated by tabs, and on a few lines a sixth field,
//! `SPELLING_CORRECTION`, for a reading spelt otherwise than the word is
//! (`ばんしあ` for ヴァンシア). `connection_single_column.txt` holds the number of
//! ids on its first line, then a cost a line: that of a word whose right id is
//! `r` followed by one whose left id is `l` at `r * ids + l`, as Mozc's
//! generator reads it. Id 0 is the beginning and the end of a sentence, whose
//! connection to itself costs nothing. `id.def` names the part of speech of
//! each id, `0 BOS/EOS,*,*,*,*,*,*` and so on. `README.txt` and `LICENSE` are
//! the notices that go with any copy.

use std::path::{Path, PathBuf};

use crate::Word;

/// A word whose reading is spelt otherwise than the word is.
pub const SPELLING_CORRECTION: u16 = 1;

/// A digit read in kana, which a path's text writes in kanji unless a suffix
/// follows it ([`mark_digits`]).
pub const DIGIT: u16 = 2;

/// A suffix of nouns, after which a digit read in kana stays an Arabic numeral.
pub const SUFFIX: u16 = 4;

/// The part of speech of Mozc's Arabic numerals, as `id.def` names it.
pub const ARABIC_NUMERAL: &str = "名詞,数,アラビア数字,*,*,*,*";

/// How the parts of speech of Mozc's suffixes of nouns begin.
pub const NOUN_SUFFIX: &str = "名詞,接尾,";

/// Why the files could not be read.
#[derive(Debug)]
pub enum Error {
    /// A file that could not be read.
    Io(PathBuf, std::io::Error),
    /// A line that is not what the file holds.
    Line(String, usize, String),
    /// What the files should name and do not.
    Missing(String),
}

impl std::fmt::Display for Error {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(path, error) => write!(formatter, "{}: {error}", path.display()),
            Self::Line(file, line, message) => write!(formatter, "{file}, line {line}: {message}"),
            Self::Missing(message) => formatter.write_str(message),
        }
    }
}

impl std::error::Error for Error {}

/// The costs of the connections between classes: that of a word of right id
/// `r` followed by one of left id `l` at `r * ids + l`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Matrix {
    /// The number of ids, the beginning and end of a sentence's 0 among them.
    pub ids: u16,
    /// The costs, row by row of right ids.
    pub costs: Vec<u16>,
}

/// Mozc's dictionary as it is read: its words, the costs of their connections,
/// and the notices that go with them.
#[derive(Debug, Clone)]
pub struct Source {
    /// The words, in the order of the files.
    pub words: Vec<Word>,
    /// The costs of the connections.
    pub matrix: Matrix,
    /// The part of speech of each id.
    pub parts_of_speech: Vec<String>,
    /// `README.txt`, then `LICENSE`.
    pub notice: String,
}

impl Source {
    /// The id of a part of speech, as `id.def` names it.
    #[must_use]
    pub fn id_of(&self, part_of_speech: &str) -> Option<u16> {
        self.parts_of_speech
            .iter()
            .position(|name| name == part_of_speech)
            .and_then(|id| u16::try_from(id).ok())
    }
}

pub(crate) fn read_text(directory: &Path, name: &str) -> Result<String, Error> {
    let path = directory.join(name);
    std::fs::read_to_string(&path).map_err(|error| Error::Io(path, error))
}

/// The dictionary in a directory of Mozc's files, as
/// `scripts/fetch_datasets.py fetch mozc-dictionary` lays them out.
///
/// # Errors
///
/// Where a file cannot be read, or holds a line it should not.
pub fn read(directory: &Path) -> Result<Source, Error> {
    let matrix = read_matrix(&read_text(directory, "connection_single_column.txt")?)?;
    let mut words = Vec::new();
    for number in 0..10 {
        let name = format!("dictionary{number:02}.txt");
        read_words(&read_text(directory, &name)?, &name, matrix.ids, &mut words)?;
    }
    let parts_of_speech = read_parts_of_speech(&read_text(directory, "id.def")?, matrix.ids)?;
    let notice = format!(
        "{}\n\n{}",
        read_text(directory, "README.txt")?.trim_end(),
        read_text(directory, "LICENSE")?.trim_end()
    );
    Ok(Source {
        words,
        matrix,
        parts_of_speech,
        notice,
    })
}

/// Marks the digits read in kana ([`DIGIT`]) and the suffixes of nouns
/// ([`SUFFIX`]), for a path's text to write the digits as Mozc shows them, and
/// returns how many words of each it marked.
///
/// Mozc's dictionary has the digits read in kana -- いち for 1, ぜろ and れい for
/// 0 -- as cheap words of its class of Arabic numerals, which Mozc's converter
/// does not show as they are where a segment's content is the digit alone (with
/// a particle, say): its number rewriter puts the number in kanji in its place
/// (一, and 零 for zero), and the Arabic numeral among the candidates below it
/// (`KANJI_FIRST` in `src/rewriter/number_rewriter.cc`). Where a suffix follows
/// the digit in its segment -- 1年, 1人前 -- the content is not the digit alone,
/// and the Arabic numeral stays. KIRAL has no segments: a digit is written in
/// kanji unless the word after it is a suffix of nouns, a word whose class on
/// its left is one of `名詞,接尾`'s.
pub fn mark_digits(source: &mut Source) -> (usize, usize) {
    let arabic = source.id_of(ARABIC_NUMERAL);
    let suffixes: Vec<bool> = source
        .parts_of_speech
        .iter()
        .map(|name| name.starts_with(NOUN_SUFFIX))
        .collect();
    let (mut digits, mut suffix_words) = (0, 0);
    for word in &mut source.words {
        if Some(word.lid) == arabic
            && word.rid == word.lid
            && kanji_digit(&word.surface).is_some()
            && !word.reading.bytes().any(|byte| byte.is_ascii_digit())
        {
            word.flags |= DIGIT;
            digits += 1;
        }
        if suffixes.get(usize::from(word.lid)).copied().unwrap_or(false) {
            word.flags |= SUFFIX;
            suffix_words += 1;
        }
    }
    (digits, suffix_words)
}

/// A digit in kanji, as Mozc's number rewriter writes it: 零 for 0, 一 to 九;
/// none for a surface that is not one digit.
#[must_use]
pub fn kanji_digit(surface: &str) -> Option<&'static str> {
    const KANJI: [&str; 10] = ["零", "一", "二", "三", "四", "五", "六", "七", "八", "九"];
    match *surface.as_bytes() {
        [digit @ b'0'..=b'9'] => Some(KANJI[usize::from(digit - b'0')]),
        _ => None,
    }
}

/// The part of speech of each id, from `id.def`'s text: a line for each id,
/// in their order, its number, a space and its name.
///
/// # Errors
///
/// Where a line is not the next id and a name, or the ids are not those of the
/// matrix.
pub fn read_parts_of_speech(text: &str, ids: u16) -> Result<Vec<String>, Error> {
    let mut names = Vec::with_capacity(usize::from(ids));
    for (number, line) in text.lines().enumerate().filter(|(_, line)| !line.is_empty()) {
        let wrong = |message: &str| Error::Line("id.def".to_owned(), number + 1, message.to_owned());
        let (id, name) = line.split_once(' ').ok_or_else(|| wrong("not an id and a name"))?;
        if id.parse::<usize>().ok() != Some(names.len()) {
            return Err(wrong("not the next id"));
        }
        names.push(name.to_owned());
    }
    if names.len() != usize::from(ids) {
        return Err(Error::Line(
            "id.def".to_owned(),
            names.len(),
            format!("{} ids, not the matrix's {ids}", names.len()),
        ));
    }
    Ok(names)
}

/// The words of a dictionary file's text, appended to `words`: their ids
/// checked to be below `ids`.
///
/// # Errors
///
/// Where a line has other fields than a word's.
pub fn read_words(text: &str, file: &str, ids: u16, words: &mut Vec<Word>) -> Result<(), Error> {
    for (number, line) in text.lines().enumerate() {
        let wrong = |message: &str| Error::Line(file.to_owned(), number + 1, message.to_owned());
        if line.is_empty() {
            continue;
        }
        let fields: Vec<&str> = line.split('\t').collect();
        let (reading, lid, rid, cost, surface, flags) = match fields[..] {
            [reading, lid, rid, cost, surface] => (reading, lid, rid, cost, surface, 0),
            [reading, lid, rid, cost, surface, "SPELLING_CORRECTION"] => {
                (reading, lid, rid, cost, surface, SPELLING_CORRECTION)
            }
            _ => return Err(wrong("not a reading, two ids, a cost and a surface")),
        };
        let id = |text: &str| text.parse::<u16>().ok().filter(|&id| id < ids);
        let (Some(lid), Some(rid)) = (id(lid), id(rid)) else {
            return Err(wrong("an id out of the matrix's"));
        };
        let cost = cost
            .parse::<u16>()
            .map_err(|_| wrong("a cost that is not a number of 16 bits"))?;
        if reading.is_empty() || surface.is_empty() {
            return Err(wrong("an empty reading or surface"));
        }
        words.push(Word {
            reading: reading.to_owned(),
            surface: surface.to_owned(),
            lid,
            rid,
            cost,
            flags,
        });
    }
    Ok(())
}

/// The matrix of `connection_single_column.txt`'s text.
///
/// # Errors
///
/// Where it does not hold the square of its first line's number of costs.
pub fn read_matrix(text: &str) -> Result<Matrix, Error> {
    let file = "connection_single_column.txt";
    let mut lines = text.lines();
    let ids = lines
        .next()
        .and_then(|line| line.trim().parse::<u16>().ok())
        .filter(|&ids| ids > 0)
        .ok_or_else(|| Error::Line(file.to_owned(), 1, "no number of ids".to_owned()))?;
    let count = usize::from(ids) * usize::from(ids);
    let mut costs = Vec::with_capacity(count);
    for (number, line) in lines.enumerate() {
        let cost = line.trim().parse::<u16>().map_err(|_| {
            Error::Line(
                file.to_owned(),
                number + 2,
                "a cost that is not a number of 16 bits".to_owned(),
            )
        })?;
        costs.push(cost);
    }
    if costs.len() != count {
        return Err(Error::Line(
            file.to_owned(),
            costs.len() + 1,
            format!("{} costs, not the {count} of {ids} ids", costs.len()),
        ));
    }
    // The beginning of a sentence followed by its end costs nothing.
    costs[0] = 0;
    Ok(Matrix { ids, costs })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn words_and_matrices_are_read() {
        let mut words = Vec::new();
        read_words(
            "かな\t3\t4\t500\t仮名\nばんしあ\t1\t1\t9000\tヴァンシア\tSPELLING_CORRECTION\n",
            "d",
            5,
            &mut words,
        )
        .expect("words");
        assert_eq!(words.len(), 2);
        assert_eq!(
            (words[0].lid, words[0].rid, words[0].cost, words[0].flags),
            (3, 4, 500, 0)
        );
        assert_eq!(words[1].flags, SPELLING_CORRECTION);
        for line in [
            "かな\t3\t4\t500",
            "かな\t5\t4\t500\t仮名",
            "かな\t3\t4\t-1\t仮名",
            "\t3\t4\t5\tx",
            "a\t1\t1\t1\tb\tOTHER",
        ] {
            assert!(read_words(line, "d", 5, &mut Vec::new()).is_err(), "{line}");
        }
        let names = read_parts_of_speech("0 BOS/EOS\n1 名詞,一般\n", 2).expect("names");
        assert_eq!(names, ["BOS/EOS", "名詞,一般"]);
        assert!(read_parts_of_speech("0 a\n2 b\n", 2).is_err());
        assert!(read_parts_of_speech("0 a\n", 2).is_err());
        let matrix = read_matrix("2\n7\n1\n2\n3\n").expect("a matrix");
        assert_eq!(
            matrix,
            Matrix {
                ids: 2,
                costs: vec![0, 1, 2, 3]
            }
        );
        assert!(read_matrix("2\n1\n2\n3\n").is_err());
        assert!(read_matrix("x\n").is_err());
    }

    #[test]
    fn digits_read_in_kana_and_suffixes_are_marked() {
        let word = |reading: &str, surface: &str, id: u16| Word {
            reading: reading.to_owned(),
            surface: surface.to_owned(),
            lid: id,
            rid: id,
            cost: 753,
            flags: 0,
        };
        let mut source = Source {
            words: vec![
                word("いち", "1", 1),
                word("ぜろ", "0", 1),
                word("いちねんご", "1年後", 1),
                word("1", "1", 1),
                word("じゅう", "10", 1),
                word("いち", "1", 2),
                word("ねん", "年", 3),
            ],
            matrix: Matrix {
                ids: 4,
                costs: vec![0; 16],
            },
            parts_of_speech: [
                "BOS/EOS",
                ARABIC_NUMERAL,
                "名詞,数,漢数字,*,*,*,*",
                "名詞,接尾,助数詞,*,*,*,*",
            ]
            .map(str::to_owned)
            .to_vec(),
            notice: String::new(),
        };
        assert_eq!(mark_digits(&mut source), (2, 1));
        let flags: Vec<u16> = source.words.iter().map(|word| word.flags).collect();
        assert_eq!(flags, [DIGIT, DIGIT, 0, 0, 0, 0, SUFFIX]);
        assert_eq!(kanji_digit("0"), Some("零"));
        assert_eq!(kanji_digit("9"), Some("九"));
        assert_eq!(kanji_digit("10"), None);
        assert_eq!(kanji_digit("一"), None);
    }
}
