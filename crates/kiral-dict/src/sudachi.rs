// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later
//
// This file is part of KIRAL. The SPDX lines name its copyright holder and its
// license, in the machine-readable form of the REUSE specification: the GNU
// General Public License, version 3 or any later version
// (LICENSES/GPL-3.0-or-later.txt).

//! SudachiDict's word lists, read for the nouns Mozc's dictionary lacks.
//!
//! `small_lex.csv`, `core_lex.csv` and `notcore_lex.csv` hold a word a line,
//! in fields separated by commas: the form it is looked up by; the ids of its
//! classes on its left and its right, and its cost, all three in Sudachi's own
//! matrix; its headword, where it is written otherwise than the first field;
//! the six fields of its part of speech, in UniDic's terms; its reading, in
//! katakana; and fields KIRAL does not read -- its normalized and dictionary
//! forms, its splits, its synonyms. A field that holds a comma is quoted, and a
//! character may be escaped as `\u` and four hexadecimal digits (`\u002c`). A
//! left id of -1 marks a word that is only a part of others' splits, never
//! looked up. A first line naming the fields is skipped.
//!
//! Sudachi's ids and costs are those of its own matrix, of UniDic's classes,
//! which Mozc's matrix does not know, and its costs do not follow Mozc's: over
//! the 258,798 nouns of the small and core lists that Mozc's dictionary has
//! too, Spearman's rank correlation between the two is 0.07. A word taken from
//! Sudachi is given the class of Mozc's that its part of speech corresponds to
//! ([`CLASSES`]), and the cost under which nine in ten of Mozc's words of that
//! class lie ([`PERCENTILE`]): a word Mozc's dictionary lacks is taken as a
//! rare one of its class. Only nouns are taken -- the words a dictionary lacks
//! are names, places, terms and loanwords, and inflecting words, which Mozc's
//! dictionary has, would need UniDic's inflections mapped to IPAdic's -- and
//! only those whose reading and surface Mozc's dictionary does not have
//! together, one entry for each new pair.

use std::borrow::Cow;
use std::collections::HashSet;
use std::path::Path;

use crate::file::LONGEST_SURFACE;
use crate::mozc::{Error, Source, read_text};
use crate::{Word, normalize};

/// Sudachi's classes of nouns, by the first four fields of their part of
/// speech, and the class of Mozc's each is taken as, as `id.def` names it:
/// nouns that take する, adjectival or not, as Mozc's サ変接続, adjectival
/// nouns as its 形容動詞語幹, nouns that may be counters as common nouns, places
/// as its 地域. Sudachi's numerals and its stems of auxiliary verbs are left to
/// Mozc's own.
pub const CLASSES: [(&str, &str); 12] = [
    ("名詞,普通名詞,一般,*", "名詞,一般,*,*,*,*,*"),
    ("名詞,普通名詞,サ変可能,*", "名詞,サ変接続,*,*,*,*,*"),
    ("名詞,普通名詞,サ変形状詞可能,*", "名詞,サ変接続,*,*,*,*,*"),
    ("名詞,普通名詞,形状詞可能,*", "名詞,形容動詞語幹,*,*,*,*,*"),
    ("名詞,普通名詞,副詞可能,*", "名詞,副詞可能,*,*,*,*,*"),
    ("名詞,普通名詞,助数詞可能,*", "名詞,一般,*,*,*,*,*"),
    ("名詞,固有名詞,一般,*", "名詞,固有名詞,一般,*,*,*,*"),
    ("名詞,固有名詞,人名,一般", "名詞,固有名詞,人名,一般,*,*,*"),
    ("名詞,固有名詞,人名,名", "名詞,固有名詞,人名,名,*,*,*"),
    ("名詞,固有名詞,人名,姓", "名詞,固有名詞,人名,姓,*,*,*"),
    ("名詞,固有名詞,地名,一般", "名詞,固有名詞,地域,一般,*,*,*"),
    ("名詞,固有名詞,地名,国", "名詞,固有名詞,地域,国,*,*,*"),
];

/// The lists of Sudachi's core dictionary: its small one and the words its
/// core one adds. `notcore_lex.csv` makes the full one.
pub const CORE: [&str; 2] = ["small_lex.csv", "core_lex.csv"];

/// The percentile, by the nearest rank, of the costs of Mozc's words of a class
/// (their classes on the left and the right both it) that a word from Sudachi
/// is given.
pub const PERCENTILE: usize = 90;

/// A class of Mozc's, as the words from Sudachi are given it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Class {
    /// Its id.
    pub id: u16,
    /// The cost the words are given.
    pub cost: u16,
    /// The number of words given it.
    pub words: usize,
}

/// The nouns of SudachiDict's lists that Mozc's dictionary lacks.
#[derive(Debug, Clone)]
pub struct Nouns {
    /// The words, in the order of the lists.
    pub words: Vec<Word>,
    /// The class of Mozc's of each of [`CLASSES`].
    pub classes: Vec<Class>,
    /// The words the lists hold, the first lines naming the fields not counted.
    pub lines: usize,
    /// The nouns of [`CLASSES`] whose reading and surface Mozc's dictionary, or
    /// a line before, has.
    pub repeated: usize,
    /// The license section of `README.md`, then `LEGAL` and `LICENSE-2.0.txt`.
    pub notice: String,
}

/// The nouns of SudachiDict's `lists` in a directory, as
/// `scripts/fetch_datasets.py fetch sudachi-dictionary` lays them out, that
/// Mozc's dictionary lacks, in Mozc's classes.
///
/// # Errors
///
/// Where a file cannot be read, a line has fewer than twelve fields, or Mozc's
/// dictionary has no class of [`CLASSES`] or no words of one.
pub fn read(directory: &Path, lists: &[&str], mozc: &Source) -> Result<Nouns, Error> {
    let mut nouns = Nouns {
        words: Vec::new(),
        classes: classes(mozc)?,
        lines: 0,
        repeated: 0,
        notice: notice(directory)?,
    };
    let mut seen: HashSet<String> = mozc
        .words
        .iter()
        .map(|word| pair(&normalize::lookup_form(&word.reading), &word.surface))
        .collect();
    for list in lists {
        read_list(&read_text(directory, list)?, list, &mut seen, &mut nouns)?;
    }
    Ok(nouns)
}

/// The class of Mozc's of each of [`CLASSES`], with the cost its words take.
fn classes(mozc: &Source) -> Result<Vec<Class>, Error> {
    CLASSES
        .iter()
        .map(|&(_, name)| {
            let id = mozc
                .id_of(name)
                .ok_or_else(|| Error::Missing(format!("id.def names no {name}")))?;
            let cost = cost_of_class(&mozc.words, id)
                .ok_or_else(|| Error::Missing(format!("Mozc's dictionary has no words of {name}")))?;
            Ok(Class { id, cost, words: 0 })
        })
        .collect()
}

/// The cost of [`PERCENTILE`] among the words of a class; none where it has
/// none.
#[must_use]
pub fn cost_of_class(words: &[Word], id: u16) -> Option<u16> {
    let mut costs: Vec<u16> = words
        .iter()
        .filter(|word| word.lid == id && word.rid == id)
        .map(|word| word.cost)
        .collect();
    costs.sort_unstable();
    let rank = (costs.len() * PERCENTILE).div_ceil(100);
    costs.get(rank.checked_sub(1)?).copied()
}

/// The words of a list's text that are nouns of [`CLASSES`] and new, appended
/// to `nouns`, their pairs of a reading and a surface added to `seen`.
///
/// # Errors
///
/// Where a line has fewer than twelve fields.
fn read_list(text: &str, list: &str, seen: &mut HashSet<String>, nouns: &mut Nouns) -> Result<(), Error> {
    for (number, line) in text.lines().enumerate() {
        if line.is_empty() || (number == 0 && line.starts_with("IndexForm,")) {
            continue;
        }
        nouns.lines += 1;
        let fields = fields(line, 12)
            .ok_or_else(|| Error::Line(list.to_owned(), number + 1, "fewer than twelve fields".to_owned()))?;
        let Some(class) = class_of(&fields) else {
            continue;
        };
        let Some(reading) = reading(&fields[11]) else {
            continue;
        };
        let surface = unescape(if fields[4].is_empty() { &fields[0] } else { &fields[4] });
        if surface.is_empty() || surface.len() > LONGEST_SURFACE || surface.chars().any(char::is_control) {
            continue;
        }
        if !seen.insert(pair(&reading, &surface)) {
            nouns.repeated += 1;
            continue;
        }
        let Some(class) = nouns.classes.get_mut(class) else {
            continue;
        };
        class.words += 1;
        nouns.words.push(Word {
            reading,
            surface: surface.into_owned(),
            lid: class.id,
            rid: class.id,
            cost: class.cost,
            flags: 0,
        });
    }
    Ok(())
}

/// A reading and a surface, as one string to look up.
fn pair(reading: &str, surface: &str) -> String {
    format!("{reading}\t{surface}")
}

/// The number in [`CLASSES`] of the class of a line's word: none for a word
/// that is never looked up, and one of another part of speech.
fn class_of(fields: &[Cow<'_, str>]) -> Option<usize> {
    if fields.get(1)? == "-1" {
        return None;
    }
    let part_of_speech = fields.get(5..9)?;
    CLASSES.iter().position(|&(sudachi, _)| {
        sudachi
            .split(',')
            .zip(part_of_speech)
            .all(|(expected, field)| expected == field)
    })
}

/// A reading of Sudachi's in its lookup form: none unless it is all kana and
/// the prolonged sound mark, as one types it.
fn reading(field: &str) -> Option<String> {
    let reading = normalize::lookup_form(&unescape(field));
    let typed = !reading.is_empty()
        && reading
            .chars()
            .all(|character| matches!(character, '\u{3041}'..='\u{3096}' | 'ー'));
    typed.then_some(reading)
}

/// The first `count` fields of a line of comma-separated values, unquoted (a
/// quote within a quoted field is two quotes): none where the line has fewer,
/// or a quote is not closed where a field ends.
#[must_use]
pub fn fields(line: &str, count: usize) -> Option<Vec<Cow<'_, str>>> {
    let mut fields = Vec::with_capacity(count);
    let mut rest = Some(line);
    while fields.len() < count {
        let text = rest?;
        let (field, after) = if let Some(quoted) = text.strip_prefix('"') {
            let (field, after) = unquote(quoted)?;
            (Cow::Owned(field), after)
        } else if let Some((field, after)) = text.split_once(',') {
            (Cow::Borrowed(field), Some(after))
        } else {
            (Cow::Borrowed(text), None)
        };
        fields.push(field);
        rest = after;
    }
    Some(fields)
}

/// A quoted field, its opening quote taken: the field, and what follows its
/// comma, or none at the end of the line.
fn unquote(text: &str) -> Option<(String, Option<&str>)> {
    let mut field = String::new();
    let mut tail = text;
    loop {
        let (part, after) = tail.split_once('"')?;
        field.push_str(part);
        if let Some(after) = after.strip_prefix('"') {
            field.push('"');
            tail = after;
        } else {
            tail = after;
            break;
        }
    }
    if tail.is_empty() {
        Some((field, None))
    } else {
        tail.strip_prefix(',').map(|after| (field, Some(after)))
    }
}

/// A field with Sudachi's escapes, `\u` and four hexadecimal digits, as the
/// characters they stand for; an escape that stands for none is left as it is.
#[must_use]
pub fn unescape(field: &str) -> Cow<'_, str> {
    if !field.contains("\\u") {
        return Cow::Borrowed(field);
    }
    let mut text = String::with_capacity(field.len());
    let mut rest = field;
    while let Some((before, after)) = rest.split_once("\\u") {
        text.push_str(before);
        let character = after
            .get(..4)
            .filter(|digits| digits.bytes().all(|byte| byte.is_ascii_hexdigit()))
            .and_then(|digits| u32::from_str_radix(digits, 16).ok())
            .and_then(char::from_u32);
        if let Some(character) = character {
            text.push(character);
            rest = after.get(4..).unwrap_or_default();
        } else {
            text.push_str("\\u");
            rest = after;
        }
    }
    text.push_str(rest);
    Cow::Owned(text)
}

/// The notices of SudachiDict: the license section of `README.md`, with the
/// copyright of its makers, `LEGAL`, on what it holds of UniDic and NEologd,
/// and the Apache License's text.
fn notice(directory: &Path) -> Result<String, Error> {
    let readme = read_text(directory, "README.md")?;
    let (_, section) = readme
        .split_once("## Licenses")
        .ok_or_else(|| Error::Missing("SudachiDict's README.md has no section ## Licenses".to_owned()))?;
    let section = section.split("\n## ").next().unwrap_or_default();
    let licenses: Vec<&str> = section.lines().filter(|line| !line.starts_with("```")).collect();
    Ok(format!(
        "SudachiDict\n\n{}\n\n{}\n\n{}",
        licenses.join("\n").trim(),
        read_text(directory, "LEGAL")?.trim_end(),
        read_text(directory, "LICENSE-2.0.txt")?.trim_end()
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mozc::Matrix;

    #[test]
    fn fields_are_split_and_unquoted() {
        let line = r#"a,"b,c","d""e",,f"#;
        let all = fields(line, 5).expect("five fields");
        assert_eq!(all, ["a", "b,c", "d\"e", "", "f"]);
        assert_eq!(fields(line, 3).expect("three").len(), 3);
        assert!(fields(line, 6).is_none());
        assert!(fields(r#"a,"b"#, 2).is_none());
        assert!(fields(r#"a,"b"c,d"#, 3).is_none());
        assert_eq!(fields("", 1).expect("one"), [""]);
        assert_eq!(unescape(r"& Co.\u002C Ltd.\u0028x\u0029"), "& Co., Ltd.(x)");
        assert_eq!(unescape(r"\u00zz\u"), r"\u00zz\u");
        assert_eq!(unescape(r"\u+0412"), r"\u+0412");
        assert_eq!(unescape(r"\ud800x"), r"\ud800x");
        assert_eq!(reading("カケイ").as_deref(), Some("かけい"));
        assert_eq!(reading("ヴァー").as_deref(), Some("ゔぁー"));
        assert_eq!(reading("CO.LTD"), None);
        assert_eq!(reading(""), None);
    }

    fn source() -> Source {
        let parts_of_speech: Vec<String> = ["BOS/EOS"]
            .into_iter()
            .chain(CLASSES.iter().map(|&(_, mozc)| mozc))
            .map(str::to_owned)
            .collect();
        let ids = u16::try_from(parts_of_speech.len()).expect("small");
        let mut words: Vec<Word> = (1..ids)
            .flat_map(|id| {
                (1..=10).map(move |cost| Word {
                    reading: "ほか".to_owned(),
                    surface: format!("他{id}"),
                    lid: id,
                    rid: id,
                    cost: cost * 100,
                    flags: 0,
                })
            })
            .collect();
        words.push(Word {
            reading: "かけい".to_owned(),
            surface: "筧".to_owned(),
            lid: 1,
            rid: 1,
            cost: 50,
            flags: 0,
        });
        Source {
            words,
            matrix: Matrix {
                ids,
                costs: vec![0; usize::from(ids) * usize::from(ids)],
            },
            parts_of_speech,
            notice: String::new(),
        }
    }

    #[test]
    fn new_nouns_take_mozc_classes() {
        let mozc = source();
        let mut nouns = Nouns {
            words: Vec::new(),
            classes: classes(&mozc).expect("classes"),
            lines: 0,
            repeated: 0,
            notice: String::new(),
        };
        // The ninth of ten costs, 100 to 1000, of each class's own words.
        assert!(nouns.classes.iter().all(|class| class.cost == 900));
        let mut seen: HashSet<String> = mozc
            .words
            .iter()
            .map(|word| pair(&normalize::lookup_form(&word.reading), &word.surface))
            .collect();
        let text = "IndexForm,LeftId,RightId,Cost,Headword,POS1,POS2,POS3,POS4,POS5,POS6,ReadingForm,Rest\n\
            筧,10,10,5000,,名詞,固有名詞,人名,姓,*,*,カケイ,,,,,,,\n\
            筧,10,10,5000,,名詞,普通名詞,一般,*,*,*,カケイ,,,,,,,\n\
            算盤,10,10,5000,,名詞,普通名詞,一般,*,*,*,ソロバン,,,,,,,\n\
            算盤,10,10,5000,,名詞,普通名詞,サ変可能,*,*,*,ソロバン,,,,,,,\n\
            てすと,-1,-1,0,,名詞,普通名詞,一般,*,*,*,テスト,,,,,,,\n\
            走る,10,10,5000,,動詞,一般,*,*,五段-ラ行,終止形-一般,ハシル,,,,,,,\n\
            & co.\\u002c ltd.,10,10,5000,& Co.\\u002C Ltd.,名詞,固有名詞,一般,*,*,*,アンドカンパニー,\"x,y\",,,,,,\n\
            abc,10,10,5000,ABC,名詞,固有名詞,一般,*,*,*,ABC,,,,,,,\n";
        read_list(text, "t.csv", &mut seen, &mut nouns).expect("a list");
        assert_eq!(nouns.lines, 8);
        // 筧 twice (Mozc's), 算盤 again under another class.
        assert_eq!(nouns.repeated, 3);
        let words: Vec<(&str, &str, u16, u16)> = nouns
            .words
            .iter()
            .map(|word| (word.reading.as_str(), word.surface.as_str(), word.lid, word.cost))
            .collect();
        assert_eq!(
            words,
            [
                ("そろばん", "算盤", 1, 900),
                ("あんどかんぱにー", "& Co., Ltd.", 7, 900)
            ]
        );
        assert_eq!(nouns.classes[0].words, 1);
        assert!(read_list("a,b,c\n", "t.csv", &mut seen, &mut nouns).is_err());
    }

    #[test]
    fn any_line_is_read_or_refused() {
        let mozc = source();
        let mut nouns = Nouns {
            words: Vec::new(),
            classes: classes(&mozc).expect("classes"),
            lines: 0,
            repeated: 0,
            notice: String::new(),
        };
        let mut seen = HashSet::new();
        let line = "\"&,\"\"\",10,10,5000,\\u0041\\u00,名詞,普通名詞,一般,*,*,*,カケイ,\"x\",,";
        let pieces: Vec<char> = line.chars().chain(['"', ',', '\\', 'u', '0', 'ア', '\n']).collect();
        let mut state = 0x2545_f491_4f6c_dd1d_u64;
        for _ in 0..3000 {
            let mut text: Vec<char> = line.chars().collect();
            for _ in 0..4 {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                let at = usize::try_from(state % 64).expect("small") % (text.len() + 1);
                let piece = pieces[usize::try_from(state >> 40).expect("small") % pieces.len()];
                match state % 3 {
                    0 if at < text.len() => {
                        text.remove(at);
                    }
                    1 if at < text.len() => text[at] = piece,
                    _ => text.insert(at, piece),
                }
            }
            let text: String = text.into_iter().collect();
            let _ = read_list(&text, "t.csv", &mut seen, &mut nouns);
        }
        assert!(nouns.words.iter().all(|word| !word.surface.is_empty()));
    }
}
