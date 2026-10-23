// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later
//
// This file is part of KIRAL. The SPDX lines name its copyright holder and its
// license, in the machine-readable form of the REUSE specification: the GNU
// General Public License, version 3 or any later version
// (LICENSES/GPL-3.0-or-later.txt).

//! Numbers read in kana: the stretches of a reading that say a number, and the
//! ways it is written.
//!
//! A number is read as Japanese says it: a section below ten thousand -- its
//! thousands, hundreds, tens and ones, each a digit and its unit, the digit 1
//! left unsaid before じゅう, ひゃく and せん -- after sections of ちょう, おく and
//! まん, the largest first, with the sounds that change: さんびゃく, ろっぴゃく,
//! はっぴゃく, さんぜん, はっせん, いっちょう. Each word of a number is a numeral of
//! its own, 四 for よん and 十 for じゅう, and the numerals one after another are
//! the number written in kanji ([`Number::kanji`]); written in Arabic numerals
//! it is its value ([`Number::arabic`]), and from ten thousand on its sections
//! in Arabic numerals with their units in kanji, as 100万 is written
//! ([`Number::mixed`]). A digit alone is no number here: the dictionary has the
//! digits read in kana as words of their own.

/// A word of a number: the stretch of the reading it reads, and its numeral.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Numeral {
    /// The position of its first character.
    pub begin: usize,
    /// The position after its last.
    pub end: usize,
    /// The numeral that writes it.
    pub kanji: char,
}

/// A number that a stretch of a reading says.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Number {
    /// The position after its last character.
    pub end: usize,
    /// Its value.
    pub value: u64,
    /// Its words, in order.
    pub numerals: Vec<Numeral>,
}

impl Number {
    /// The number in kanji, as it is said: 四十五, 千九百九十一, 二千五, 一万.
    #[must_use]
    pub fn kanji(&self) -> String {
        self.numerals.iter().map(|numeral| numeral.kanji).collect()
    }

    /// The number in Arabic numerals: 45, 1991.
    #[must_use]
    pub fn arabic(&self) -> String {
        self.value.to_string()
    }

    /// The number's sections in Arabic numerals and their units of ten
    /// thousand and more in kanji, a section of nothing left out: 100万,
    /// 1億2000万; none below ten thousand.
    #[must_use]
    pub fn mixed(&self) -> Option<String> {
        if self.value < 10_000 {
            return None;
        }
        let mut text = String::new();
        let mut rest = self.value;
        for (unit, kanji) in BIG_UNITS.map(|(unit, _, kanji)| (unit, kanji)) {
            let section = rest / unit;
            rest %= unit;
            if section > 0 {
                text.push_str(&section.to_string());
                text.push(kanji);
            }
        }
        if rest > 0 {
            text.push_str(&rest.to_string());
        }
        Some(text)
    }
}

/// The digits as the ones of a section says them: the kana, the value and the
/// numeral.
const ONES: [(&str, u64, char); 10] = [
    ("いち", 1, '一'),
    ("に", 2, '二'),
    ("さん", 3, '三'),
    ("よん", 4, '四'),
    ("ご", 5, '五'),
    ("ろく", 6, '六'),
    ("なな", 7, '七'),
    ("しち", 7, '七'),
    ("はち", 8, '八'),
    ("きゅう", 9, '九'),
];

/// A digit said before a unit: its kana, its value, its numeral, and the unit
/// as it is said after it.
#[derive(Debug, Clone, Copy)]
struct Before {
    kana: &'static str,
    value: u64,
    kanji: char,
    unit: &'static str,
}

/// A unit within a section: its value, its numeral, its kana said alone, and
/// the digits said before it.
#[derive(Debug, Clone, Copy)]
struct Unit {
    value: u64,
    kanji: char,
    alone: &'static str,
    after: &'static [Before],
}

const fn before(kana: &'static str, value: u64, kanji: char, unit: &'static str) -> Before {
    Before {
        kana,
        value,
        kanji,
        unit,
    }
}

/// The units within a section, the largest first.
const UNITS: [Unit; 3] = [
    Unit {
        value: 1000,
        kanji: '千',
        alone: "せん",
        after: &[
            before("に", 2, '二', "せん"),
            before("さん", 3, '三', "ぜん"),
            before("よん", 4, '四', "せん"),
            before("ご", 5, '五', "せん"),
            before("ろく", 6, '六', "せん"),
            before("なな", 7, '七', "せん"),
            before("はっ", 8, '八', "せん"),
            before("きゅう", 9, '九', "せん"),
        ],
    },
    Unit {
        value: 100,
        kanji: '百',
        alone: "ひゃく",
        after: &[
            before("に", 2, '二', "ひゃく"),
            before("さん", 3, '三', "びゃく"),
            before("よん", 4, '四', "ひゃく"),
            before("ご", 5, '五', "ひゃく"),
            before("ろっ", 6, '六', "ぴゃく"),
            before("なな", 7, '七', "ひゃく"),
            before("はっ", 8, '八', "ぴゃく"),
            before("きゅう", 9, '九', "ひゃく"),
        ],
    },
    Unit {
        value: 10,
        kanji: '十',
        alone: "じゅう",
        after: &[
            before("に", 2, '二', "じゅう"),
            before("さん", 3, '三', "じゅう"),
            before("よん", 4, '四', "じゅう"),
            before("ご", 5, '五', "じゅう"),
            before("ろく", 6, '六', "じゅう"),
            before("なな", 7, '七', "じゅう"),
            before("しち", 7, '七', "じゅう"),
            before("はち", 8, '八', "じゅう"),
            before("きゅう", 9, '九', "じゅう"),
        ],
    },
];

/// The units of sections, the largest first: the value, the unit as it is
/// said, and the numeral.
const BIG_UNITS: [(u64, &str, char); 3] = [
    (1_000_000_000_000, "ちょう", '兆'),
    (100_000_000, "おく", '億'),
    (10_000, "まん", '万'),
];

/// The position after `kana` where the reading has it at `at`.
fn word(reading: &[char], at: usize, kana: &str) -> Option<usize> {
    let mut end = at;
    for character in kana.chars() {
        if reading.get(end) != Some(&character) {
            return None;
        }
        end += 1;
    }
    Some(end)
}

/// A part of a number: where it ends, its value, and its numerals.
type Part = (usize, u64, Vec<Numeral>);

/// The ways a unit of a section is said at a position: alone, or after a digit.
fn terms(reading: &[char], at: usize, unit: &Unit) -> Vec<Part> {
    let mut found = Vec::new();
    if let Some(end) = word(reading, at, unit.alone) {
        found.push((
            end,
            unit.value,
            vec![Numeral {
                begin: at,
                end,
                kanji: unit.kanji,
            }],
        ));
    }
    for digit in unit.after {
        let Some(middle) = word(reading, at, digit.kana) else {
            continue;
        };
        if let Some(end) = word(reading, middle, digit.unit) {
            found.push((
                end,
                digit.value * unit.value,
                vec![
                    Numeral {
                        begin: at,
                        end: middle,
                        kanji: digit.kanji,
                    },
                    Numeral {
                        begin: middle,
                        end,
                        kanji: unit.kanji,
                    },
                ],
            ));
        }
    }
    found
}

/// The sections said from a position: thousands, hundreds, tens and ones,
/// each there or not, one at least.
fn sections(reading: &[char], at: usize) -> Vec<Part> {
    let mut parts: Vec<Part> = vec![(at, 0, Vec::new())];
    for unit in &UNITS {
        let mut longer = parts.clone();
        for (end, value, numerals) in &parts {
            for (term_end, term, term_numerals) in terms(reading, *end, unit) {
                longer.push((term_end, value + term, [&numerals[..], &term_numerals].concat()));
            }
        }
        parts = longer;
    }
    let mut found = Vec::new();
    for (end, value, numerals) in parts {
        for &(digit, one, kanji) in &ONES {
            if let Some(after) = word(reading, end, digit) {
                let mut numerals = numerals.clone();
                numerals.push(Numeral {
                    begin: end,
                    end: after,
                    kanji,
                });
                found.push((after, value + one, numerals));
            }
        }
        if !numerals.is_empty() {
            found.push((end, value, numerals));
        }
    }
    found
}

/// The units as they are said, the sounds that change among them.
const UNIT_WORDS: [&str; 9] = [
    "じゅう",
    "ひゃく",
    "びゃく",
    "ぴゃく",
    "せん",
    "ぜん",
    "まん",
    "おく",
    "ちょう",
];

/// The digits as they are said before a unit, the sounds that change among them.
const DIGIT_WORDS: [&str; 13] = [
    "いち",
    "いっ",
    "に",
    "さん",
    "よん",
    "ご",
    "ろく",
    "ろっ",
    "なな",
    "しち",
    "はち",
    "はっ",
    "きゅう",
];

/// Whether a number of more than a digit may begin at a position: a unit is
/// there, or a digit and a unit after it -- as every number but a digit alone
/// begins -- so that a digit of no number, as the particle に, is left at once.
fn begins_a_number(reading: &[char], at: usize) -> bool {
    let unit_at = |position: usize| UNIT_WORDS.iter().any(|unit| word(reading, position, unit).is_some());
    unit_at(at)
        || DIGIT_WORDS
            .iter()
            .any(|digit| word(reading, at, digit).is_some_and(unit_at))
}

/// The numbers said from a position of a reading in its lookup form, digits
/// alone left out.
#[must_use]
pub fn numbers_at(reading: &[char], at: usize) -> Vec<Number> {
    // Most positions begin no number: they are left at once, nothing made.
    if !begins_a_number(reading, at) {
        return Vec::new();
    }
    // The ways the big units are said, from the largest down, each there or not.
    let mut parts: Vec<Part> = vec![(at, 0, Vec::new())];
    for (unit, said, kanji) in BIG_UNITS {
        let mut longer = parts.clone();
        for (end, value, numerals) in &parts {
            let mut before = sections(reading, *end);
            // 1兆 is いっちょう.
            if unit == BIG_UNITS[0].0
                && let Some(after) = word(reading, *end, "いっ")
            {
                before.push((
                    after,
                    1,
                    vec![Numeral {
                        begin: *end,
                        end: after,
                        kanji: '一',
                    }],
                ));
            }
            for (section_end, section, section_numerals) in before {
                if let Some(unit_end) = word(reading, section_end, said) {
                    let mut all = [&numerals[..], &section_numerals].concat();
                    all.push(Numeral {
                        begin: section_end,
                        end: unit_end,
                        kanji,
                    });
                    longer.push((unit_end, value + section * unit, all));
                }
            }
        }
        parts = longer;
    }
    let mut numbers = Vec::new();
    for (end, value, numerals) in parts {
        for (section_end, section, section_numerals) in sections(reading, end) {
            numbers.push(Number {
                end: section_end,
                value: value + section,
                numerals: [&numerals[..], &section_numerals].concat(),
            });
        }
        if !numerals.is_empty() {
            numbers.push(Number { end, value, numerals });
        }
    }
    numbers
        .retain(|number| !matches!(&number.numerals[..], [numeral] if ONES.iter().any(|one| one.2 == numeral.kanji)));
    numbers
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The numbers said by the whole of a reading: their values and their
    /// three ways of being written.
    fn whole(reading: &str) -> Vec<(u64, String, String, Option<String>)> {
        let characters: Vec<char> = reading.chars().collect();
        numbers_at(&characters, 0)
            .into_iter()
            .filter(|number| number.end == characters.len())
            .map(|number| (number.value, number.kanji(), number.arabic(), number.mixed()))
            .collect()
    }

    fn one(reading: &str, value: u64, kanji: &str) {
        let found = whole(reading);
        assert_eq!(found.len(), 1, "{reading}: {found:?}");
        assert_eq!((found[0].0, found[0].1.as_str()), (value, kanji), "{reading}");
        assert_eq!(found[0].2, value.to_string());
    }

    #[test]
    fn numbers_are_read_as_they_are_said() {
        one("よんじゅうご", 45, "四十五");
        one("せんきゅうひゃくきゅうじゅういち", 1991, "千九百九十一");
        one("にせんご", 2005, "二千五");
        one("にじゅうろく", 26, "二十六");
        one("ごじゅう", 50, "五十");
        one("じゅう", 10, "十");
        one("じゅうしち", 17, "十七");
        one("ひゃく", 100, "百");
        one("せん", 1000, "千");
        one("さんびゃく", 300, "三百");
        one("ろっぴゃく", 600, "六百");
        one("はっぴゃく", 800, "八百");
        one("さんぜん", 3000, "三千");
        one("はっせん", 8000, "八千");
        one("いちまん", 10_000, "一万");
        one("ひゃくまん", 1_000_000, "百万");
        one("いっちょう", 1_000_000_000_000, "一兆");
        one(
            "にじゅうまんさんぜんよんひゃくごじゅうろく",
            203_456,
            "二十万三千四百五十六",
        );
        one("いちおくにせんまん", 120_000_000, "一億二千万");
    }

    #[test]
    fn numbers_of_ten_thousand_and_more_are_written_with_their_units() {
        let mixed = |reading: &str| whole(reading).pop().and_then(|(_, _, _, mixed)| mixed);
        assert_eq!(mixed("ひゃくまん").as_deref(), Some("100万"));
        assert_eq!(mixed("いちおくにせんまん").as_deref(), Some("1億2000万"));
        assert_eq!(
            mixed("にじゅうまんさんぜんよんひゃくごじゅうろく").as_deref(),
            Some("20万3456")
        );
        assert_eq!(mixed("きゅうせんきゅうひゃくきゅうじゅうきゅう"), None);
    }

    #[test]
    fn what_is_not_a_number_is_not_read_as_one() {
        for reading in [
            "ご",
            "いち",
            "ごご",
            "にに",
            "じゅうじゅう",
            "ひゃくせん",
            "さんひゃく",
            "ろくひゃく",
            "いちじゅう",
            "まん",
            "し",
            "かな",
        ] {
            assert_eq!(whole(reading), Vec::new(), "{reading}");
        }
    }

    #[test]
    fn every_number_a_reading_begins_with_is_found() {
        let characters: Vec<char> = "よんじゅうごきろ".chars().collect();
        let mut found: Vec<(usize, u64)> = numbers_at(&characters, 0)
            .iter()
            .map(|number| (number.end, number.value))
            .collect();
        found.sort_unstable();
        // よん alone is a digit, the dictionary's.
        assert_eq!(found, [(5, 40), (6, 45)]);
        let numeral = numbers_at(&characters, 0)
            .into_iter()
            .find(|number| number.value == 45)
            .expect("45")
            .numerals;
        assert_eq!(
            numeral,
            [
                Numeral {
                    begin: 0,
                    end: 2,
                    kanji: '四'
                },
                Numeral {
                    begin: 2,
                    end: 5,
                    kanji: '十'
                },
                Numeral {
                    begin: 5,
                    end: 6,
                    kanji: '五'
                },
            ]
        );
    }
}
