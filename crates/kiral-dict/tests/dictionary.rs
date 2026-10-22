// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later
//
// This file is part of KIRAL. The SPDX lines name its copyright holder and its
// license, in the machine-readable form of the REUSE specification: the GNU
// General Public License, version 3 or any later version
// (LICENSES/GPL-3.0-or-later.txt).

//! A dictionary round-trips through its file: every word built in is found
//! under its reading, cheapest first, the words given twice once, and the
//! readings that begin an input are those a plain search of the words finds.
//! And the file's parser takes any bytes: a file changed at random is read or
//! refused, and looked up, never a panic.

use std::collections::BTreeMap;

use kiral_dict::file::{Unknown, build};
use kiral_dict::mozc::{Matrix, SPELLING_CORRECTION};
use kiral_dict::{Dictionary, Metadata, Word, normalize};

/// `SplitMix64`: a fixed stream of numbers, the same on every system.
struct Random(u64);

impl Random {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    fn below(&mut self, bound: usize) -> usize {
        usize::try_from(self.next() % u64::try_from(bound).expect("small")).expect("small")
    }
}

const UNKNOWN: Unknown = Unknown { id: 1, cost: 20_000 };

const METADATA: Metadata<'static> = Metadata {
    name: "test",
    license: "CC0-1.0",
    source: "made by the test",
    notice: "nothing to note",
};

/// Readings of a few kana and symbols, so that many share prefixes.
fn reading(random: &mut Random) -> String {
    let characters = ['か', 'き', 'く', 'ん', 'ー', 'ゔ', 'a', '(', '、', '漢'];
    (0..=random.below(5))
        .map(|_| characters[random.below(characters.len())])
        .collect()
}

fn words(random: &mut Random, count: usize, ids: u16) -> Vec<Word> {
    (0..count)
        .map(|number| Word {
            reading: reading(random),
            surface: format!("語{}", number % 97),
            lid: u16::try_from(random.below(usize::from(ids))).expect("small"),
            rid: u16::try_from(random.below(usize::from(ids))).expect("small"),
            cost: u16::try_from(random.below(20_000)).expect("small"),
            flags: if random.below(10) == 0 { SPELLING_CORRECTION } else { 0 },
        })
        .collect()
}

fn matrix(random: &mut Random, ids: u16) -> Matrix {
    let count = usize::from(ids) * usize::from(ids);
    Matrix {
        ids,
        costs: (0..count)
            .map(|_| u16::try_from(random.below(10_000)).expect("small"))
            .collect(),
    }
}

/// The fields of a reading's entries, cheapest first.
type Listed = Vec<(u16, u16, u16, String, u16)>;

/// The words of each reading, as the dictionary should give them back: in
/// their lookup form, cheapest first, the same word once.
type Expected = BTreeMap<String, Listed>;

fn expected(words: &[Word]) -> Expected {
    let mut expected: Expected = BTreeMap::new();
    for word in words {
        expected
            .entry(normalize::lookup_form(&word.reading))
            .or_default()
            .push((word.cost, word.lid, word.rid, word.surface.clone(), word.flags));
    }
    for entries in expected.values_mut() {
        entries.sort();
        entries.dedup();
    }
    expected
}

fn entries_of(entries: kiral_dict::Entries<'_>) -> Listed {
    entries
        .map(|entry| (entry.cost, entry.lid, entry.rid, entry.surface.to_owned(), entry.flags))
        .collect()
}

#[test]
fn words_come_back_under_their_readings() {
    let mut random = Random(31);
    let ids = 13;
    let mut words = words(&mut random, 3000, ids);
    // A word given twice, and a reading in katakana and full width.
    words.push(words[0].clone());
    words.push(Word {
        reading: "カ（".to_owned(),
        surface: "カ括弧".to_owned(),
        lid: 1,
        rid: 2,
        cost: 7,
        flags: 0,
    });
    let matrix = matrix(&mut random, ids);
    let bytes = build(&words, &matrix, UNKNOWN, &METADATA).expect("a dictionary");
    let dictionary = Dictionary::parse(&bytes).expect("a dictionary");
    let expected = expected(&words);
    assert_eq!(dictionary.metadata(), METADATA);
    assert_eq!(dictionary.unknown(), UNKNOWN);
    assert_eq!(usize::try_from(dictionary.readings()).expect("small"), expected.len());
    assert_eq!(
        usize::try_from(dictionary.entries()).expect("small"),
        expected.values().map(Vec::len).sum::<usize>()
    );
    for (reading, entries) in &expected {
        assert_eq!(
            &entries_of(dictionary.lookup(reading).expect("a reading built in")),
            entries,
            "{reading}"
        );
    }
    assert_eq!(
        entries_of(dictionary.lookup("か(").expect("its lookup form")).len(),
        entries_of(dictionary.lookup("カ（").expect("as written")).len()
    );
    assert!(dictionary.lookup("ぬ").is_none());
    for rid in 0..ids {
        for lid in 0..ids {
            assert_eq!(
                dictionary.connection(rid, lid),
                matrix.costs[usize::from(rid) * usize::from(ids) + usize::from(lid)]
            );
        }
    }
    assert_eq!(dictionary.connection(ids, 0), u16::MAX);
    // The prefixes of inputs, against a plain search of the readings.
    for _ in 0..300 {
        let input: Vec<char> = reading(&mut random)
            .chars()
            .chain(reading(&mut random).chars())
            .collect();
        let found: Vec<(usize, Listed)> = dictionary
            .prefixes(&input)
            .map(|(length, entries)| (length, entries_of(entries)))
            .collect();
        let plain: Vec<(usize, Listed)> = (1..=input.len())
            .filter_map(|length| {
                let prefix: String = input[..length].iter().collect();
                expected.get(&prefix).map(|entries| (length, entries.clone()))
            })
            .collect();
        assert_eq!(found, plain, "{input:?}");
    }
}

#[test]
fn what_cannot_be_built_is_refused() {
    let mut random = Random(32);
    let matrix = matrix(&mut random, 4);
    let mut words = words(&mut random, 10, 4);
    words[3].lid = 4;
    assert!(build(&words, &matrix, UNKNOWN, &METADATA).is_err());
    let mut words = self::words(&mut random, 10, 4);
    words[0].surface = "あ".repeat(2000);
    assert!(build(&words, &matrix, UNKNOWN, &METADATA).is_err());
    let unknown = Unknown { id: 4, cost: 1 };
    assert!(build(&self::words(&mut random, 10, 4), &matrix, unknown, &METADATA).is_err());
    let short = Matrix {
        ids: 4,
        costs: vec![0; 15],
    };
    assert!(build(&self::words(&mut random, 10, 4), &short, UNKNOWN, &METADATA).is_err());
    let empty = build(&[], &matrix, UNKNOWN, &METADATA).expect("an empty dictionary");
    let dictionary = Dictionary::parse(&empty).expect("an empty dictionary");
    assert!(dictionary.lookup("か").is_none());
    assert_eq!(dictionary.prefixes(&['か']).count(), 0);
}

#[test]
fn any_bytes_are_read_or_refused() {
    let mut random = Random(33);
    let ids = 5;
    let words = words(&mut random, 200, ids);
    let bytes = build(&words, &matrix(&mut random, ids), UNKNOWN, &METADATA).expect("a dictionary");
    let input: Vec<char> = "かきくんーゔa(、漢か".chars().collect();
    for _ in 0..3000 {
        let mut changed = bytes.clone();
        for _ in 0..=random.below(8) {
            let at = random.below(changed.len());
            changed[at] ^= 1 << random.below(8);
        }
        if random.below(10) == 0 {
            changed.truncate(random.below(changed.len()));
        }
        if let Ok(dictionary) = Dictionary::parse(&changed) {
            for start in 0..input.len() {
                for (_, entries) in dictionary.prefixes(&input[start..]) {
                    for entry in entries {
                        let _ = dictionary.connection(entry.rid, entry.lid);
                    }
                }
            }
            let _ = dictionary.lookup("かき").map(Iterator::count);
        }
    }
    for length in 0..200 {
        let _ = Dictionary::parse(&bytes[..length]);
    }
}
