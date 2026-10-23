// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later
//
// This file is part of KIRAL. The SPDX lines name its copyright holder and its
// license, in the machine-readable form of the REUSE specification: the GNU
// General Public License, version 3 or any later version
// (LICENSES/GPL-3.0-or-later.txt).

//! Viterbi against every path: on small dictionaries of random words, classes
//! and costs, the cheapest path [`Lattice::best`] finds costs what the cheapest
//! of all the paths, enumerated one by one, costs, and what its own nodes add up
//! to; and the texts the oracle says the lattice reaches are those of the paths.
//! Within a constraint, the cheapest path [`Lattice::best_within`] finds is the
//! cheapest of those whose text the constraint admits, and there is none where
//! no path's text is admitted, on dictionaries with digits read in kana and
//! suffixes, each digit a node for each spelling; the best path spells each as
//! the suffix after it has it spelt. And the texts of paths: words written as
//! typed, digits read in kana, numbers read in kana made of the dictionary's
//! numerals.

use std::collections::BTreeSet;

use kiral_dict::Dictionary;
use kiral_dict::file::{Unknown, build};
use kiral_dict::mozc::{self, Matrix};
use kiral_dict::{Metadata, Word};
use kiral_lattice::{Constraint, Lattice, Spelling};

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

const KANA: [char; 4] = ['か', 'な', 'ん', 'ア'];

fn dictionary(random: &mut Random, ids: u16) -> Vec<u8> {
    dictionary_of(random, ids, false)
}

/// A dictionary of random words, with digits read in kana and suffixes of nouns
/// among them where `digits`.
fn dictionary_of(random: &mut Random, ids: u16, digits: bool) -> Vec<u8> {
    let words: Vec<Word> = (0..40)
        .map(|number| {
            let (surface, flags) = match number % 8 {
                6 if digits => ((number % 10).to_string(), mozc::DIGIT),
                7 if digits => ("年".to_owned(), mozc::SUFFIX),
                _ => (['甲', '乙', '丙', '丁', 'か', 'な'][number % 6].to_string(), 0),
            };
            Word {
                reading: (0..=random.below(3)).map(|_| KANA[random.below(3)]).collect(),
                surface,
                lid: u16::try_from(1 + random.below(usize::from(ids) - 1)).expect("small"),
                rid: u16::try_from(1 + random.below(usize::from(ids) - 1)).expect("small"),
                cost: u16::try_from(random.below(5000)).expect("small"),
                flags,
            }
        })
        .collect();
    let matrix = Matrix {
        ids,
        costs: (0..usize::from(ids) * usize::from(ids))
            .map(|_| u16::try_from(random.below(3000)).expect("small"))
            .collect(),
    };
    let metadata = Metadata {
        name: "test",
        license: "CC0-1.0",
        source: "made by the test",
        notice: "",
    };
    build(&words, &matrix, Unknown { id: 1, cost: 8000 }, &metadata).expect("a dictionary")
}

/// Every path from a position to the end: its nodes, in order.
fn paths(lattice: &Lattice<'_>, position: usize) -> Vec<Vec<usize>> {
    if position == lattice.len() {
        return vec![Vec::new()];
    }
    let mut all = Vec::new();
    for node in lattice.beginning_at(position) {
        for mut rest in paths(lattice, lattice.nodes()[node].end) {
            rest.insert(0, node);
            all.push(rest);
        }
    }
    all
}

#[test]
fn the_best_path_is_the_cheapest_of_all() {
    let mut random = Random(41);
    for round in 0..20 {
        let bytes = dictionary(&mut random, 6);
        let dictionary = Dictionary::parse(&bytes).expect("a dictionary");
        for _ in 0..20 {
            let reading: String = (0..random.below(7)).map(|_| KANA[random.below(KANA.len())]).collect();
            let lattice = Lattice::new(&dictionary, &reading);
            let best = lattice.best(&dictionary);
            let all = paths(&lattice, 0);
            let cheapest = all
                .iter()
                .map(|path| lattice.cost_of(&dictionary, path))
                .min()
                .expect("a path, characters of no reading being nodes too");
            assert_eq!(best.cost, cheapest, "round {round}, {reading}");
            assert_eq!(
                lattice.cost_of(&dictionary, &best.nodes),
                best.cost,
                "round {round}, {reading}"
            );
            let covered: usize = best
                .nodes
                .iter()
                .map(|&node| {
                    let node = &lattice.nodes()[node];
                    node.end - node.begin
                })
                .sum();
            assert_eq!(covered, lattice.len());
            // The oracle reaches the texts of the paths, and no other.
            let texts: BTreeSet<String> = all
                .iter()
                .map(|path| {
                    lattice.text(&kiral_lattice::Path {
                        nodes: path.clone(),
                        cost: 0,
                    })
                })
                .collect();
            for text in &texts {
                assert!(lattice.reaches(text, false), "{reading}: {text}");
            }
            assert!(!lattice.reaches(&format!("{}x", texts.first().expect("a text")), false));
        }
    }
}

#[test]
fn the_best_path_within_a_constraint_is_the_cheapest_it_admits() {
    let mut random = Random(43);
    for round in 0..20 {
        let bytes = dictionary_of(&mut random, 6, true);
        let dictionary = Dictionary::parse(&bytes).expect("a dictionary");
        for _ in 0..10 {
            let reading: String = (0..random.below(7)).map(|_| KANA[random.below(KANA.len())]).collect();
            let lattice = Lattice::new(&dictionary, &reading);
            let all: Vec<(String, i64)> = paths(&lattice, 0)
                .into_iter()
                .map(|nodes| {
                    let cost = lattice.cost_of(&dictionary, &nodes);
                    (lattice.text(&kiral_lattice::Path { nodes, cost }), cost)
                })
                .collect();
            // The empty constraint; every prefix of some paths' texts, at every
            // byte, characters cut short among them, and each whole text as a
            // complete constraint; and constraints no text meets.
            let mut constraints = vec![
                Constraint::default(),
                Constraint {
                    prefix: "丁丁丁丁丁丁丁丁".as_bytes().to_vec(),
                    complete: false,
                },
                Constraint {
                    prefix: Vec::new(),
                    complete: true,
                },
            ];
            for _ in 0..3 {
                let (text, _) = &all[random.below(all.len())];
                for cut in 0..=text.len() {
                    constraints.push(Constraint {
                        prefix: text.as_bytes()[..cut].to_vec(),
                        complete: false,
                    });
                }
                constraints.push(Constraint {
                    prefix: text.as_bytes().to_vec(),
                    complete: true,
                });
                constraints.push(Constraint {
                    prefix: [text.as_bytes(), "甲".as_bytes()].concat(),
                    complete: false,
                });
            }
            for constraint in &constraints {
                let cheapest = all
                    .iter()
                    .filter(|(text, _)| constraint.admits(text))
                    .map(|&(_, cost)| cost)
                    .min();
                let found = lattice.best_within(&dictionary, constraint);
                let context = format!("round {round}, {reading}, {constraint:?}");
                match (cheapest, found) {
                    (None, None) => {}
                    (Some(cost), Some(path)) => {
                        assert_eq!(path.cost, cost, "{context}");
                        assert_eq!(lattice.cost_of(&dictionary, &path.nodes), cost, "{context}");
                        assert!(constraint.admits(&lattice.text(&path)), "{context}");
                    }
                    (cheapest, found) => panic!("{context}: {cheapest:?} against {found:?}"),
                }
            }
            let best = lattice.best(&dictionary);
            assert_eq!(
                lattice
                    .best_within(&dictionary, &Constraint::default())
                    .map(|path| path.cost),
                Some(best.cost)
            );
            // A digit in Arabic numerals before a suffix, in kanji before
            // anything else.
            for (number, &node) in best.nodes.iter().enumerate() {
                let suffix_after = best
                    .nodes
                    .get(number + 1)
                    .is_some_and(|&next| lattice.nodes()[next].flags & mozc::SUFFIX != 0);
                match lattice.nodes()[node].spelling {
                    Spelling::Arabic => assert!(suffix_after, "round {round}, {reading}"),
                    Spelling::Kanji => assert!(!suffix_after, "round {round}, {reading}"),
                    Spelling::None => {}
                }
            }
        }
    }
}

#[test]
fn words_that_are_their_reading_are_written_as_typed() {
    let mut random = Random(42);
    let bytes = dictionary(&mut random, 4);
    let dictionary = Dictionary::parse(&bytes).expect("a dictionary");
    let lattice = Lattice::new(&dictionary, "ア");
    let best = lattice.best(&dictionary);
    // ア has no word of its own here: it is a character of no reading, written
    // as hiragana, as the input is shown.
    let text = lattice.text(&best);
    assert!(text == "あ" || !text.contains('ア'), "{text}");
    assert_eq!(
        Lattice::new(&dictionary, "").best(&dictionary).nodes,
        Vec::<usize>::new()
    );
}

#[test]
fn digits_read_in_kana_are_written_in_kanji_unless_a_suffix_follows() {
    let word = |reading: &str, surface: &str, id: u16, flags: u16| Word {
        reading: reading.to_owned(),
        surface: surface.to_owned(),
        lid: id,
        rid: id,
        cost: 100,
        flags,
    };
    let words = [
        word("いち", "1", 1, mozc::DIGIT),
        word("ねん", "年", 2, mozc::SUFFIX),
        word("の", "の", 3, 0),
    ];
    let matrix = Matrix {
        ids: 4,
        costs: vec![0; 16],
    };
    let metadata = Metadata {
        name: "test",
        license: "CC0-1.0",
        source: "made by the test",
        notice: "",
    };
    let bytes = build(&words, &matrix, Unknown { id: 3, cost: 8000 }, &metadata).expect("a dictionary");
    let dictionary = Dictionary::parse(&bytes).expect("a dictionary");
    let text = |reading: &str| {
        let lattice = Lattice::new(&dictionary, reading);
        lattice.text(&lattice.best(&dictionary))
    };
    assert_eq!(text("いち"), "一");
    assert_eq!(text("いちの"), "一の");
    assert_eq!(text("いちねん"), "1年");
    assert_eq!(text("いちいち"), "一一");
    // The oracle takes the digit either way, as a conversion offers both.
    let lattice = Lattice::new(&dictionary, "いちねん");
    for reached in ["1年", "一年"] {
        assert!(lattice.reaches(reached, true), "{reached}");
    }
    let lattice = Lattice::new(&dictionary, "いちの");
    for reached in ["1の", "一の"] {
        assert!(lattice.reaches(reached, true), "{reached}");
    }
    assert!(!lattice.reaches("壱の", false));
}

#[test]
fn numbers_read_in_kana_are_made_of_the_dictionarys_numerals() {
    let word = |reading: &str, surface: &str, id: u16, cost: u16, flags: u16| Word {
        reading: reading.to_owned(),
        surface: surface.to_owned(),
        lid: id,
        rid: id,
        cost,
        flags,
    };
    // Classes: 1 the unknown word, 2 the digits in kanji, 3 十, 4 the suffix
    // 年, 5 万, 6 the digits in Arabic numerals, which, as in Mozc's matrix, are
    // dear before a numeral.
    let words = [
        word("よん", "四", 2, 400, 0),
        word("ご", "五", 2, 500, 0),
        word("いち", "一", 2, 300, 0),
        word("じゅう", "十", 3, 300, 0),
        word("まん", "万", 5, 50, 0),
        word("ねん", "年", 4, 100, mozc::SUFFIX),
        word("よん", "4", 6, 200, mozc::DIGIT),
        word("ろく", "ロク", 1, 900, 0),
    ];
    let ids: u16 = 7;
    let mut costs = vec![10u16; usize::from(ids) * usize::from(ids)];
    costs[2 * usize::from(ids) + 3] = 7;
    costs[3 * usize::from(ids) + 2] = 9;
    for numeral in [2, 3, 5, 6] {
        costs[6 * usize::from(ids) + numeral] = 5000;
    }
    let matrix = Matrix { ids, costs };
    let metadata = Metadata {
        name: "test",
        license: "CC0-1.0",
        source: "made by the test",
        notice: "",
    };
    let bytes = build(&words, &matrix, Unknown { id: 1, cost: 8000 }, &metadata).expect("a dictionary");
    let dictionary = Dictionary::parse(&bytes).expect("a dictionary");
    let text = |reading: &str| {
        let lattice = Lattice::new(&dictionary, reading);
        lattice.text(&lattice.best(&dictionary))
    };
    // 四十五 costs its numerals and their connections, 四 to 十 and 十 to 五, less
    // one a connection.
    let lattice = Lattice::new(&dictionary, "よんじゅうご");
    let spelt: Vec<(String, u32, Spelling)> = lattice
        .nodes()
        .iter()
        .filter(|node| node.begin == 0 && node.end == 6)
        .map(|node| {
            let mut spelling = String::new();
            lattice.push_text(node, &mut spelling);
            (spelling, node.cost, node.spelling)
        })
        .collect();
    assert_eq!(
        spelt,
        [
            ("四十五".to_owned(), 400 + 6 + 300 + 8 + 500, Spelling::Kanji),
            ("45".to_owned(), 400 + 6 + 300 + 8 + 500, Spelling::Arabic),
        ]
    );
    assert_eq!(text("よんじゅうご"), "四十五");
    assert_eq!(text("よんじゅうごねん"), "45年");
    assert_eq!(text("よんねん"), "4年");
    assert_eq!(text("よん"), "四");
    let lattice = Lattice::new(&dictionary, "よんじゅうごねん");
    for reached in ["45年", "四十五年"] {
        assert!(lattice.reaches(reached, true), "{reached}");
    }
    let within = |reading: &str, prefix: &str| {
        let lattice = Lattice::new(&dictionary, reading);
        let constraint = Constraint {
            prefix: prefix.as_bytes().to_vec(),
            complete: false,
        };
        lattice
            .best_within(&dictionary, &constraint)
            .map(|path| lattice.text(&path))
    };
    assert_eq!(within("よんじゅうご", "4").as_deref(), Some("45"));
    assert_eq!(within("よんじゅうごねん", "四十五").as_deref(), Some("四十五年"));
    // From ten thousand on, the sections in Arabic numerals and the units in
    // kanji besides, before the number in Arabic numerals alone.
    assert_eq!(text("いちまんねん"), "1万年");
    assert_eq!(text("いちまん"), "一万");
    assert_eq!(within("いちまん", "10000").as_deref(), Some("10000"));
    // A number of a numeral the dictionary lacks is not made: ろく is no 六.
    let lattice = Lattice::new(&dictionary, "ろくじゅう");
    assert!(!lattice.reaches("60", true));
    assert!(!lattice.reaches("六十", false));
}
