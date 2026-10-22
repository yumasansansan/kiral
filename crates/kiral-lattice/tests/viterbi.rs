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
//! And the texts of paths: words written as typed, digits read in kana.

use std::collections::BTreeSet;

use kiral_dict::Dictionary;
use kiral_dict::file::{Unknown, build};
use kiral_dict::mozc::{self, Matrix};
use kiral_dict::{Metadata, Word};
use kiral_lattice::Lattice;

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
    let words: Vec<Word> = (0..40)
        .map(|number| Word {
            reading: (0..=random.below(3)).map(|_| KANA[random.below(3)]).collect(),
            surface: format!("{}", ['甲', '乙', '丙', '丁', 'か', 'な'][number % 6]),
            lid: u16::try_from(1 + random.below(usize::from(ids) - 1)).expect("small"),
            rid: u16::try_from(1 + random.below(usize::from(ids) - 1)).expect("small"),
            cost: u16::try_from(random.below(5000)).expect("small"),
            flags: 0,
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
