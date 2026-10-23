// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later
//
// This file is part of KIRAL. The SPDX lines name its copyright holder and its
// license, in the machine-readable form of the REUSE specification: the GNU
// General Public License, version 3 or any later version
// (LICENSES/GPL-3.0-or-later.txt).

//! Rendaku: the second word of a compound voiced -- 駆け, かけ, read がけ in 朝駆け
//! -- where the dictionary's own compounds attest it.
//!
//! A word read with its first kana voiced is no word of the sources until a
//! compound the sources do not hold is typed: 夜駆け, よがけ, is in neither
//! Mozc's dictionary nor SudachiDict, but 朝駆け, 先駆け and 抜け駆け are, each a
//! word and 駆け read がけ after it. So a voiced form is attested where at least
//! [`COMPOUNDS`] compounds of different readings make it so: each a common
//! noun whose surface is a word's and a second's, and whose reading is the
//! first's and the second's with its first kana voiced -- か to が, さ to ざ, た to
//! だ, は to ば and their rows, し and ち to じ, す and つ to ず, as the kana of
//! today write them -- the second a common noun or a verb's continuative form.
//! Proper nouns are left out, as compounds and as words of them: the reading of
//! a family's or a place's name is its own, no evidence of a word's.
//!
//! A voiced form the dictionary lacks becomes a word: its reading and surface;
//! on its left the class of nouns' suffixes, as a voiced word follows another
//! and never begins; on its right the class most of the compounds have; its
//! cost the median of theirs, so that a compound made of it costs its first
//! word and their connection more than one that attests it, and is taken where
//! the connections, or a model, want it. Nothing is guessed: no word is voiced
//! that no compound of the sources shows voiced.

use std::collections::{BTreeMap, HashMap, HashSet};

use crate::{Word, normalize};

/// The least compounds of different readings that attest a voiced form.
pub const COMPOUNDS: usize = 2;

/// The part of speech a voiced form takes on its left, as `id.def` names it:
/// nouns' suffixes, which follow another word.
pub const SUFFIX_CLASS: &str = "名詞,接尾,一般,*,*,*,*";

/// What the compounds attest, and the words made of it.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Voiced {
    /// The words of the voiced forms that the dictionary lacks, by their
    /// readings and surfaces in order.
    pub words: Vec<Word>,
    /// The voiced forms that enough compounds attest, those the dictionary has
    /// counted.
    pub attested: usize,
}

/// The kana a voiced kana may be voiced from: か for が, し and ち for じ.
fn plain(kana: char) -> &'static [char] {
    match kana {
        'が' => &['か'],
        'ぎ' => &['き'],
        'ぐ' => &['く'],
        'げ' => &['け'],
        'ご' => &['こ'],
        'ざ' => &['さ'],
        'じ' => &['し', 'ち'],
        'ず' => &['す', 'つ'],
        'ぜ' => &['せ'],
        'ぞ' => &['そ'],
        'だ' => &['た'],
        'ぢ' => &['ち'],
        'づ' => &['つ'],
        'で' => &['て'],
        'ど' => &['と'],
        'ば' => &['は'],
        'び' => &['ひ'],
        'ぶ' => &['ふ'],
        'べ' => &['へ'],
        'ぼ' => &['ほ'],
        _ => &[],
    }
}

/// Whether a part of speech is a common noun's: a noun, but not a proper
/// noun's, a numeral's, a suffix's, a pronoun's nor a dependent noun's.
fn common_noun(part_of_speech: &str) -> bool {
    part_of_speech.starts_with("名詞,")
        && ![
            "固有名詞",
            "名詞,数",
            "名詞,接尾",
            "代名詞",
            "非自立",
            "特殊",
            "接続詞的",
            "動詞非自立的",
            "引用文字列",
        ]
        .iter()
        .any(|kind| part_of_speech.contains(kind))
}

/// Whether a part of speech is a verb's continuative form.
fn continuative(part_of_speech: &str) -> bool {
    part_of_speech.starts_with("動詞,") && part_of_speech.contains(",連用形,")
}

/// The voiced forms that the words' compounds attest, and the words of those
/// the words lack: `parts_of_speech` names each class, as `id.def` does, and
/// `suffix` is the class of [`SUFFIX_CLASS`].
#[must_use]
pub fn voiced(words: &[Word], parts_of_speech: &[String], suffix: u16) -> Voiced {
    let kind = |id: u16| parts_of_speech.get(usize::from(id)).map_or("", String::as_str);
    let proper = |id: u16| kind(id).contains("固有名詞");
    let readings: Vec<String> = words.iter().map(|word| normalize::lookup_form(&word.reading)).collect();
    let mut by_surface: HashMap<&str, Vec<usize>> = HashMap::new();
    for (index, word) in words.iter().enumerate() {
        by_surface.entry(word.surface.as_str()).or_default().push(index);
    }
    // Two words alike in all but where they are listed, which the dictionary
    // file keeps as one: Mozc's files list 先駆け twice.
    let same = |left: usize, right: usize| {
        let (one, other) = (&words[left], &words[right]);
        readings[left] == readings[right]
            && (one.surface.as_str(), one.lid, one.rid, one.cost, one.flags)
                == (other.surface.as_str(), other.lid, other.rid, other.cost, other.flags)
    };
    // Each voiced form, its reading and surface, and the compounds that attest
    // it.
    let mut attested: BTreeMap<(String, &str), Vec<usize>> = BTreeMap::new();
    let mut base = String::new();
    for (index, compound) in words.iter().enumerate() {
        if !common_noun(kind(compound.lid)) || !common_noun(kind(compound.rid)) {
            continue;
        }
        let reading = readings[index].as_str();
        for (at, _) in compound.surface.char_indices().skip(1) {
            let (first, second) = compound.surface.split_at(at);
            let (Some(firsts), Some(seconds)) = (by_surface.get(first), by_surface.get(second)) else {
                continue;
            };
            for &before in firsts {
                if proper(words[before].lid) {
                    continue;
                }
                let Some(voiced) = reading.strip_prefix(readings[before].as_str()) else {
                    continue;
                };
                let mut kana = voiced.chars();
                let Some(head) = kana.next() else {
                    continue;
                };
                let rest = kana.as_str();
                for &from in plain(head) {
                    base.clear();
                    base.push(from);
                    base.push_str(rest);
                    let shown = seconds.iter().any(|&after| {
                        let word = &words[after];
                        readings[after] == base
                            && !proper(word.lid)
                            && (common_noun(kind(word.lid)) || continuative(kind(word.lid)))
                    });
                    if shown {
                        let compounds = attested.entry((voiced.to_owned(), second)).or_default();
                        if !compounds.iter().any(|&other| same(other, index)) {
                            compounds.push(index);
                        }
                    }
                }
            }
        }
    }
    let held: HashSet<(&str, &str)> = readings
        .iter()
        .zip(words)
        .map(|(reading, word)| (reading.as_str(), word.surface.as_str()))
        .collect();
    let mut result = Voiced::default();
    for ((reading, surface), compounds) in &attested {
        let different: HashSet<&str> = compounds.iter().map(|&index| readings[index].as_str()).collect();
        if different.len() < COMPOUNDS {
            continue;
        }
        result.attested += 1;
        if held.contains(&(reading.as_str(), *surface)) {
            continue;
        }
        let mut costs: Vec<u16> = compounds.iter().map(|&index| words[index].cost).collect();
        costs.sort_unstable();
        let mut classes: BTreeMap<u16, usize> = BTreeMap::new();
        for &index in compounds {
            *classes.entry(words[index].rid).or_default() += 1;
        }
        // The class most of them have, the least id of those as many.
        let rid = classes
            .iter()
            .max_by(|(left_id, left), (right_id, right)| left.cmp(right).then(right_id.cmp(left_id)))
            .map_or(suffix, |(&id, _)| id);
        result.words.push(Word {
            reading: reading.clone(),
            surface: (*surface).to_owned(),
            lid: suffix,
            rid,
            cost: costs[(costs.len() - 1) / 2],
            flags: 0,
        });
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The parts of speech of the tests' classes, by their ids.
    fn parts_of_speech() -> Vec<String> {
        [
            "BOS/EOS,*,*,*,*,*,*",
            "名詞,一般,*,*,*,*,*",
            "名詞,サ変接続,*,*,*,*,*",
            "動詞,自立,*,*,一段,連用形,*",
            "名詞,固有名詞,人名,姓,*,*,*",
            "名詞,接尾,一般,*,*,*,*",
            "形容詞,自立,*,*,形容詞・アウオ段,基本形,*",
        ]
        .map(str::to_owned)
        .to_vec()
    }

    fn word(reading: &str, surface: &str, class: u16, cost: u16) -> Word {
        Word {
            reading: reading.to_owned(),
            surface: surface.to_owned(),
            lid: class,
            rid: class,
            cost,
            flags: 0,
        }
    }

    #[test]
    fn a_form_two_compounds_attest_becomes_a_word() {
        let words = [
            word("あさ", "朝", 1, 4000),
            word("さき", "先", 1, 4000),
            word("ぬけ", "抜け", 3, 5000),
            word("かけ", "駆け", 3, 6000),
            word("あさがけ", "朝駆け", 2, 7268),
            word("さきがけ", "先駆け", 1, 6000),
            word("ぬけがけ", "抜け駆け", 1, 6768),
            // The same compound in another spelling attests nothing more, its
            // cost counted with theirs.
            word("ぬけ", "抜", 3, 5500),
            word("ぬけがけ", "抜駆け", 1, 9223),
            // A word listed twice is one: its cost is counted once.
            word("あさがけ", "朝駆け", 2, 7268),
        ];
        let voiced = voiced(&words, &parts_of_speech(), 5);
        assert_eq!(voiced.attested, 1);
        assert_eq!(
            voiced.words,
            [Word {
                reading: "がけ".to_owned(),
                surface: "駆け".to_owned(),
                lid: 5,
                rid: 1,
                cost: 6768,
                flags: 0,
            }]
        );
    }

    #[test]
    fn what_is_no_evidence_attests_nothing() {
        let words = [
            word("よ", "夜", 1, 4000),
            word("かけ", "駆け", 3, 6000),
            // A compound alone.
            word("あさ", "朝", 1, 4000),
            word("あさがけ", "朝駆け", 1, 7268),
            word("ぬけがけ", "抜駆け", 1, 9223),
            // Names: a family's reading is its own.
            word("こ", "小", 1, 3000),
            word("なか", "中", 1, 3000),
            word("はやし", "林", 1, 5000),
            word("こばやし", "小林", 4, 5000),
            word("なかばやし", "中林", 4, 5000),
            // Not a common noun.
            word("くち", "口", 1, 3000),
            word("ぎり", "義理", 1, 3000),
            word("かたい", "堅い", 6, 3000),
            word("くちがたい", "口堅い", 6, 6000),
            word("ぎりがたい", "義理堅い", 6, 6000),
            // A voiced kana the second word is not read with unvoiced: 城 is no
            // き.
            word("みや", "宮", 1, 4000),
            word("しろ", "城", 1, 4000),
            word("じょう", "城", 1, 4000),
            word("みやぎ", "宮城", 1, 6000),
            word("いしき", "石城", 1, 6000),
            word("いし", "石", 1, 4000),
        ];
        assert_eq!(voiced(&words, &parts_of_speech(), 5), Voiced::default());
    }

    #[test]
    fn a_form_the_dictionary_has_is_counted_and_not_added() {
        let words = [
            word("いき", "行き", 3, 4000),
            word("いのち", "命", 1, 4000),
            word("かけ", "掛け", 3, 4000),
            word("がけ", "掛け", 1, 6755),
            word("いきがけ", "行き掛け", 1, 7000),
            word("いのちがけ", "命掛け", 1, 7000),
        ];
        assert_eq!(
            voiced(&words, &parts_of_speech(), 5),
            Voiced {
                words: Vec::new(),
                attested: 1,
            }
        );
    }

    #[test]
    fn ji_and_zu_are_voiced_from_both_rows() {
        // せかいじゅう and いちねんじゅう: 中, ちゅう, written じゅう as today's kana
        // write it; and みかづき kept with づ.
        let words = [
            word("せかい", "世界", 1, 3000),
            word("いちねん", "一年", 1, 3000),
            word("ちゅう", "中", 1, 3000),
            word("せかいじゅう", "世界中", 1, 5000),
            word("いちねんじゅう", "一年中", 1, 5200),
            word("みか", "三日", 1, 3000),
            word("ゆう", "夕", 1, 3000),
            word("つき", "月", 1, 3000),
            word("みかづき", "三日月", 1, 6000),
            word("ゆうづき", "夕月", 1, 6100),
        ];
        let voiced = voiced(&words, &parts_of_speech(), 5);
        let forms: Vec<(&str, &str, u16)> = voiced
            .words
            .iter()
            .map(|word| (word.reading.as_str(), word.surface.as_str(), word.cost))
            .collect();
        assert_eq!(forms, [("じゅう", "中", 5000), ("づき", "月", 6000)]);
    }
}
