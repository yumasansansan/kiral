// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later
//
// This file is part of KIRAL. The SPDX lines name its copyright holder and its
// license, in the machine-readable form of the REUSE specification: the GNU
// General Public License, version 3 or any later version
// (LICENSES/GPL-3.0-or-later.txt).

//! The lattice of a reading, and its best path.
//!
//! A node of the lattice is a word of the dictionary whose reading is a stretch
//! of the input's, and, at every position, one character taken as a word the
//! dictionary does not have -- dearer than any word, so that it is taken only
//! where no word covers the character -- so that every reading has a path. A
//! path's cost is the sum of its words' costs and of the costs of the
//! connections from each word's class on its right to the next one's on its
//! left, from the beginning of the sentence (id 0) before the first, to its end
//! (id 0) after the last. [`Lattice::best`] finds the cheapest by Viterbi,
//! taking the words that end at a position together by their class on the
//! right, and those that begin there by their class on the left, so that the
//! matrix is read once for each pair of classes rather than each pair of words.
//!
//! The text of a path is its words' surfaces, but a word whose surface is its
//! own reading, and a character of no reading, are written as the input wrote
//! them, so that a parenthesis typed full-width stays full-width. A number has
//! a node for each of its spellings, of the same classes and cost: a digit read
//! in kana -- the dictionary's いち for 1 -- in Arabic numerals and in kanji;
//! and a number read in kana that the dictionary's numerals make ([`number`]),
//! 45 and 四十五 for よんじゅうご, with its sections in Arabic numerals and its
//! units in kanji besides from ten thousand on, 100万. Such a number costs what
//! its numerals and their connections cost, less one a connection, the classes
//! on its left and right its first's and its last's, so that a stretch is read
//! as a number where the dictionary's numerals would read it, whole rather than
//! cut, and the number is written either way.
//! [`Lattice::best`] writes a number as Mozc shows one, in kanji unless a suffix
//! of nouns follows (いち alone as 一, いちねん as 1年): a spelling in Arabic
//! numerals only before a suffix, one in kanji only before anything else.
//!
//! [`Lattice::best_within`] finds the cheapest path whose text begins with
//! given bytes, or is them ([`Constraint`]): what a model has accepted of a
//! draft and the token it wants next, which the next draft must keep.
//! [`Lattice::reaches`] tells whether a text is the text of some path: the
//! oracle, the share of a benchmark's accepted conversions that the dictionary
//! can make at all. Both take a number in either spelling, whatever follows
//! it, as a conversion offers both.
#![forbid(unsafe_code)]

use kiral_dict::{Dictionary, mozc, normalize};

pub mod number;

/// The numerals that numbers are made of.
const NUMERALS: &str = "一二三四五六七八九十百千万億兆";

/// How a node is written.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Text<'a> {
    /// A surface of the dictionary's, or a numeral.
    Word(&'a str),
    /// As the input wrote the stretch it reads.
    Typed,
    /// A text the lattice made, a number's spelling: by its number among them.
    Made(usize),
}

/// Which spelling of a number a node is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Spelling {
    /// None: the node is no number.
    None,
    /// In Arabic numerals, or its sections in them: what [`Lattice::best`]
    /// writes before a suffix.
    Arabic,
    /// In kanji: what [`Lattice::best`] writes before anything else.
    Kanji,
}

/// A node of the lattice: a word, and the stretch of the input it reads.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Node<'a> {
    /// The position of its first character.
    pub begin: usize,
    /// The position after its last character.
    pub end: usize,
    /// The id of its class on its left.
    pub lid: u16,
    /// The id of its class on its right.
    pub rid: u16,
    /// Its cost.
    pub cost: u32,
    /// How it is written.
    pub text: Text<'a>,
    /// Which spelling of a number it is.
    pub spelling: Spelling,
    /// Its flags of the dictionary's: [`mozc::SUFFIX`] among them.
    pub flags: u16,
    /// Whether the dictionary has it: a word, or a number of its numerals.
    pub known: bool,
}

/// The lattice of a reading.
#[derive(Debug, Clone)]
pub struct Lattice<'a> {
    /// The input as it is shown: its katakana as hiragana.
    shown: Vec<char>,
    nodes: Vec<Node<'a>>,
    /// The words that begin at each position: `starts[p]..starts[p + 1]`.
    starts: Vec<usize>,
    /// The numbers that begin at each position, after every word:
    /// `numbers[p]..numbers[p + 1]`.
    numbers: Vec<usize>,
    /// The nodes that end at each position.
    ends: Vec<Vec<usize>>,
    /// The texts the lattice made: the numbers' spellings.
    made: Vec<String>,
}

/// A path through a lattice: its nodes, in order, and its cost.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Path {
    /// The nodes, by their numbers in the lattice.
    pub nodes: Vec<usize>,
    /// The cost: the words' costs and the connections', the beginning and the
    /// end of the sentence included.
    pub cost: i64,
}

/// No node: the beginning of the sentence, before the first.
const BEGINNING: usize = usize::MAX;

/// What the text of a path must begin with, and whether it must end there:
/// what a model has accepted of a draft and the token it wants next, or the
/// whole of the text it wants.
#[derive(Debug, Clone, Default, PartialEq, Eq, Hash)]
pub struct Constraint {
    /// The bytes the text's UTF-8 begins with: whole characters or not, as a
    /// model's token may end within a character.
    pub prefix: Vec<u8>,
    /// Whether the text is these bytes and nothing more.
    pub complete: bool,
}

impl Constraint {
    /// Whether the constraint admits a text.
    #[must_use]
    pub fn admits(&self, text: &str) -> bool {
        if self.complete {
            text.as_bytes() == self.prefix
        } else {
            text.as_bytes().starts_with(&self.prefix)
        }
    }

    /// The bytes of the prefix matched after a piece of text that follows a
    /// text matching `before` of them, the prefix's length once it is passed;
    /// none where the piece leaves the prefix, or goes on past a complete one.
    fn advance(&self, before: usize, piece: &[u8]) -> Option<usize> {
        let length = self.prefix.len();
        if before >= length {
            return (!self.complete || piece.is_empty()).then_some(length);
        }
        let overlap = piece.len().min(length - before);
        if piece[..overlap] != self.prefix[before..before + overlap] {
            return None;
        }
        let after = before + piece.len();
        if after <= length {
            Some(after)
        } else if self.complete {
            None
        } else {
            Some(length)
        }
    }
}

/// A way to a node, for [`Lattice::best_within`]: the bytes of the
/// constraint's prefix that the text before the node matches, the cost from
/// the beginning of the sentence to the node, the node's own included, and the
/// way to the node before it.
#[derive(Debug, Clone, Copy)]
struct Way {
    node: usize,
    matched: usize,
    cost: i64,
    from: usize,
}

/// The cheapest of the ways that end at a position with a class on their right
/// and the bytes of the prefix their text matches, their last node's text
/// included.
#[derive(Debug, Clone, Copy)]
struct Ending {
    rid: u16,
    matched: usize,
    cost: i64,
    way: usize,
}

/// No way: the beginning of the sentence, before the first node.
const NO_WAY: usize = usize::MAX;

/// Keeps the cheapest ending of each class and bytes matched.
fn keep_cheapest(endings: &mut Vec<Ending>, ending: Ending) {
    if let Some(kept) = endings
        .iter_mut()
        .find(|kept| kept.rid == ending.rid && kept.matched == ending.matched)
    {
        if ending.cost < kept.cost {
            *kept = ending;
        }
    } else {
        endings.push(ending);
    }
}

/// The cheapest entry of a numeral that reads a stretch: what numbers are made
/// of.
#[derive(Debug, Clone, Copy)]
struct NumeralEntry {
    begin: usize,
    end: usize,
    kanji: char,
    cost: u32,
    lid: u16,
    rid: u16,
}

/// The cost of a number made of the dictionary's numerals, and its classes on
/// the left and on the right: its numerals' costs and the connections between
/// them, less one for each connection, its first's class on the left and its
/// last's on the right; none where the dictionary lacks a numeral, with the
/// stretch it reads. Less one, so that a number said whole costs less than its
/// numerals taken one by one and than any two numbers it may be cut into --
/// 45 than 4 and 15, which would write 415 -- for which its numerals and
/// connections would cost the same.
fn number_cost(
    dictionary: &Dictionary<'_>,
    numerals: &[NumeralEntry],
    number: &number::Number,
) -> Option<(u32, u16, u16)> {
    let mut cost = 0u32;
    let mut classes: Option<(u16, u16)> = None;
    for numeral in &number.numerals {
        let entry = numerals
            .iter()
            .find(|entry| entry.begin == numeral.begin && entry.end == numeral.end && entry.kanji == numeral.kanji)?;
        cost += entry.cost;
        classes = Some(match classes {
            None => (entry.lid, entry.rid),
            Some((first, previous)) => {
                cost += u32::from(dictionary.connection(previous, entry.lid)).saturating_sub(1);
                (first, entry.rid)
            }
        });
    }
    classes.map(|(lid, rid)| (cost, lid, rid))
}

impl<'a> Lattice<'a> {
    /// The lattice of a reading, in any form, over a dictionary: the words of
    /// the dictionary that begin at each position, a digit read in kana a node
    /// for each spelling, and a character of no reading at each; then, after
    /// them all, the numbers the dictionary's numerals make that begin at each
    /// position, as their numerals are known only once the words of every
    /// position are.
    #[must_use]
    pub fn new(dictionary: &Dictionary<'a>, reading: &str) -> Self {
        let shown: Vec<char> = reading.chars().map(normalize::display).collect();
        let lookup: Vec<char> = reading.chars().map(normalize::lookup).collect();
        let mut lattice = Self {
            shown,
            nodes: Vec::new(),
            starts: Vec::with_capacity(lookup.len() + 1),
            numbers: Vec::with_capacity(lookup.len() + 1),
            ends: vec![Vec::new(); lookup.len() + 1],
            made: Vec::new(),
        };
        let numerals = lattice.add_words(dictionary, &lookup);
        for begin in 0..lookup.len() {
            lattice.numbers.push(lattice.nodes.len());
            for number in number::numbers_at(&lookup, begin) {
                let Some((cost, lid, rid)) = number_cost(dictionary, &numerals, &number) else {
                    continue;
                };
                let spell = |lattice: &mut Self, text: String, spelling: Spelling, cost: u32| {
                    lattice.made.push(text);
                    let made = lattice.made.len() - 1;
                    lattice.push(Node {
                        begin,
                        end: number.end,
                        lid,
                        rid,
                        cost,
                        text: Text::Made(made),
                        spelling,
                        flags: 0,
                        known: true,
                    });
                };
                spell(&mut lattice, number.kanji(), Spelling::Kanji, cost);
                // From ten thousand on, the sections in Arabic numerals before
                // the number in Arabic numerals alone, the dearer by one.
                match number.mixed() {
                    Some(mixed) => {
                        spell(&mut lattice, mixed, Spelling::Arabic, cost);
                        spell(&mut lattice, number.arabic(), Spelling::Arabic, cost + 1);
                    }
                    None => spell(&mut lattice, number.arabic(), Spelling::Arabic, cost),
                }
            }
        }
        lattice.numbers.push(lattice.nodes.len());
        lattice
    }

    /// Adds the words of the dictionary that begin at each position, a digit
    /// read in kana a node for each spelling, and a character of no reading at
    /// each; returns the cheapest entry of each numeral at each stretch.
    fn add_words(&mut self, dictionary: &Dictionary<'a>, lookup: &[char]) -> Vec<NumeralEntry> {
        let unknown = dictionary.unknown();
        let mut numerals: Vec<NumeralEntry> = Vec::new();
        let mut stretch = String::new();
        for begin in 0..lookup.len() {
            self.starts.push(self.nodes.len());
            for (length, entries) in dictionary.prefixes(&lookup[begin..]) {
                let end = begin + length;
                stretch.clear();
                stretch.extend(&lookup[begin..end]);
                for entry in entries {
                    let node = Node {
                        begin,
                        end,
                        lid: entry.lid,
                        rid: entry.rid,
                        cost: u32::from(entry.cost),
                        text: if entry.surface == stretch {
                            Text::Typed
                        } else {
                            Text::Word(entry.surface)
                        },
                        spelling: Spelling::None,
                        flags: entry.flags,
                        known: true,
                    };
                    let digit = (entry.flags & mozc::DIGIT != 0)
                        .then(|| mozc::kanji_digit(entry.surface))
                        .flatten();
                    if let Some(kanji) = digit {
                        self.push(Node {
                            spelling: Spelling::Arabic,
                            ..node
                        });
                        self.push(Node {
                            text: Text::Word(kanji),
                            spelling: Spelling::Kanji,
                            ..node
                        });
                    } else {
                        self.push(node);
                    }
                    let mut characters = entry.surface.chars();
                    if let (Some(kanji), None) = (characters.next(), characters.next())
                        && NUMERALS.contains(kanji)
                    {
                        let cost = u32::from(entry.cost);
                        match numerals
                            .iter_mut()
                            .find(|kept| kept.begin == begin && kept.end == end && kept.kanji == kanji)
                        {
                            Some(kept) if cost < kept.cost => {
                                (kept.cost, kept.lid, kept.rid) = (cost, entry.lid, entry.rid);
                            }
                            Some(_) => {}
                            None => numerals.push(NumeralEntry {
                                begin,
                                end,
                                kanji,
                                cost,
                                lid: entry.lid,
                                rid: entry.rid,
                            }),
                        }
                    }
                }
            }
            self.push(Node {
                begin,
                end: begin + 1,
                lid: unknown.id,
                rid: unknown.id,
                cost: u32::from(unknown.cost),
                text: Text::Typed,
                spelling: Spelling::None,
                flags: 0,
                known: false,
            });
        }
        self.starts.push(self.nodes.len());
        numerals
    }

    /// Adds a node, and notes where it ends.
    fn push(&mut self, node: Node<'a>) {
        self.ends[node.end].push(self.nodes.len());
        self.nodes.push(node);
    }

    /// The characters of the reading.
    #[must_use]
    pub fn len(&self) -> usize {
        self.shown.len()
    }

    /// Whether the reading is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.shown.is_empty()
    }

    /// The nodes, those that begin earlier first.
    #[must_use]
    pub fn nodes(&self) -> &[Node<'a>] {
        &self.nodes
    }

    /// The numbers of the nodes that begin at a position: its words, then its
    /// numbers.
    pub fn beginning_at(&self, position: usize) -> std::iter::Chain<std::ops::Range<usize>, std::ops::Range<usize>> {
        let range = |bounds: &[usize]| match (bounds.get(position), bounds.get(position + 1)) {
            (Some(&start), Some(&end)) => start..end,
            _ => 0..0,
        };
        range(&self.starts).chain(range(&self.numbers))
    }

    /// Appends the text of a node to `text`.
    pub fn push_text(&self, node: &Node<'a>, text: &mut String) {
        match node.text {
            Text::Word(word) => text.push_str(word),
            Text::Typed => text.extend(&self.shown[node.begin..node.end]),
            Text::Made(made) => text.push_str(&self.made[made]),
        }
    }

    /// The text of a path: its nodes' texts, one after another.
    #[must_use]
    pub fn text(&self, path: &Path) -> String {
        let mut text = String::new();
        for &node in &path.nodes {
            self.push_text(&self.nodes[node], &mut text);
        }
        text
    }

    /// The cheapest path, by Viterbi, a number spelt in Arabic numerals only
    /// before a suffix and one in kanji only before anything else; the empty
    /// path for an empty reading. The two spellings costing the same, the path
    /// costs what the cheapest path of any spelling does.
    #[must_use]
    pub fn best(&self, dictionary: &Dictionary<'_>) -> Path {
        let length = self.len();
        let mut best = vec![i64::MAX; self.nodes.len()];
        let mut back = vec![BEGINNING; self.nodes.len()];
        // The cheapest way to each class on the right of the words that end at a
        // position: before a word that is no suffix, and before a suffix.
        let mut rights: [Vec<(u16, i64, usize)>; 2] = [Vec::new(), Vec::new()];
        // And to each class on the left of those that begin there, before a
        // suffix or not.
        let mut lefts: Vec<(u16, bool, i64, usize)> = Vec::new();
        for position in 0..length {
            for kept in &mut rights {
                kept.clear();
            }
            // Where no number ends, the ways before a suffix are those before
            // anything else, and are taken once.
            let spelled = self.ends[position]
                .iter()
                .any(|&node| self.nodes[node].spelling != Spelling::None);
            if position == 0 {
                rights[0].push((0, 0, BEGINNING));
            }
            for &node in &self.ends[position] {
                let Node { rid, spelling, .. } = self.nodes[node];
                if spelling != Spelling::Arabic {
                    group(&mut rights[0], rid, best[node], node);
                }
                if spelled && spelling != Spelling::Kanji {
                    group(&mut rights[1], rid, best[node], node);
                }
            }
            lefts.clear();
            for node in self.beginning_at(position) {
                let Node { lid, flags, cost, .. } = self.nodes[node];
                let suffix = spelled && flags & mozc::SUFFIX != 0;
                let (way, from) = if let Some(&(_, _, way, from)) = lefts
                    .iter()
                    .find(|&&(left, before_suffix, _, _)| left == lid && before_suffix == suffix)
                {
                    (way, from)
                } else {
                    let (way, from) = rights[usize::from(suffix)]
                        .iter()
                        .map(|&(rid, way, from)| (way + i64::from(dictionary.connection(rid, lid)), from))
                        .min_by_key(|&(way, _)| way)
                        .unwrap_or((i64::MAX / 2, BEGINNING));
                    lefts.push((lid, suffix, way, from));
                    (way, from)
                };
                best[node] = way + i64::from(cost);
                back[node] = from;
            }
        }
        let Some((cost, last)) = self.ends[length]
            .iter()
            .filter(|&&node| self.nodes[node].spelling != Spelling::Arabic)
            .map(|&node| {
                (
                    best[node] + i64::from(dictionary.connection(self.nodes[node].rid, 0)),
                    node,
                )
            })
            .min_by_key(|&(cost, _)| cost)
        else {
            return Path {
                nodes: Vec::new(),
                cost: i64::from(dictionary.connection(0, 0)),
            };
        };
        let mut nodes = vec![last];
        while let Some(&node) = nodes.last()
            && back[node] != BEGINNING
        {
            nodes.push(back[node]);
        }
        nodes.reverse();
        Path { nodes, cost }
    }

    /// The cheapest path whose text the constraint admits, a number in either
    /// spelling; none where no path's text is admitted. Viterbi over the nodes
    /// and the bytes of the constraint's prefix that the text before each
    /// matches, a node's own text matched on the way to the next; the ways that
    /// end at a position are taken together by their class on the right and
    /// the bytes they matched, as [`Lattice::best`] takes them by their class.
    #[must_use]
    pub fn best_within(&self, dictionary: &Dictionary<'_>, constraint: &Constraint) -> Option<Path> {
        let length = self.len();
        if length == 0 {
            return constraint.admits("").then(|| Path {
                nodes: Vec::new(),
                cost: i64::from(dictionary.connection(0, 0)),
            });
        }
        let mut ways: Vec<Way> = Vec::new();
        let mut ranges = vec![0..0; self.nodes.len()];
        let mut endings: Vec<Ending> = Vec::new();
        let mut text = String::new();
        for position in 0..length {
            endings.clear();
            if position == 0 {
                endings.push(Ending {
                    rid: 0,
                    matched: 0,
                    cost: 0,
                    way: NO_WAY,
                });
            }
            for &node in &self.ends[position] {
                text.clear();
                self.push_text(&self.nodes[node], &mut text);
                for way in ranges[node].clone() {
                    if let Some(matched) = constraint.advance(ways[way].matched, text.as_bytes()) {
                        let ending = Ending {
                            rid: self.nodes[node].rid,
                            matched,
                            cost: ways[way].cost,
                            way,
                        };
                        keep_cheapest(&mut endings, ending);
                    }
                }
            }
            for node in self.beginning_at(position) {
                let start = ways.len();
                let Node { lid, cost, .. } = self.nodes[node];
                for ending in &endings {
                    let total = ending.cost + i64::from(dictionary.connection(ending.rid, lid)) + i64::from(cost);
                    let way = Way {
                        node,
                        matched: ending.matched,
                        cost: total,
                        from: ending.way,
                    };
                    if let Some(kept) = ways[start..].iter_mut().find(|kept| kept.matched == ending.matched) {
                        if total < kept.cost {
                            *kept = way;
                        }
                    } else {
                        ways.push(way);
                    }
                }
                ranges[node] = start..ways.len();
            }
        }
        // The ends of the ways that reach the end of the reading, the whole
        // prefix matched.
        let mut best: Option<(i64, usize)> = None;
        for &node in &self.ends[length] {
            text.clear();
            self.push_text(&self.nodes[node], &mut text);
            for way in ranges[node].clone() {
                if constraint.advance(ways[way].matched, text.as_bytes()) != Some(constraint.prefix.len()) {
                    continue;
                }
                let cost = ways[way].cost + i64::from(dictionary.connection(self.nodes[node].rid, 0));
                if best.is_none_or(|(least, _)| cost < least) {
                    best = Some((cost, way));
                }
            }
        }
        let (cost, last) = best?;
        let mut nodes = Vec::new();
        let mut way = last;
        while way != NO_WAY {
            nodes.push(ways[way].node);
            way = ways[way].from;
        }
        nodes.reverse();
        Some(Path { nodes, cost })
    }

    /// The cost of a path, recomputed from its nodes: for the tests of
    /// [`Lattice::best`].
    #[must_use]
    pub fn cost_of(&self, dictionary: &Dictionary<'_>, nodes: &[usize]) -> i64 {
        let mut cost = 0;
        let mut rid = 0;
        for &node in nodes {
            let node = &self.nodes[node];
            cost += i64::from(dictionary.connection(rid, node.lid)) + i64::from(node.cost);
            rid = node.rid;
        }
        cost + i64::from(dictionary.connection(rid, 0))
    }

    /// Whether a text is the text of some path: of words the dictionary has
    /// alone, where `known_only`, or of characters of no reading as well. A
    /// number may be written in either spelling, whatever follows it.
    #[must_use]
    pub fn reaches(&self, text: &str, known_only: bool) -> bool {
        let target: Vec<char> = text.chars().collect();
        let width = target.len() + 1;
        let mut reachable = vec![false; (self.len() + 1) * width];
        reachable[0] = true;
        let mut node_text = String::new();
        for position in 0..self.len() {
            for done in 0..width {
                if !reachable[position * width + done] {
                    continue;
                }
                for node in self.beginning_at(position) {
                    let node = &self.nodes[node];
                    if known_only && !node.known {
                        continue;
                    }
                    node_text.clear();
                    self.push_text(node, &mut node_text);
                    let mut taken = 0;
                    let fits = node_text.chars().all(|character| {
                        taken += 1;
                        target.get(done + taken - 1) == Some(&character)
                    });
                    if fits {
                        reachable[node.end * width + done + taken] = true;
                    }
                }
            }
        }
        reachable[self.len() * width + target.len()]
    }
}

/// Keeps the cheapest way to a class: `(class, cost, node)`, one for each class.
fn group(groups: &mut Vec<(u16, i64, usize)>, class: u16, cost: i64, node: usize) {
    if let Some(entry) = groups.iter_mut().find(|(existing, _, _)| *existing == class) {
        if cost < entry.1 {
            *entry = (class, cost, node);
        }
    } else {
        groups.push((class, cost, node));
    }
}
