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
//! them, so that a parenthesis typed full-width stays full-width.
//! [`Lattice::reaches`] tells whether a text is the text of some path: the
//! oracle, the share of a benchmark's accepted conversions that the dictionary
//! can make at all.
#![forbid(unsafe_code)]

use kiral_dict::{Dictionary, normalize};

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
    pub cost: u16,
    /// Its surface; none where it is written as the input wrote it.
    pub surface: Option<&'a str>,
    /// Whether the dictionary has it.
    pub known: bool,
}

/// The lattice of a reading.
#[derive(Debug, Clone)]
pub struct Lattice<'a> {
    /// The input as it is shown: its katakana as hiragana.
    shown: Vec<char>,
    nodes: Vec<Node<'a>>,
    /// The nodes that begin at each position: `starts[p]..starts[p + 1]`.
    starts: Vec<usize>,
    /// The nodes that end at each position.
    ends: Vec<Vec<usize>>,
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

impl<'a> Lattice<'a> {
    /// The lattice of a reading, in any form, over a dictionary.
    #[must_use]
    pub fn new(dictionary: &Dictionary<'a>, reading: &str) -> Self {
        let shown: Vec<char> = reading.chars().map(normalize::display).collect();
        let lookup: Vec<char> = reading.chars().map(normalize::lookup).collect();
        let unknown = dictionary.unknown();
        let mut nodes = Vec::new();
        let mut starts = Vec::with_capacity(lookup.len() + 1);
        let mut ends = vec![Vec::new(); lookup.len() + 1];
        let mut stretch = String::new();
        for begin in 0..lookup.len() {
            starts.push(nodes.len());
            for (length, entries) in dictionary.prefixes(&lookup[begin..]) {
                let end = begin + length;
                stretch.clear();
                stretch.extend(&lookup[begin..end]);
                for entry in entries {
                    ends[end].push(nodes.len());
                    nodes.push(Node {
                        begin,
                        end,
                        lid: entry.lid,
                        rid: entry.rid,
                        cost: entry.cost,
                        surface: (entry.surface != stretch).then_some(entry.surface),
                        known: true,
                    });
                }
            }
            ends[begin + 1].push(nodes.len());
            nodes.push(Node {
                begin,
                end: begin + 1,
                lid: unknown.id,
                rid: unknown.id,
                cost: unknown.cost,
                surface: None,
                known: false,
            });
        }
        starts.push(nodes.len());
        Self {
            shown,
            nodes,
            starts,
            ends,
        }
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

    /// The numbers of the nodes that begin at a position.
    #[must_use]
    pub fn beginning_at(&self, position: usize) -> std::ops::Range<usize> {
        match (self.starts.get(position), self.starts.get(position + 1)) {
            (Some(&start), Some(&end)) => start..end,
            _ => 0..0,
        }
    }

    /// Appends the text of a node to `text`.
    pub fn push_text(&self, node: &Node<'a>, text: &mut String) {
        match node.surface {
            Some(surface) => text.push_str(surface),
            None => text.extend(&self.shown[node.begin..node.end]),
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

    /// The cheapest path, by Viterbi; the empty path for an empty reading.
    #[must_use]
    pub fn best(&self, dictionary: &Dictionary<'_>) -> Path {
        let length = self.len();
        let mut best = vec![i64::MAX; self.nodes.len()];
        let mut back = vec![BEGINNING; self.nodes.len()];
        // The cheapest way to each class on the right of the words that end at a
        // position, and to each class on the left of those that begin there.
        let mut rights: Vec<(u16, i64, usize)> = Vec::new();
        let mut lefts: Vec<(u16, i64, usize)> = Vec::new();
        for position in 0..length {
            rights.clear();
            if position == 0 {
                rights.push((0, 0, BEGINNING));
            } else {
                for &node in &self.ends[position] {
                    group(&mut rights, self.nodes[node].rid, best[node], node);
                }
            }
            lefts.clear();
            for node in self.beginning_at(position) {
                let lid = self.nodes[node].lid;
                let (cost, from) = if let Some(&(_, cost, from)) = lefts.iter().find(|(left, _, _)| *left == lid) {
                    (cost, from)
                } else {
                    let (cost, from) = rights
                        .iter()
                        .map(|&(rid, cost, from)| (cost + i64::from(dictionary.connection(rid, lid)), from))
                        .min_by_key(|&(cost, _)| cost)
                        .unwrap_or((i64::MAX / 2, BEGINNING));
                    lefts.push((lid, cost, from));
                    (cost, from)
                };
                best[node] = cost + i64::from(self.nodes[node].cost);
                back[node] = from;
            }
        }
        let Some((cost, last)) = self.ends[length]
            .iter()
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
    /// alone, where `known_only`, or of characters of no reading as well.
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
