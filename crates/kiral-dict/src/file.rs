// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later
//
// This file is part of KIRAL. The SPDX lines name its copyright holder and its
// license, in the machine-readable form of the REUSE specification: the GNU
// General Public License, version 3 or any later version
// (LICENSES/GPL-3.0-or-later.txt).

//! The dictionary file: a header, then seven sections, each beginning on a
//! multiple of eight bytes, every number little-endian (docs/DICT-FORMAT.md).
//!
//! The header is the magic `KIRALDIC`, the version (1), the number of ids of
//! the matrix, of readings and of entries (four bytes each), the id and the cost
//! of a word the dictionary does not have (two bytes each, then four reserved),
//! and the offset and length of each section (eight bytes each): the metadata,
//! the alphabet, the units of the trie, the start of each reading's entries,
//! the entries, their surfaces, and the matrix of the connections' costs.

use std::collections::HashMap;

use crate::mozc::Matrix;
use crate::{Metadata, Word, normalize, trie};

/// The file's first eight bytes.
pub const MAGIC: &[u8; 8] = b"KIRALDIC";

/// The version of the format this reads and writes.
pub const VERSION: u32 = 1;

/// The sections, in their order.
pub const SECTIONS: usize = 7;

/// The bytes of the header before its table of sections.
pub const TABLE: usize = 32;

/// The bytes of the header.
pub const HEADER: usize = TABLE + 16 * SECTIONS;

/// The bytes of an entry.
pub const ENTRY: usize = 12;

/// The longest surface an entry holds, in bytes.
pub const LONGEST_SURFACE: usize = 0x0fff;

/// The section of the metadata.
pub const META: usize = 0;
/// The section of the alphabet.
pub const ALPHABET: usize = 1;
/// The section of the trie's units.
pub const UNITS: usize = 2;
/// The section of the start of each reading's entries.
pub const STARTS: usize = 3;
/// The section of the entries.
pub const ENTRIES: usize = 4;
/// The section of the surfaces.
pub const STRINGS: usize = 5;
/// The section of the connections' costs.
pub const MATRIX: usize = 6;

/// Why a dictionary was not built.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BuildError {
    /// A word with an id the matrix does not have.
    Id(String),
    /// A surface longer than an entry holds.
    Surface(String),
    /// A matrix of other costs than the square of its ids.
    Matrix,
    /// More than the format counts: readings, entries, characters or bytes.
    Size(&'static str),
}

impl std::fmt::Display for BuildError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Id(word) => write!(formatter, "the word {word} has an id the matrix does not"),
            Self::Surface(word) => write!(formatter, "the word {word} has a surface of more than 4095 bytes"),
            Self::Matrix => formatter.write_str("a matrix of other costs than the square of its ids"),
            Self::Size(what) => write!(formatter, "more {what} than the format counts"),
        }
    }
}

impl std::error::Error for BuildError {}

fn count(value: usize, what: &'static str) -> Result<u32, BuildError> {
    u32::try_from(value).map_err(|_| BuildError::Size(what))
}

/// The bytes of a string as the metadata holds it: its length, then it.
fn push_string(bytes: &mut Vec<u8>, text: &str) -> Result<(), BuildError> {
    bytes.extend(count(text.len(), "bytes of metadata")?.to_le_bytes());
    bytes.extend(text.as_bytes());
    Ok(())
}

/// The id and the cost of a word a dictionary does not have: what a lattice
/// takes a character of no reading as.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Unknown {
    /// The id of its class, on its left and on its right.
    pub id: u16,
    /// Its cost.
    pub cost: u16,
}

/// The file of words, the matrix of their connections, the class and cost of a
/// word it does not have, and the metadata: each word's reading in its lookup
/// form, the words of a reading together, cheapest first, and words given twice
/// once.
///
/// # Errors
///
/// Where a word, or the unknown word, has an id the matrix has not, a word has
/// a surface longer than an entry holds or flags of more than four bits, or
/// there is more of something than the format counts.
pub fn build(
    words: &[Word],
    matrix: &Matrix,
    unknown: Unknown,
    metadata: &Metadata<'_>,
) -> Result<Vec<u8>, BuildError> {
    if matrix.costs.len() != usize::from(matrix.ids) * usize::from(matrix.ids) {
        return Err(BuildError::Matrix);
    }
    if unknown.id >= matrix.ids {
        return Err(BuildError::Id("the unknown word".to_owned()));
    }
    for word in words {
        if word.lid >= matrix.ids || word.rid >= matrix.ids {
            return Err(BuildError::Id(format!("{} ({})", word.surface, word.reading)));
        }
        if word.surface.len() > LONGEST_SURFACE {
            return Err(BuildError::Surface(word.reading.clone()));
        }
        if word.flags > 0xf {
            return Err(BuildError::Size("flags"));
        }
    }
    let readings: Vec<String> = words.iter().map(|word| normalize::lookup_form(&word.reading)).collect();
    let codes = alphabet(&readings)?;
    let keys: Vec<Vec<u16>> = readings
        .iter()
        .map(|reading| reading.chars().map(|character| codes[&character]).collect())
        .collect();
    let mut order: Vec<usize> = (0..words.len()).collect();
    let rank = |index: &usize| {
        let word = &words[*index];
        (&keys[*index], word.cost, word.lid, word.rid, &word.surface, word.flags)
    };
    order.sort_by(|a, b| rank(a).cmp(&rank(b)));
    order.dedup_by(|a, b| rank(a) == rank(b));
    let (distinct, starts, entries, strings) = entries(words, &keys, &order)?;
    let units = trie::to_bytes(&trie::build(&distinct));

    let mut meta = Vec::new();
    for text in [metadata.name, metadata.license, metadata.source, metadata.notice] {
        push_string(&mut meta, text)?;
    }
    let mut pairs: Vec<(char, u16)> = codes.into_iter().collect();
    pairs.sort_unstable();
    let alphabet: Vec<u8> = pairs
        .iter()
        .flat_map(|&(character, code)| {
            u32::from(character)
                .to_le_bytes()
                .into_iter()
                .chain(u32::from(code).to_le_bytes())
        })
        .collect();
    let starts: Vec<u8> = starts.iter().flat_map(|start| start.to_le_bytes()).collect();
    let matrix_bytes: Vec<u8> = matrix.costs.iter().flat_map(|cost| cost.to_le_bytes()).collect();
    let mut header = Vec::with_capacity(TABLE);
    header.extend(MAGIC);
    header.extend(VERSION.to_le_bytes());
    header.extend(u32::from(matrix.ids).to_le_bytes());
    header.extend(count(distinct.len(), "readings")?.to_le_bytes());
    header.extend(count(order.len(), "entries")?.to_le_bytes());
    header.extend(unknown.id.to_le_bytes());
    header.extend(unknown.cost.to_le_bytes());
    assemble(
        &header,
        [&meta, &alphabet, &units, &starts, &entries, &strings, &matrix_bytes],
    )
}

/// The code of each character of the readings, the commonest first, so that the
/// trie's nodes, whose children are mostly of them, pack close.
fn alphabet(readings: &[String]) -> Result<HashMap<char, u16>, BuildError> {
    let mut frequencies: HashMap<char, usize> = HashMap::new();
    for reading in readings {
        for character in reading.chars() {
            *frequencies.entry(character).or_default() += 1;
        }
    }
    let mut characters: Vec<(char, usize)> = frequencies.into_iter().collect();
    characters.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    if characters.len() >= usize::from(u16::MAX) {
        return Err(BuildError::Size("characters"));
    }
    Ok(characters
        .iter()
        .zip(1u16..)
        .map(|(&(character, _), code)| (character, code))
        .collect())
}

/// The distinct keys, the start of each one's entries and the end, the entries
/// and their surfaces, each surface once, of the words in `order`.
#[expect(clippy::type_complexity, reason = "the four parts of the file that the entries make")]
fn entries(
    words: &[Word],
    keys: &[Vec<u16>],
    order: &[usize],
) -> Result<(Vec<Vec<u16>>, Vec<u32>, Vec<u8>, Vec<u8>), BuildError> {
    let mut distinct: Vec<Vec<u16>> = Vec::new();
    let mut starts: Vec<u32> = Vec::new();
    let mut entries = Vec::with_capacity(order.len() * ENTRY);
    let mut strings: Vec<u8> = Vec::new();
    let mut placed: HashMap<&str, u32> = HashMap::new();
    for (number, &index) in order.iter().enumerate() {
        if distinct.last() != Some(&keys[index]) {
            distinct.push(keys[index].clone());
            starts.push(count(number, "entries")?);
        }
        let word = &words[index];
        let offset = if let Some(&offset) = placed.get(word.surface.as_str()) {
            offset
        } else {
            let offset = count(strings.len(), "bytes of surfaces")?;
            strings.extend(word.surface.as_bytes());
            placed.insert(&word.surface, offset);
            offset
        };
        let length = u16::try_from(word.surface.len()).map_err(|_| BuildError::Surface(word.reading.clone()))?;
        entries.extend(offset.to_le_bytes());
        entries.extend((length | (word.flags << 12)).to_le_bytes());
        entries.extend(word.lid.to_le_bytes());
        entries.extend(word.rid.to_le_bytes());
        entries.extend(word.cost.to_le_bytes());
    }
    starts.push(count(order.len(), "entries")?);
    Ok((distinct, starts, entries, strings))
}

/// The file of a header's first part and the sections: each section on a
/// multiple of eight bytes, its offset and length in the header's table.
fn assemble(header: &[u8], sections: [&[u8]; SECTIONS]) -> Result<Vec<u8>, BuildError> {
    let mut file = Vec::with_capacity(HEADER + sections.iter().map(|section| section.len() + 8).sum::<usize>());
    file.extend(header);
    file.resize(HEADER, 0);
    let mut table = Vec::with_capacity(16 * SECTIONS);
    for section in sections {
        file.resize(file.len().next_multiple_of(8), 0);
        let offset = u64::try_from(file.len()).map_err(|_| BuildError::Size("bytes"))?;
        let length = u64::try_from(section.len()).map_err(|_| BuildError::Size("bytes"))?;
        table.extend(offset.to_le_bytes());
        table.extend(length.to_le_bytes());
        file.extend(section);
    }
    file[TABLE..HEADER].copy_from_slice(&table);
    Ok(file)
}
