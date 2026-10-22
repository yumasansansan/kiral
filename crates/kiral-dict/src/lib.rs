// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later
//
// This file is part of KIRAL. The SPDX lines name its copyright holder and its
// license, in the machine-readable form of the REUSE specification: the GNU
// General Public License, version 3 or any later version
// (LICENSES/GPL-3.0-or-later.txt).

//! KIRAL's dictionaries: words keyed by their readings, and the costs of the
//! connections between their classes, in a file that is read where it lies.
//!
//! A dictionary is built from Mozc's open-source dictionary ([`mozc`],
//! [`file::build`]) into a file whose readings are a double-array trie over the
//! codes of their characters ([`trie`]) and whose words are entries of a
//! surface, the ids of the classes on their left and right, and a cost. Read
//! ([`Dictionary::parse`]), it takes nothing but the bytes: the trie is walked,
//! and the entries are read, where they lie, so that a file mapped from the
//! disk is looked up without being copied. Its parser takes any bytes: what is
//! not a dictionary is an error, and a corrupt entry is never a panic.
//!
//! Readings are looked up in one form, their katakana as hiragana and the
//! full-width forms of ASCII as ASCII ([`normalize`]).
#![forbid(unsafe_code)]

use std::fmt;
use std::ops::Range;

pub mod file;
pub mod mozc;
pub mod normalize;
pub mod trie;

/// A word as it is built into a dictionary.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Word {
    /// Its reading.
    pub reading: String,
    /// Its surface.
    pub surface: String,
    /// The id of its class on its left.
    pub lid: u16,
    /// The id of its class on its right.
    pub rid: u16,
    /// Its cost: the higher, the less likely.
    pub cost: u16,
    /// Its flags, four bits: [`mozc::SPELLING_CORRECTION`].
    pub flags: u16,
}

/// What a dictionary file says of itself.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Metadata<'a> {
    /// Its name.
    pub name: &'a str,
    /// The license of its words, as an SPDX expression.
    pub license: &'a str,
    /// Where its words come from.
    pub source: &'a str,
    /// The notices that go with any copy of it.
    pub notice: &'a str,
}

/// A word of a dictionary, read where it lies.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Entry<'a> {
    /// Its surface.
    pub surface: &'a str,
    /// The id of its class on its left.
    pub lid: u16,
    /// The id of its class on its right.
    pub rid: u16,
    /// Its cost.
    pub cost: u16,
    /// Its flags.
    pub flags: u16,
}

/// Why bytes are not a dictionary that can be read.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    /// Not a dictionary file, or one of another version.
    NotDictionary(&'static str),
    /// A section out of the file, out of place, or of another length than its
    /// counts say.
    Section(usize, &'static str),
}

impl fmt::Display for Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotDictionary(what) => write!(formatter, "not a dictionary: {what}"),
            Self::Section(section, what) => write!(formatter, "section {section}: {what}"),
        }
    }
}

impl std::error::Error for Error {}

/// A little-endian number at an offset, none past the end.
fn read_u16(bytes: &[u8], offset: usize) -> Option<u16> {
    let bytes: [u8; 2] = bytes.get(offset..offset.checked_add(2)?)?.try_into().ok()?;
    Some(u16::from_le_bytes(bytes))
}

fn read_u32(bytes: &[u8], offset: usize) -> Option<u32> {
    let bytes: [u8; 4] = bytes.get(offset..offset.checked_add(4)?)?.try_into().ok()?;
    Some(u32::from_le_bytes(bytes))
}

fn read_u64(bytes: &[u8], offset: usize) -> Option<u64> {
    let bytes: [u8; 8] = bytes.get(offset..offset.checked_add(8)?)?.try_into().ok()?;
    Some(u64::from_le_bytes(bytes))
}

fn index(value: u32) -> usize {
    usize::try_from(value).expect("a 32-bit index fits a usize")
}

/// Checks that the starts of the readings' entries go from 0 to the entries'
/// end, never back.
fn check_starts(starts: &[u8], entries: u32) -> Result<(), Error> {
    let mut previous = 0;
    for (number, start) in starts.as_chunks::<4>().0.iter().enumerate() {
        let start = u32::from_le_bytes(*start);
        if start < previous || start > entries || (number == 0 && start != 0) {
            return Err(Error::Section(file::STARTS, "starts out of order"));
        }
        previous = start;
    }
    if previous != entries {
        return Err(Error::Section(
            file::STARTS,
            "starts that do not end at the entries' end",
        ));
    }
    Ok(())
}

/// The four strings of the metadata: the name, the license, the source and the
/// notice, each its length and its bytes.
fn read_metadata(meta: &[u8]) -> Result<[&str; 4], Error> {
    let mut texts = [""; 4];
    let mut at = 0;
    for text in &mut texts {
        let length = read_u32(meta, at)
            .map(index)
            .ok_or(Error::Section(file::META, "a string's length past its end"))?;
        let start = at + 4;
        let bytes = start
            .checked_add(length)
            .and_then(|end| meta.get(start..end))
            .ok_or(Error::Section(file::META, "a string past its end"))?;
        *text = std::str::from_utf8(bytes).map_err(|_| Error::Section(file::META, "a string that is not UTF-8"))?;
        at = start + length;
    }
    Ok(texts)
}

/// A dictionary file, read where it lies.
#[derive(Debug, Clone, Copy)]
pub struct Dictionary<'a> {
    metadata: Metadata<'a>,
    ids: u16,
    unknown: file::Unknown,
    readings: u32,
    entries: u32,
    alphabet: &'a [u8],
    units: &'a [u8],
    starts: &'a [u8],
    entry_bytes: &'a [u8],
    strings: &'a [u8],
    matrix: &'a [u8],
}

impl<'a> Dictionary<'a> {
    /// The dictionary of a file's bytes: its header and sections checked.
    ///
    /// # Errors
    ///
    /// Where the bytes are not a dictionary of this version, or a section is out
    /// of place or of another length than the header's counts make it.
    pub fn parse(bytes: &'a [u8]) -> Result<Self, Error> {
        if bytes.len() < file::HEADER {
            return Err(Error::NotDictionary("shorter than a header"));
        }
        if &bytes[..8] != file::MAGIC {
            return Err(Error::NotDictionary("no magic"));
        }
        if read_u32(bytes, 8) != Some(file::VERSION) {
            return Err(Error::NotDictionary("another version"));
        }
        let ids = read_u32(bytes, 12)
            .and_then(|ids| u16::try_from(ids).ok())
            .ok_or(Error::NotDictionary("more ids than 16 bits number"))?;
        let readings = read_u32(bytes, 16).ok_or(Error::NotDictionary("no count of readings"))?;
        let entries = read_u32(bytes, 20).ok_or(Error::NotDictionary("no count of entries"))?;
        let unknown = file::Unknown {
            id: read_u16(bytes, 24).ok_or(Error::NotDictionary("no unknown word"))?,
            cost: read_u16(bytes, 26).ok_or(Error::NotDictionary("no unknown word"))?,
        };
        let section = |number: usize| -> Result<&'a [u8], Error> {
            let at = file::TABLE + 16 * number;
            let offset = read_u64(bytes, at).and_then(|offset| usize::try_from(offset).ok());
            let length = read_u64(bytes, at + 8).and_then(|length| usize::try_from(length).ok());
            let (Some(offset), Some(length)) = (offset, length) else {
                return Err(Error::Section(number, "an offset or a length out of range"));
            };
            if offset % 8 != 0 || offset < file::HEADER {
                return Err(Error::Section(number, "an offset out of place"));
            }
            offset
                .checked_add(length)
                .and_then(|end| bytes.get(offset..end))
                .ok_or(Error::Section(number, "past the end of the file"))
        };
        let meta = section(file::META)?;
        let alphabet = section(file::ALPHABET)?;
        let units = section(file::UNITS)?;
        let starts = section(file::STARTS)?;
        let entry_bytes = section(file::ENTRIES)?;
        let strings = section(file::STRINGS)?;
        let matrix = section(file::MATRIX)?;
        if alphabet.len() % 8 != 0 {
            return Err(Error::Section(file::ALPHABET, "not whole pairs"));
        }
        if units.len() % trie::UNIT != 0 {
            return Err(Error::Section(file::UNITS, "not whole units"));
        }
        if starts.len() != (index(readings) + 1) * 4 {
            return Err(Error::Section(file::STARTS, "not a start for each reading and the end"));
        }
        if entry_bytes.len() != index(entries) * file::ENTRY {
            return Err(Error::Section(file::ENTRIES, "not the entries counted"));
        }
        if matrix.len() != usize::from(ids) * usize::from(ids) * 2 {
            return Err(Error::Section(file::MATRIX, "not the square of the ids"));
        }
        check_starts(starts, entries)?;
        let texts = read_metadata(meta)?;
        let [name, license, source, notice] = texts;
        Ok(Self {
            metadata: Metadata {
                name,
                license,
                source,
                notice,
            },
            ids,
            unknown,
            readings,
            entries,
            alphabet,
            units,
            starts,
            entry_bytes,
            strings,
            matrix,
        })
    }

    /// What the file says of itself.
    #[must_use]
    pub fn metadata(&self) -> Metadata<'a> {
        self.metadata
    }

    /// The number of ids of the classes, the beginning and end of a sentence's
    /// 0 among them.
    #[must_use]
    pub fn ids(&self) -> u16 {
        self.ids
    }

    /// The id and the cost of a word the dictionary does not have.
    #[must_use]
    pub fn unknown(&self) -> file::Unknown {
        self.unknown
    }

    /// The number of distinct readings.
    #[must_use]
    pub fn readings(&self) -> u32 {
        self.readings
    }

    /// The number of entries.
    #[must_use]
    pub fn entries(&self) -> u32 {
        self.entries
    }

    /// The bytes of each section, for the log of a build.
    #[must_use]
    pub fn sizes(&self) -> String {
        format!(
            "alphabet {} characters, trie {} units, entries {} bytes, surfaces {} bytes, matrix {} ids, notice {} bytes",
            self.alphabet.len() / 8,
            self.units.len() / trie::UNIT,
            self.entry_bytes.len(),
            self.strings.len(),
            self.ids,
            self.metadata.notice.len()
        )
    }

    /// The cost of a word of right id `rid` followed by one of left id `lid`;
    /// the most there is for ids the matrix does not have.
    #[must_use]
    pub fn connection(&self, rid: u16, lid: u16) -> u16 {
        let at = (usize::from(rid) * usize::from(self.ids) + usize::from(lid)) * 2;
        if rid >= self.ids || lid >= self.ids {
            return u16::MAX;
        }
        read_u16(self.matrix, at).unwrap_or(u16::MAX)
    }

    /// The code of a character in the trie, where a reading has it.
    #[must_use]
    pub fn code(&self, character: char) -> Option<u16> {
        let pairs = self.alphabet.len() / 8;
        let target = u32::from(character);
        let (mut low, mut high) = (0, pairs);
        while low < high {
            let middle = low + (high - low) / 2;
            let at = middle * 8;
            match read_u32(self.alphabet, at)?.cmp(&target) {
                std::cmp::Ordering::Less => low = middle + 1,
                std::cmp::Ordering::Greater => high = middle,
                std::cmp::Ordering::Equal => {
                    return read_u32(self.alphabet, at + 4).and_then(|code| u16::try_from(code).ok());
                }
            }
        }
        None
    }

    /// The entries of the reading of a number.
    fn entries_of(&self, reading: u32) -> Entries<'a> {
        let at = index(reading) * 4;
        let start = read_u32(self.starts, at).unwrap_or(0);
        let end = read_u32(self.starts, at + 4).unwrap_or(start);
        Entries {
            entry_bytes: self.entry_bytes,
            strings: self.strings,
            range: start..end.max(start),
        }
    }

    /// The readings that begin `input`, which is in its lookup form: the length
    /// of each, in characters, the shortest first, and its entries.
    #[must_use]
    pub fn prefixes<'s>(&'s self, input: &'s [char]) -> Prefixes<'s, 'a> {
        Prefixes {
            dictionary: self,
            input,
            unit: 0,
            position: 0,
        }
    }

    /// The entries of a reading, in any form.
    #[must_use]
    pub fn lookup(&self, reading: &str) -> Option<Entries<'a>> {
        let mut unit = 0;
        for character in reading.chars() {
            unit = trie::child(self.units, unit, self.code(normalize::lookup(character))?)?;
        }
        trie::reading_at(self.units, unit)
            .filter(|&reading| reading < self.readings)
            .map(|reading| self.entries_of(reading))
    }
}

/// The readings that begin an input, and their entries.
#[derive(Debug, Clone)]
pub struct Prefixes<'s, 'a> {
    dictionary: &'s Dictionary<'a>,
    input: &'s [char],
    unit: u32,
    position: usize,
}

impl<'a> Iterator for Prefixes<'_, 'a> {
    type Item = (usize, Entries<'a>);

    fn next(&mut self) -> Option<Self::Item> {
        while let Some(&character) = self.input.get(self.position) {
            let code = self.dictionary.code(character)?;
            self.unit = trie::child(self.dictionary.units, self.unit, code)?;
            self.position += 1;
            if let Some(reading) = trie::reading_at(self.dictionary.units, self.unit)
                && reading < self.dictionary.readings
            {
                return Some((self.position, self.dictionary.entries_of(reading)));
            }
        }
        None
    }
}

/// The entries of a reading, cheapest first: an entry whose surface is not
/// UTF-8, or lies past the surfaces' end, is left out.
#[derive(Debug, Clone)]
pub struct Entries<'a> {
    entry_bytes: &'a [u8],
    strings: &'a [u8],
    range: Range<u32>,
}

impl<'a> Iterator for Entries<'a> {
    type Item = Entry<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        while let Some(number) = self.range.next() {
            let at = index(number) * file::ENTRY;
            let (Some(offset), Some(length), Some(lid), Some(rid), Some(cost)) = (
                read_u32(self.entry_bytes, at),
                read_u16(self.entry_bytes, at + 4),
                read_u16(self.entry_bytes, at + 6),
                read_u16(self.entry_bytes, at + 8),
                read_u16(self.entry_bytes, at + 10),
            ) else {
                continue;
            };
            let start = index(offset);
            let surface = start
                .checked_add(usize::from(length & 0x0fff))
                .and_then(|end| self.strings.get(start..end))
                .and_then(|bytes| std::str::from_utf8(bytes).ok());
            if let Some(surface) = surface {
                return Some(Entry {
                    surface,
                    lid,
                    rid,
                    cost,
                    flags: length >> 12,
                });
            }
        }
        None
    }
}
