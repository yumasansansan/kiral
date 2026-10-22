// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later
//
// This file is part of KIRAL. The SPDX lines name its copyright holder and its
// license, in the machine-readable form of the REUSE specification: the GNU
// General Public License, version 3 or any later version
// (LICENSES/GPL-3.0-or-later.txt).

//! The model file: a header, then five sections, each beginning on a multiple
//! of 64 bytes, every number little-endian (docs/MODEL-FORMAT.md).
//!
//! The header is the magic `KIRALMDL`, the version (1), four reserved bytes,
//! the SHA-256 of every byte from offset 48 on, the names of the architecture
//! and of the tokenizer (sixteen bytes of ASCII each), and the offset and
//! length of each section (eight bytes each): the metadata, the architecture's
//! configuration, the tokenizer, the tensors' entries and names, and the
//! tensors' bytes. A tensor is floats, of single precision or bfloat16, or a
//! linear layer's weights quantized and packed in panels as the kernels read
//! them; each tensor's bytes begin on a multiple of 64 bytes, so that a file
//! mapped from the start of a page gives the kernels panels on the lines of the
//! caches.
//!
//! [`write`] makes a file. [`ModelFile::parse`] reads one where it lies: it
//! checks the header, the sections, the tokenizer's tables and every tensor's
//! entry -- its bytes within the data, of the length its type and shape make,
//! overlapping no other's -- but leaves the digest to [`ModelFile::verify`],
//! which reads the whole file.

use std::collections::HashSet;
use std::fmt;

use crate::{half, sha256};

/// The file's first eight bytes.
pub const MAGIC: &[u8; 8] = b"KIRALMDL";

/// The version of the format this reads and writes.
pub const VERSION: u32 = 1;

/// The first byte the digest covers: the one after it.
pub const DIGESTED: usize = 48;

/// The bytes of a name of the header: the architecture's, the tokenizer's.
pub const NAME: usize = 16;

/// The sections, in their order.
pub const SECTIONS: usize = 5;

/// The bytes of the header before its table of sections.
pub const TABLE: usize = DIGESTED + 2 * NAME;

/// The bytes of the header.
pub const HEADER: usize = TABLE + 16 * SECTIONS;

/// The multiple of bytes each section and each tensor's bytes begin on: a line
/// of the caches.
pub const ALIGNMENT: usize = 64;

/// The bytes of a tensor's entry.
pub const ENTRY: usize = 64;

/// The most lengths of a tensor's shape.
pub const RANK: usize = 4;

/// The word that stands for no special word.
pub const NONE: u32 = u32::MAX;

/// The section of the metadata.
pub const META: usize = 0;
/// The section of the architecture's configuration.
pub const CONFIGURATION: usize = 1;
/// The section of the tokenizer.
pub const TOKENIZER: usize = 2;
/// The section of the tensors' entries and names.
pub const TENSORS: usize = 3;
/// The section of the tensors' bytes.
pub const DATA: usize = 4;

/// The bytes of the tokenizer's section before the starts of its words.
const TOKENIZER_COUNTS: usize = 24;

/// The bytes of the tensors' section before their entries.
const TENSORS_COUNTS: usize = 8;

/// What a model file says of itself.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Metadata<'a> {
    /// Its name.
    pub name: &'a str,
    /// The license of its weights, as an SPDX expression.
    pub license: &'a str,
    /// Where its weights come from.
    pub source: &'a str,
    /// The notices that go with any copy of it.
    pub notice: &'a str,
}

/// The special words of a tokenizer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Specials {
    /// The unknown word, for a byte that no word is.
    pub unknown: u32,
    /// The word that ends a text.
    pub end: u32,
    /// The word that begins one, where there is one.
    pub begin: Option<u32>,
    /// The word of padding, where there is one.
    pub padding: Option<u32>,
}

/// How a linear layer's weights are packed and quantized.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Panels {
    /// The columns of a panel.
    pub columns: u16,
    /// The inputs of a group, a multiple of 4.
    pub group: u16,
    /// The bits of a weight, from 2 to 8.
    pub bits: u8,
    /// Whether each group's scale is the one of least squared error, rather
    /// than that of its greatest magnitude.
    pub fitted: bool,
    /// The order of the inputs' rotation, a power of four from 4; 0 where they
    /// are not rotated.
    pub rotation: u32,
}

impl Panels {
    /// The bytes of a layer's weights of `depth` inputs and `width` outputs:
    /// its panels' groups, each its integers and its columns' scales and
    /// offsets; none past a `usize`, or of a panel or group of nothing.
    #[must_use]
    pub fn length(&self, depth: usize, width: usize) -> Option<usize> {
        let columns = usize::from(self.columns);
        let group = usize::from(self.group);
        if columns == 0 || group == 0 {
            return None;
        }
        let group_bytes = group.checked_mul(columns)?.checked_add(8 * columns)?;
        width
            .div_ceil(columns)
            .checked_mul(depth.div_ceil(group))?
            .checked_mul(group_bytes)
    }

    /// Whether the format takes the layout and the scheme.
    fn is_valid(&self) -> bool {
        self.columns > 0
            && self.group > 0
            && self.group.is_multiple_of(4)
            && (2..=8).contains(&self.bits)
            && (self.rotation == 0
                || (self.rotation >= 4
                    && self.rotation.is_power_of_two()
                    && self.rotation.trailing_zeros().is_multiple_of(2)))
    }
}

/// What a tensor's bytes hold.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    /// Floats of single precision.
    F32,
    /// Floats of bfloat16: the upper halves of single precision's bits.
    Bf16,
    /// A linear layer's weights, its inputs by its outputs, quantized and
    /// packed in panels.
    Panels(Panels),
}

impl Kind {
    /// The bytes of a tensor of this kind and shape; none where the kind takes
    /// no such shape -- of no lengths, of more than [`RANK`], of a length 0, or
    /// for weights in panels of other than two -- or the bytes pass a `usize`.
    #[must_use]
    pub fn length(&self, shape: &[usize]) -> Option<usize> {
        if shape.is_empty() || shape.len() > RANK || shape.contains(&0) {
            return None;
        }
        let count = shape
            .iter()
            .try_fold(1usize, |count, &length| count.checked_mul(length))?;
        match self {
            Self::F32 => count.checked_mul(4),
            Self::Bf16 => count.checked_mul(2),
            Self::Panels(panels) => match *shape {
                [depth, width] => panels.length(depth, width),
                _ => None,
            },
        }
    }

    /// The number of the type in an entry.
    fn code(&self) -> u32 {
        match self {
            Self::F32 => 1,
            Self::Bf16 => 2,
            Self::Panels(_) => 3,
        }
    }
}

/// A tensor of a model file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Tensor<'a> {
    /// Its name.
    pub name: &'a str,
    /// What its bytes hold.
    pub kind: Kind,
    /// Its shape.
    pub shape: Vec<usize>,
    /// Its bytes.
    pub bytes: &'a [u8],
}

impl<'a> Tensor<'a> {
    /// Its floats, where it is floats.
    #[must_use]
    pub fn floats(&self) -> Option<Floats<'a>> {
        match self.kind {
            Kind::F32 => Some(Floats {
                bytes: self.bytes,
                bf16: false,
            }),
            Kind::Bf16 => Some(Floats {
                bytes: self.bytes,
                bf16: true,
            }),
            Kind::Panels(_) => None,
        }
    }
}

/// Floats of single precision or bfloat16, read where they lie.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Floats<'a> {
    bytes: &'a [u8],
    bf16: bool,
}

impl Floats<'_> {
    /// The bytes of a float.
    fn width(&self) -> usize {
        if self.bf16 { 2 } else { 4 }
    }

    /// The number of floats.
    #[must_use]
    pub fn len(&self) -> usize {
        self.bytes.len() / self.width()
    }

    /// Whether there are none.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.bytes.len() < self.width()
    }

    /// Writes the floats from the `start`-th on, as many as `output` holds, in
    /// single precision, which holds every one of them exactly.
    ///
    /// # Panics
    ///
    /// Where they pass the last float.
    pub fn read(&self, start: usize, output: &mut [f32]) {
        let width = self.width();
        let bytes = &self.bytes[start * width..][..output.len() * width];
        if self.bf16 {
            for (value, bytes) in output.iter_mut().zip(bytes.as_chunks::<2>().0) {
                *value = half::bf16_to_f32(u16::from_le_bytes(*bytes));
            }
        } else {
            for (value, bytes) in output.iter_mut().zip(bytes.as_chunks::<4>().0) {
                *value = f32::from_le_bytes(*bytes);
            }
        }
    }

    /// Every float, in single precision.
    #[must_use]
    pub fn to_vec(&self) -> Vec<f32> {
        let mut values = vec![0.0; self.len()];
        self.read(0, &mut values);
        values
    }
}

/// A tokenizer's words, merges and special words, read where they lie.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Vocabulary<'a> {
    specials: Specials,
    /// The start of each word's bytes, and their end.
    starts: &'a [u8],
    /// The merges, twelve bytes each.
    merges: &'a [u8],
    /// The words' bytes.
    pieces: &'a [u8],
}

impl<'a> Vocabulary<'a> {
    /// The words.
    #[must_use]
    pub fn len(&self) -> usize {
        self.starts.len() / 4 - 1
    }

    /// Whether there are no words.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The bytes of a word; none for a word out of the vocabulary.
    #[must_use]
    pub fn word(&self, word: usize) -> Option<&'a [u8]> {
        let start = index(read_u32(self.starts, 4 * word)?);
        let end = index(read_u32(self.starts, 4 * word.checked_add(1)?)?);
        self.pieces.get(start..end)
    }

    /// The merges, in their order: the left word, the right word and the word
    /// they make.
    pub fn merges(&self) -> impl Iterator<Item = [u32; 3]> + 'a {
        self.merges.as_chunks::<12>().0.iter().map(|merge| {
            let word = |at: usize| u32::from_le_bytes([merge[at], merge[at + 1], merge[at + 2], merge[at + 3]]);
            [word(0), word(4), word(8)]
        })
    }

    /// The number of merges.
    #[must_use]
    pub fn merge_count(&self) -> usize {
        self.merges.len() / 12
    }

    /// The special words.
    #[must_use]
    pub fn specials(&self) -> Specials {
        self.specials
    }
}

/// Why bytes are not a model file that can be read.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    /// Not a model file, or one of another version.
    NotModel(&'static str),
    /// A section out of the file, out of place, or not what it should hold.
    Section(usize, &'static str),
    /// A tensor's entry, by its number, that is not what it should be.
    Tensor(usize, &'static str),
}

impl fmt::Display for Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotModel(what) => write!(formatter, "not a model file: {what}"),
            Self::Section(section, what) => write!(formatter, "section {section}: {what}"),
            Self::Tensor(tensor, what) => write!(formatter, "tensor {tensor}: {what}"),
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

/// A name of the header: printable ASCII, then zeros to its sixteenth byte.
fn read_name(bytes: &[u8]) -> Option<&str> {
    let length = bytes.iter().position(|&byte| byte == 0).unwrap_or(bytes.len());
    let (name, rest) = bytes.split_at(length);
    (length > 0 && name.iter().all(u8::is_ascii_graphic) && rest.iter().all(|&byte| byte == 0))
        .then(|| std::str::from_utf8(name).ok())
        .flatten()
}

/// The four strings of the metadata: the name, the license, the source and the
/// notice, each its length and its bytes.
fn read_metadata(meta: &[u8]) -> Result<[&str; 4], Error> {
    let mut texts = [""; 4];
    let mut at = 0;
    for text in &mut texts {
        let length = read_u32(meta, at)
            .map(index)
            .ok_or(Error::Section(META, "a string's length past its end"))?;
        let start = at + 4;
        let bytes = start
            .checked_add(length)
            .and_then(|end| meta.get(start..end))
            .ok_or(Error::Section(META, "a string past its end"))?;
        *text = std::str::from_utf8(bytes).map_err(|_| Error::Section(META, "a string that is not UTF-8"))?;
        at = start + length;
    }
    Ok(texts)
}

/// The tokenizer's section: its counts and special words checked, its starts
/// going from 0 to its bytes' end, never back, and its merges of words it has.
fn read_vocabulary(section: &[u8]) -> Result<Vocabulary<'_>, Error> {
    let failed = |what| Error::Section(TOKENIZER, what);
    let number = |at| read_u32(section, at).ok_or(failed("shorter than its counts"));
    let words = index(number(0)?);
    let merges = index(number(4)?);
    let word = |value: u32| (index(value) < words).then_some(value);
    let unknown = word(number(8)?).ok_or(failed("an unknown word out of the vocabulary"))?;
    let end = word(number(12)?).ok_or(failed("an end out of the vocabulary"))?;
    let optional = |value: u32| -> Result<Option<u32>, Error> {
        if value == NONE {
            Ok(None)
        } else {
            word(value)
                .map(Some)
                .ok_or(failed("a special word out of the vocabulary"))
        }
    };
    let begin = optional(number(16)?)?;
    let padding = optional(number(20)?)?;
    let merges_at = words
        .checked_add(1)
        .and_then(|starts| starts.checked_mul(4))
        .and_then(|starts| starts.checked_add(TOKENIZER_COUNTS))
        .ok_or(failed("more words than it holds"))?;
    let pieces_at = merges
        .checked_mul(12)
        .and_then(|merges| merges.checked_add(merges_at))
        .ok_or(failed("more merges than it holds"))?;
    let starts = section
        .get(TOKENIZER_COUNTS..merges_at)
        .ok_or(failed("starts past its end"))?;
    let merge_bytes = section.get(merges_at..pieces_at).ok_or(failed("merges past its end"))?;
    // The words' bytes are what the merges leave of the section.
    let pieces = &section[pieces_at..];
    let mut previous = 0;
    for (number, start) in starts.as_chunks::<4>().0.iter().enumerate() {
        let start = index(u32::from_le_bytes(*start));
        if start < previous || start > pieces.len() || (number == 0 && start != 0) {
            return Err(failed("starts out of order"));
        }
        previous = start;
    }
    if previous != pieces.len() {
        return Err(failed("starts that do not end at its end"));
    }
    let vocabulary = Vocabulary {
        specials: Specials {
            unknown,
            end,
            begin,
            padding,
        },
        starts,
        merges: merge_bytes,
        pieces,
    };
    if vocabulary
        .merges()
        .any(|merge| merge.iter().any(|&value| word(value).is_none()))
    {
        return Err(failed("a merge of a word out of the vocabulary"));
    }
    Ok(vocabulary)
}

/// A tensor's entry, its name among the names and its bytes among the data;
/// and the offset of its bytes.
fn read_entry<'a>(entry: &[u8; ENTRY], names: &'a [u8], data: &'a [u8]) -> Result<(Tensor<'a>, usize), &'static str> {
    let number = |at| read_u32(entry, at).map(index).ok_or("a short entry");
    let name_at = number(0)?;
    let name = name_at
        .checked_add(number(4)?)
        .and_then(|end| names.get(name_at..end))
        .ok_or("a name past the names' end")?;
    let name = std::str::from_utf8(name).map_err(|_| "a name that is not UTF-8")?;
    let rank = number(12)?;
    if !(1..=RANK).contains(&rank) {
        return Err("a rank out of range");
    }
    let lengths: Vec<usize> = (0..RANK).map(|axis| number(16 + 4 * axis)).collect::<Result<_, _>>()?;
    if lengths[..rank].contains(&0) || lengths[rank..].iter().any(|&length| length != 0) {
        return Err("a shape out of place");
    }
    let shape = lengths[..rank].to_vec();
    let scheme = &entry[48..];
    let kind = match number(8)? {
        1 | 2 if scheme.iter().any(|&byte| byte != 0) => return Err("floats with a scheme"),
        1 => Kind::F32,
        2 => Kind::Bf16,
        3 => {
            let fitted = match scheme[5] {
                0 => false,
                1 => true,
                _ => return Err("a choice of scales out of range"),
            };
            let panels = Panels {
                columns: read_u16(scheme, 0).ok_or("a short entry")?,
                group: read_u16(scheme, 2).ok_or("a short entry")?,
                bits: scheme[4],
                fitted,
                rotation: read_u32(scheme, 8).ok_or("a short entry")?,
            };
            if !panels.is_valid() || read_u16(scheme, 6) != Some(0) || read_u32(scheme, 12) != Some(0) {
                return Err("a layout or scheme of panels out of range");
            }
            Kind::Panels(panels)
        }
        _ => return Err("a type out of range"),
    };
    let offset = read_u64(entry, 32)
        .and_then(|offset| usize::try_from(offset).ok())
        .ok_or("an offset out of range")?;
    let length = read_u64(entry, 40)
        .and_then(|length| usize::try_from(length).ok())
        .ok_or("a length out of range")?;
    if offset % ALIGNMENT != 0 {
        return Err("bytes out of place");
    }
    if kind.length(&shape) != Some(length) {
        return Err("bytes of another length than its type and shape make");
    }
    let bytes = offset
        .checked_add(length)
        .and_then(|end| data.get(offset..end))
        .ok_or("bytes past the data's end")?;
    Ok((
        Tensor {
            name,
            kind,
            shape,
            bytes,
        },
        offset,
    ))
}

/// A model file, read where it lies.
#[derive(Debug, Clone)]
pub struct ModelFile<'a> {
    bytes: &'a [u8],
    architecture: &'a str,
    tokenizer: &'a str,
    metadata: Metadata<'a>,
    configuration: &'a [u8],
    vocabulary: Vocabulary<'a>,
    tensors: Vec<Tensor<'a>>,
}

impl<'a> ModelFile<'a> {
    /// The model file of bytes: its header, sections, tokenizer and tensors
    /// checked, not its digest.
    ///
    /// # Errors
    ///
    /// Where the bytes are not a model file of this version, a section is out
    /// of place or not what it should hold, or a tensor's entry is not what it
    /// should be: its name not UTF-8 or another's, its type or shape not one
    /// the format takes, its bytes out of place, of another length than its
    /// type and shape make, or overlapping another's.
    pub fn parse(bytes: &'a [u8]) -> Result<Self, Error> {
        if bytes.len() < HEADER {
            return Err(Error::NotModel("shorter than a header"));
        }
        if &bytes[..8] != MAGIC {
            return Err(Error::NotModel("no magic"));
        }
        if read_u32(bytes, 8) != Some(VERSION) {
            return Err(Error::NotModel("another version"));
        }
        if read_u32(bytes, 12) != Some(0) {
            return Err(Error::NotModel("reserved bytes that are not 0"));
        }
        let architecture =
            read_name(&bytes[DIGESTED..DIGESTED + NAME]).ok_or(Error::NotModel("an architecture of no name"))?;
        let tokenizer = read_name(&bytes[DIGESTED + NAME..TABLE]).ok_or(Error::NotModel("a tokenizer of no name"))?;
        let section = |number: usize| -> Result<&'a [u8], Error> {
            let at = TABLE + 16 * number;
            let offset = read_u64(bytes, at).and_then(|offset| usize::try_from(offset).ok());
            let length = read_u64(bytes, at + 8).and_then(|length| usize::try_from(length).ok());
            let (Some(offset), Some(length)) = (offset, length) else {
                return Err(Error::Section(number, "an offset or a length out of range"));
            };
            if offset % ALIGNMENT != 0 || offset < HEADER {
                return Err(Error::Section(number, "an offset out of place"));
            }
            offset
                .checked_add(length)
                .and_then(|end| bytes.get(offset..end))
                .ok_or(Error::Section(number, "past the end of the file"))
        };
        let [name, license, source, notice] = read_metadata(section(META)?)?;
        let configuration = section(CONFIGURATION)?;
        let vocabulary = read_vocabulary(section(TOKENIZER)?)?;
        let table = section(TENSORS)?;
        let data = section(DATA)?;
        let tensor_count = read_u32(table, 0)
            .map(index)
            .ok_or(Error::Section(TENSORS, "shorter than its count"))?;
        if read_u32(table, 4) != Some(0) {
            return Err(Error::Section(TENSORS, "reserved bytes that are not 0"));
        }
        let entries = tensor_count
            .checked_mul(ENTRY)
            .and_then(|entries| entries.checked_add(TENSORS_COUNTS))
            .and_then(|end| table.get(TENSORS_COUNTS..end))
            .ok_or(Error::Section(TENSORS, "entries past its end"))?;
        let names = &table[TENSORS_COUNTS + entries.len()..];
        let (tensors, offsets): (Vec<_>, Vec<_>) = entries
            .as_chunks::<ENTRY>()
            .0
            .iter()
            .enumerate()
            .map(|(number, entry)| read_entry(entry, names, data).map_err(|what| Error::Tensor(number, what)))
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .unzip();
        check_tensors(&tensors, &offsets)?;
        Ok(Self {
            bytes,
            architecture,
            tokenizer,
            metadata: Metadata {
                name,
                license,
                source,
                notice,
            },
            configuration,
            vocabulary,
            tensors,
        })
    }

    /// The name of the architecture.
    #[must_use]
    pub fn architecture(&self) -> &'a str {
        self.architecture
    }

    /// The name of the tokenizer.
    #[must_use]
    pub fn tokenizer(&self) -> &'a str {
        self.tokenizer
    }

    /// What the file says of itself.
    #[must_use]
    pub fn metadata(&self) -> Metadata<'a> {
        self.metadata
    }

    /// The architecture's configuration, as the architecture lays it out.
    #[must_use]
    pub fn configuration(&self) -> &'a [u8] {
        self.configuration
    }

    /// The tokenizer's words, merges and special words.
    #[must_use]
    pub fn vocabulary(&self) -> Vocabulary<'a> {
        self.vocabulary
    }

    /// The tensors, in the order of their entries.
    #[must_use]
    pub fn tensors(&self) -> &[Tensor<'a>] {
        &self.tensors
    }

    /// The tensor of a name.
    #[must_use]
    pub fn tensor(&self, name: &str) -> Option<&Tensor<'a>> {
        self.tensors.iter().find(|tensor| tensor.name == name)
    }

    /// The bytes of the file.
    #[must_use]
    pub fn bytes(&self) -> &'a [u8] {
        self.bytes
    }

    /// The digest the file carries.
    #[must_use]
    pub fn digest(&self) -> [u8; 32] {
        let mut digest = [0; 32];
        digest.copy_from_slice(&self.bytes[16..DIGESTED]);
        digest
    }

    /// Whether the file's bytes from offset 48 on have the digest it carries:
    /// reads every byte of the file.
    #[must_use]
    pub fn verify(&self) -> bool {
        sha256::digest(&self.bytes[DIGESTED..]) == self.digest()
    }
}

/// Checks that no two tensors have the same name, or bytes that overlap, each
/// tensor's bytes at its offset in the data.
fn check_tensors(tensors: &[Tensor<'_>], offsets: &[usize]) -> Result<(), Error> {
    let mut by_name: Vec<usize> = (0..tensors.len()).collect();
    by_name.sort_by_key(|&number| tensors[number].name);
    if let Some(pair) = by_name
        .windows(2)
        .find(|pair| tensors[pair[0]].name == tensors[pair[1]].name)
    {
        return Err(Error::Tensor(pair[1], "the name of another"));
    }
    let mut by_place: Vec<usize> = (0..tensors.len()).collect();
    by_place.sort_by_key(|&number| offsets[number]);
    if let Some(pair) = by_place
        .windows(2)
        .find(|pair| offsets[pair[0]] + tensors[pair[0]].bytes.len() > offsets[pair[1]])
    {
        return Err(Error::Tensor(pair[1], "bytes that overlap another's"));
    }
    Ok(())
}

/// What a model file is made of.
#[derive(Debug, Clone, Copy)]
pub struct Contents<'a> {
    /// The architecture's name: printable ASCII, at most sixteen bytes.
    pub architecture: &'a str,
    /// The tokenizer's name: printable ASCII, at most sixteen bytes.
    pub tokenizer: &'a str,
    /// What the file says of itself.
    pub metadata: Metadata<'a>,
    /// The architecture's configuration, as the architecture lays it out.
    pub configuration: &'a [u8],
    /// The bytes of each word, none for a special word.
    pub words: &'a [&'a [u8]],
    /// The merges, in their order: the left word, the right word and the word
    /// they make.
    pub merges: &'a [[u32; 3]],
    /// The special words.
    pub specials: Specials,
    /// The tensors, in the order their bytes are laid out in.
    pub tensors: &'a [Tensor<'a>],
}

/// Why a model file was not written.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BuildError {
    /// A name of the header that is not one to sixteen bytes of printable
    /// ASCII.
    Name(String),
    /// A tensor, named, that the format does not take as it is.
    Tensor(String, &'static str),
    /// A word of the tokenizer out of its vocabulary.
    Word(u32),
    /// More of something than the format counts.
    Size(&'static str),
}

impl fmt::Display for BuildError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Name(name) => write!(formatter, "the name {name:?}, not 1 to 16 bytes of printable ASCII"),
            Self::Tensor(name, what) => write!(formatter, "the tensor {name}: {what}"),
            Self::Word(word) => write!(formatter, "the word {word}, out of the vocabulary"),
            Self::Size(what) => write!(formatter, "more {what} than the format counts"),
        }
    }
}

impl std::error::Error for BuildError {}

fn count(value: usize, what: &'static str) -> Result<u32, BuildError> {
    u32::try_from(value).map_err(|_| BuildError::Size(what))
}

fn count64(value: usize) -> Result<u64, BuildError> {
    u64::try_from(value).map_err(|_| BuildError::Size("bytes"))
}

/// A name of the header in its sixteen bytes.
fn header_name(name: &str) -> Result<[u8; NAME], BuildError> {
    let mut bytes = [0; NAME];
    if name.is_empty() || name.len() > NAME || !name.bytes().all(|byte| byte.is_ascii_graphic()) {
        return Err(BuildError::Name(name.to_owned()));
    }
    bytes[..name.len()].copy_from_slice(name.as_bytes());
    Ok(bytes)
}

/// The tokenizer's section.
fn vocabulary_bytes(contents: &Contents<'_>) -> Result<Vec<u8>, BuildError> {
    let words = count(contents.words.len(), "words")?;
    let check = |word: u32| {
        if word < words {
            Ok(word)
        } else {
            Err(BuildError::Word(word))
        }
    };
    let specials = contents.specials;
    let mut bytes = Vec::new();
    bytes.extend(words.to_le_bytes());
    bytes.extend(count(contents.merges.len(), "merges")?.to_le_bytes());
    bytes.extend(check(specials.unknown)?.to_le_bytes());
    bytes.extend(check(specials.end)?.to_le_bytes());
    for optional in [specials.begin, specials.padding] {
        bytes.extend(optional.map_or(Ok(NONE), check)?.to_le_bytes());
    }
    let mut start = 0usize;
    bytes.extend(0u32.to_le_bytes());
    for word in contents.words {
        start += word.len();
        bytes.extend(count(start, "bytes of words")?.to_le_bytes());
    }
    for merge in contents.merges {
        for &word in merge {
            bytes.extend(check(word)?.to_le_bytes());
        }
    }
    for word in contents.words {
        bytes.extend(*word);
    }
    Ok(bytes)
}

/// The tensors' section, and the offset of each tensor's bytes in the data and
/// the data's length.
fn tensor_table(tensors: &[Tensor<'_>]) -> Result<(Vec<u8>, Vec<usize>, usize), BuildError> {
    let mut names = HashSet::new();
    let mut entries = Vec::with_capacity(TENSORS_COUNTS + ENTRY * tensors.len());
    let mut name_bytes: Vec<u8> = Vec::new();
    let mut offsets = Vec::with_capacity(tensors.len());
    let mut end = 0usize;
    entries.extend(count(tensors.len(), "tensors")?.to_le_bytes());
    entries.extend([0; 4]);
    for tensor in tensors {
        let failed = |what| BuildError::Tensor(tensor.name.to_owned(), what);
        if tensor.name.is_empty() || !names.insert(tensor.name) {
            return Err(failed("a name that is empty or another's"));
        }
        if let Kind::Panels(panels) = tensor.kind
            && !panels.is_valid()
        {
            return Err(failed("a layout or scheme of panels the format does not take"));
        }
        if tensor.kind.length(&tensor.shape) != Some(tensor.bytes.len()) {
            return Err(failed("bytes of another length than its type and shape make"));
        }
        let offset = end.next_multiple_of(ALIGNMENT);
        end = offset + tensor.bytes.len();
        offsets.push(offset);
        entries.extend(count(name_bytes.len(), "bytes of names")?.to_le_bytes());
        entries.extend(count(tensor.name.len(), "bytes of names")?.to_le_bytes());
        name_bytes.extend(tensor.name.as_bytes());
        entries.extend(tensor.kind.code().to_le_bytes());
        entries.extend(count(tensor.shape.len(), "lengths")?.to_le_bytes());
        for axis in 0..RANK {
            let length = tensor.shape.get(axis).copied().unwrap_or(0);
            entries.extend(count(length, "values along an axis")?.to_le_bytes());
        }
        entries.extend(count64(offset)?.to_le_bytes());
        entries.extend(count64(tensor.bytes.len())?.to_le_bytes());
        let mut scheme = [0u8; 16];
        if let Kind::Panels(panels) = tensor.kind {
            scheme[0..2].copy_from_slice(&panels.columns.to_le_bytes());
            scheme[2..4].copy_from_slice(&panels.group.to_le_bytes());
            scheme[4] = panels.bits;
            scheme[5] = u8::from(panels.fitted);
            scheme[8..12].copy_from_slice(&panels.rotation.to_le_bytes());
        }
        entries.extend(scheme);
    }
    entries.extend(name_bytes);
    Ok((entries, offsets, end))
}

/// The bytes of a string as the metadata holds it: its length, then it.
fn push_string(bytes: &mut Vec<u8>, text: &str) -> Result<(), BuildError> {
    bytes.extend(count(text.len(), "bytes of metadata")?.to_le_bytes());
    bytes.extend(text.as_bytes());
    Ok(())
}

/// The file of a model's contents, its digest computed.
///
/// # Errors
///
/// Where a name of the header is not one to sixteen bytes of printable ASCII, a
/// word of the tokenizer is out of its vocabulary, a tensor has an empty name
/// or another's, a layout or scheme of panels the format does not take, or
/// bytes of another length than its type and shape make, or there is more of
/// something than the format counts.
pub fn write(contents: &Contents<'_>) -> Result<Vec<u8>, BuildError> {
    let architecture = header_name(contents.architecture)?;
    let tokenizer = header_name(contents.tokenizer)?;
    let metadata = contents.metadata;
    let mut meta = Vec::new();
    for text in [metadata.name, metadata.license, metadata.source, metadata.notice] {
        push_string(&mut meta, text)?;
    }
    let vocabulary = vocabulary_bytes(contents)?;
    let (table, offsets, data_length) = tensor_table(contents.tensors)?;
    let small = [&meta[..], contents.configuration, &vocabulary, &table];
    let capacity = HEADER + small.iter().map(|section| section.len() + ALIGNMENT).sum::<usize>() + data_length;
    let mut file = Vec::with_capacity(capacity);
    file.extend(MAGIC);
    file.extend(VERSION.to_le_bytes());
    file.extend([0; 4]);
    file.extend([0; 32]);
    file.extend(architecture);
    file.extend(tokenizer);
    file.resize(HEADER, 0);
    let mut sections = Vec::with_capacity(16 * SECTIONS);
    for section in small {
        file.resize(file.len().next_multiple_of(ALIGNMENT), 0);
        sections.extend(count64(file.len())?.to_le_bytes());
        sections.extend(count64(section.len())?.to_le_bytes());
        file.extend(section);
    }
    file.resize(file.len().next_multiple_of(ALIGNMENT), 0);
    let data = file.len();
    sections.extend(count64(data)?.to_le_bytes());
    sections.extend(count64(data_length)?.to_le_bytes());
    for (tensor, offset) in contents.tensors.iter().zip(offsets) {
        file.resize(data + offset, 0);
        file.extend(tensor.bytes);
    }
    file[TABLE..HEADER].copy_from_slice(&sections);
    let digest = sha256::digest(&file[DIGESTED..]);
    file[16..DIGESTED].copy_from_slice(&digest);
    Ok(file)
}
