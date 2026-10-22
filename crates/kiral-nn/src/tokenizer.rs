// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later
//
// This file is part of KIRAL. The SPDX lines name its copyright holder and its
// license, in the machine-readable form of the REUSE specification: the GNU
// General Public License, version 3 or any later version
// (LICENSES/GPL-3.0-or-later.txt).

//! zenz's tokenizer: GPT-2's byte-level BPE.
//!
//! Each byte of a text is a symbol of its own -- GPT-2 writes the 256 bytes as
//! printable characters, which the vocabulary's words are spelt in -- and the
//! merges join two adjacent symbols into the word they make, the earliest merge
//! of the list first, until none applies. A byte the vocabulary has no word for
//! is the unknown word.
//!
//! GPT-2 splits a text into pieces before it merges, by a regular expression of
//! Unicode's categories, and merges within each piece. zenz's merges never join
//! two characters, only the bytes of one, so that splitting at every character
//! gives the same tokens; this tokenizer does that, and refuses a vocabulary
//! with a merge that joins characters, for which it would not. The special words
//! (`</s>` and the like) are never made from text, as llama.cpp does not make
//! them when Zenzai tokenizes.

use std::collections::HashMap;

use kiral_model::json::{self, Value};

use crate::Error;

/// GPT-2's characters for the bytes: the printable bytes of Latin-1 as
/// themselves, and the others as the characters from U+0100 on, in their order.
fn byte_characters() -> [char; 256] {
    let mut characters = ['\0'; 256];
    let mut next = 0x100;
    for (byte, character) in (0..=255u8).zip(characters.iter_mut()) {
        *character = if matches!(byte, b'!'..=b'~' | 0xa1..=0xac | 0xae..=0xff) {
            char::from(byte)
        } else {
            next += 1;
            char::from_u32(next - 1).expect("a character below U+0144")
        };
    }
    characters
}

/// Whether the bytes lie within one character of UTF-8: the first bytes of one,
/// or bytes of its continuation.
fn within_one_character(bytes: &[u8]) -> bool {
    let continuation = |byte: &u8| byte & 0xc0 == 0x80;
    let Some((first, rest)) = bytes.split_first() else {
        return false;
    };
    let length = match first {
        0x00..=0x7f => 1,
        0x80..=0xbf => return bytes.len() <= 3 && bytes.iter().all(continuation),
        0xc0..=0xdf => 2,
        0xe0..=0xef => 3,
        0xf0..=0xf7 => 4,
        _ => return false,
    };
    bytes.len() <= length && rest.iter().all(continuation)
}

/// The special words of a vocabulary.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Specials {
    /// The unknown word, for a byte without a word.
    pub unknown: u32,
    /// The word that ends a text.
    pub end: u32,
    /// The word that begins one, where the vocabulary has it.
    pub begin: Option<u32>,
    /// The word of padding, where the vocabulary has it.
    pub padding: Option<u32>,
}

/// A tokenizer of GPT-2's byte-level BPE.
#[derive(Debug, Clone)]
pub struct Tokenizer {
    /// The bytes of each word; none for the special words.
    pieces: Vec<Vec<u8>>,
    /// The word of each byte's symbol, the unknown word where there is none.
    bytes: [u32; 256],
    /// For each pair of words a merge joins: its rank in the list, and the word
    /// it makes.
    merges: HashMap<(u32, u32), (u32, u32)>,
    specials: Specials,
}

impl Tokenizer {
    /// The tokenizer of a vocabulary (`vocab.json`), its merges (`merges.txt`)
    /// and its special words (`special_tokens_map.json`).
    ///
    /// # Errors
    ///
    /// Where a file is not what it should be, a word of the merges is not in
    /// the vocabulary, or a merge joins two characters.
    pub fn from_files(vocabulary: &[u8], merges: &str, specials: &[u8]) -> Result<Self, Error> {
        let vocabulary = json::parse(vocabulary).map_err(|error| Error::Json("vocab.json", error))?;
        let vocabulary = vocabulary
            .as_object()
            .ok_or_else(|| Error::Tokenizer("a vocabulary that is not an object".to_owned()))?;
        let mut ids: HashMap<&str, u32> = HashMap::with_capacity(vocabulary.len());
        let mut words: Vec<Option<&str>> = vec![None; vocabulary.len()];
        for (word, id) in vocabulary.iter() {
            let (index, id) = id
                .as_u64()
                .and_then(|id| Some((usize::try_from(id).ok()?, u32::try_from(id).ok()?)))
                .filter(|&(index, _)| index < words.len() && words[index].is_none())
                .ok_or_else(|| Error::Tokenizer(format!("the word {word:?} with an id out of place")))?;
            words[index] = Some(word);
            ids.insert(word, id);
        }
        // As many words as ids, each id below their number and given once: every
        // id has its word.
        let words = words
            .into_iter()
            .collect::<Option<Vec<&str>>>()
            .ok_or_else(|| Error::Tokenizer("an id without a word".to_owned()))?;
        let specials = read_specials(specials, &ids)?;
        let special = [
            Some(specials.unknown),
            Some(specials.end),
            specials.begin,
            specials.padding,
        ];
        let characters = byte_characters();
        let byte_of: HashMap<char, u8> = (0..=255u8).map(|byte| (characters[usize::from(byte)], byte)).collect();
        let mut pieces = Vec::with_capacity(words.len());
        for (word, id) in words.iter().zip(0u32..) {
            if special.contains(&Some(id)) {
                pieces.push(Vec::new());
                continue;
            }
            let piece = word
                .chars()
                .map(|character| byte_of.get(&character).copied())
                .collect::<Option<Vec<u8>>>()
                .ok_or_else(|| Error::Tokenizer(format!("the word {word:?}, not spelt in GPT-2's bytes")))?;
            pieces.push(piece);
        }
        let bytes = characters.map(|character| {
            ids.get(&*character.encode_utf8(&mut [0; 4]))
                .copied()
                .unwrap_or(specials.unknown)
        });
        let mut table = HashMap::new();
        let lines = merges
            .lines()
            .filter(|line| !line.starts_with("#version") && !line.is_empty());
        for (line, rank) in lines.zip(0u32..) {
            let (left, right) = line
                .split_once(' ')
                .ok_or_else(|| Error::Tokenizer(format!("the merge {line:?}, not two words")))?;
            let missing = || Error::Tokenizer(format!("the merge {line:?}, of words not in the vocabulary"));
            let id = |word: &str| ids.get(word).copied().ok_or_else(missing);
            let word = id(&[left, right].concat())?;
            let pair = (id(left)?, id(right)?);
            let piece = usize::try_from(word)
                .ok()
                .and_then(|word| pieces.get(word))
                .ok_or_else(missing)?;
            if !within_one_character(piece) {
                return Err(Error::Tokenizer(format!(
                    "the merge {line:?}, which joins characters: tokens would need GPT-2's splitting of the text"
                )));
            }
            if table.insert(pair, (rank, word)).is_some() {
                return Err(Error::Tokenizer(format!("the merge {line:?}, given twice")));
            }
        }
        Ok(Self {
            pieces,
            bytes,
            merges: table,
            specials,
        })
    }

    /// The tokens of a text, appended to `tokens`.
    pub fn encode(&self, text: &str, tokens: &mut Vec<u32>) {
        let mut symbols = Vec::with_capacity(4);
        for character in text.chars() {
            symbols.clear();
            let mut buffer = [0; 4];
            symbols.extend(
                character
                    .encode_utf8(&mut buffer)
                    .bytes()
                    .map(|byte| self.bytes[usize::from(byte)]),
            );
            // The earliest merge of the adjacent pairs, until none applies.
            while let Some((at, merged)) = symbols
                .windows(2)
                .enumerate()
                .filter_map(|(at, pair)| {
                    self.merges
                        .get(&(pair[0], pair[1]))
                        .map(|&(rank, merged)| (rank, at, merged))
                })
                .min()
                .map(|(_, at, merged)| (at, merged))
            {
                symbols[at] = merged;
                symbols.remove(at + 1);
            }
            tokens.extend_from_slice(&symbols);
        }
    }

    /// The bytes of a token: none for a special word, or a token out of the
    /// vocabulary.
    #[must_use]
    pub fn piece(&self, token: u32) -> &[u8] {
        usize::try_from(token)
            .ok()
            .and_then(|token| self.pieces.get(token))
            .map_or(&[], Vec::as_slice)
    }

    /// The text of tokens: their bytes, as UTF-8, with U+FFFD for what is not.
    #[must_use]
    pub fn decode(&self, tokens: &[u32]) -> String {
        let bytes = tokens
            .iter()
            .flat_map(|&token| self.piece(token))
            .copied()
            .collect::<Vec<_>>();
        String::from_utf8_lossy(&bytes).into_owned()
    }

    /// The special words.
    #[must_use]
    pub fn specials(&self) -> Specials {
        self.specials
    }

    /// The number of words of the vocabulary.
    #[must_use]
    pub fn len(&self) -> usize {
        self.pieces.len()
    }

    /// Whether the vocabulary has no words.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.pieces.is_empty()
    }
}

/// The special words of `special_tokens_map.json`, each a word or an object
/// whose `content` is one.
fn read_specials(bytes: &[u8], ids: &HashMap<&str, u32>) -> Result<Specials, Error> {
    let map = json::parse(bytes).map_err(|error| Error::Json("special_tokens_map.json", error))?;
    let map = map
        .as_object()
        .ok_or_else(|| Error::Tokenizer("special words that are not an object".to_owned()))?;
    let id = |name: &str| -> Result<Option<u32>, Error> {
        let Some(value) = map.get(name) else {
            return Ok(None);
        };
        let word = match value {
            Value::Object(object) => object.get("content").and_then(Value::as_str),
            value => value.as_str(),
        }
        .ok_or_else(|| Error::Tokenizer(format!("the special word {name}, not a word")))?;
        ids.get(word)
            .copied()
            .map(Some)
            .ok_or_else(|| Error::Tokenizer(format!("the special word {name}, {word:?}, not in the vocabulary")))
    };
    let required = |name: &str| id(name)?.ok_or_else(|| Error::Tokenizer(format!("no special word {name}")));
    Ok(Specials {
        unknown: required("unk_token")?,
        end: required("eos_token")?,
        begin: id("bos_token")?,
        padding: id("pad_token")?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const SPECIALS: &[u8] = br#"{"bos_token": "<s>", "eos_token": "</s>", "pad_token": "[PAD]", "unk_token": "[UNK]"}"#;

    /// The word of bytes, spelt in GPT-2's characters.
    fn spelt(bytes: &[u8]) -> String {
        let characters = byte_characters();
        bytes.iter().map(|&byte| characters[usize::from(byte)]).collect()
    }

    /// A vocabulary of the special words, the bytes of "ab", "の" (E3 81 AE) and
    /// "あ" (E3 81 82), and the merges that make the two characters.
    fn tokenizer() -> Tokenizer {
        let mut words = vec![
            "[UNK]".to_owned(),
            "[PAD]".to_owned(),
            "<s>".to_owned(),
            "</s>".to_owned(),
        ];
        for byte in [b'a', b'b', 0xe3, 0x81, 0xae, 0x82] {
            words.push(spelt(&[byte]));
        }
        words.extend([
            spelt(&[0xe3, 0x81]),
            spelt(&[0xe3, 0x81, 0xae]),
            spelt(&[0xe3, 0x81, 0x82]),
        ]);
        let members = words
            .iter()
            .enumerate()
            .map(|(id, word)| format!("{word:?}: {id}"))
            .collect::<Vec<_>>();
        let vocabulary = format!("{{{}}}", members.join(", "));
        let merges = format!(
            "#version: 0.2\n{} {}\n{} {}\n{} {}\n",
            spelt(&[0xe3]),
            spelt(&[0x81]),
            spelt(&[0xe3, 0x81]),
            spelt(&[0xae]),
            spelt(&[0xe3, 0x81]),
            spelt(&[0x82]),
        );
        Tokenizer::from_files(vocabulary.as_bytes(), &merges, SPECIALS).expect("a tokenizer")
    }

    #[test]
    fn the_bytes_are_gpt2s_characters() {
        let characters = byte_characters();
        assert_eq!(characters[usize::from(b'A')], 'A');
        assert_eq!(characters[usize::from(b' ')], '\u{120}');
        assert_eq!(characters[0], '\u{100}');
        assert_eq!(characters[0xad], '\u{143}');
        let mut sorted = characters.to_vec();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), 256);
    }

    #[test]
    fn characters_are_merged_and_unknown_bytes_are_unknown() {
        let tokenizer = tokenizer();
        let mut tokens = Vec::new();
        tokenizer.encode("aのbあc", &mut tokens);
        // c has no word: it is the unknown word, 0.
        assert_eq!(tokens, [4, 11, 5, 12, 0]);
        assert_eq!(tokenizer.decode(&tokens[..4]), "aのbあ");
        // The special words are never made from text, and decode to nothing.
        tokens.clear();
        tokenizer.encode("</s>", &mut tokens);
        assert!(!tokens.contains(&3));
        assert_eq!(tokenizer.decode(&[3, 4]), "a");
        assert_eq!(tokenizer.specials().end, 3);
    }

    #[test]
    fn a_character_cut_short_decodes_to_the_replacement() {
        let tokenizer = tokenizer();
        assert_eq!(tokenizer.decode(&[10, 4]), "\u{fffd}a");
    }

    #[test]
    fn merges_across_characters_are_refused() {
        let vocabulary = format!(
            r#"{{"[UNK]": 0, "</s>": 1, "a": 2, "b": 3, "ab": 4, {:?}: 5}}"#,
            spelt(&[0xe3])
        );
        let specials = br#"{"eos_token": "</s>", "unk_token": "[UNK]"}"#;
        let error = Tokenizer::from_files(vocabulary.as_bytes(), "a b\n", specials).expect_err("refused");
        assert!(error.to_string().contains("joins characters"), "{error}");
        assert!(Tokenizer::from_files(vocabulary.as_bytes(), "a c\n", specials).is_err());
        assert!(Tokenizer::from_files(br#"{"[UNK]": 0, "</s>": 2}"#, "", specials).is_err());
    }

    #[test]
    fn pieces_within_one_character() {
        assert!(within_one_character(b"a"));
        assert!(!within_one_character(b"ab"));
        assert!(within_one_character(&[0xe3, 0x81]));
        assert!(within_one_character(&[0x81, 0xae]));
        assert!(!within_one_character(&[0xe3, 0x81, 0xae, 0xe3]));
        assert!(!within_one_character(&[0xae, b'a']));
        assert!(!within_one_character(&[0xff]));
        assert!(!within_one_character(&[]));
    }
}
