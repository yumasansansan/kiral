// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later
//
// This file is part of KIRAL. The SPDX lines name its copyright holder and its
// license, in the machine-readable form of the REUSE specification: the GNU
// General Public License, version 3 or any later version
// (LICENSES/GPL-3.0-or-later.txt).

//! zenz, Zenzai's model of kana-kanji conversion, used alone: the prompt of
//! zenz-v3 and greedy decoding, with no dictionary.
//!
//! The prompt is `U+EE02 left context U+EE00 reading U+EE01`, the reading in
//! katakana and the context left out where there is none, as Zenzai writes it
//! for zenz-v3; the model writes the conversion after it. Decoding takes the
//! most likely token after another and stops at the end of text, or at any of
//! the characters U+EE00 to U+EE0F that the prompts are made of: zenz-v3.2 does
//! not end its conversions, and goes on to a prompt of its own instead.
//!
//! zenz is read from a directory as Hugging Face publishes it ([`Zenz::load`]),
//! or from a model file of the architecture `gpt2-zenz` and the tokenizer
//! `byte-bpe` ([`Zenz::from_file`]), which [`Zenz::to_file`] writes from zenz in
//! integers (docs/MODEL-FORMAT.md).

use std::path::{Path, PathBuf};

use kiral_kernels::vector::argmax;
use kiral_model::file::{self, Contents, Metadata, ModelFile};
use kiral_model::json;
use kiral_model::safetensors::SafeTensors;

use crate::Error;
use crate::gpt2::{Config, FileTensor, Gpt2, Precision, Session};
use crate::tokenizer::Tokenizer;

/// The architecture of zenz's model files.
pub const ARCHITECTURE: &str = "gpt2-zenz";

/// The tokenizer of zenz's model files.
pub const TOKENIZER: &str = "byte-bpe";

/// The tag before the reading.
pub const INPUT: char = '\u{ee00}';
/// The tag after the reading, after which the conversion is written.
pub const OUTPUT: char = '\u{ee01}';
/// The tag before the left context.
pub const CONTEXT: char = '\u{ee02}';

/// The most characters of the left context the prompt takes, as Zenzai does:
/// the last forty.
pub const CONTEXT_LENGTH: usize = 40;

/// A text as Zenzai gives it to zenz: spaces as ideographic spaces and newlines
/// left out, since the vocabulary has no word for either byte.
#[must_use]
pub fn preprocess(text: &str) -> String {
    text.chars()
        .filter(|&character| character != '\n')
        .map(|character| if character == ' ' { '\u{3000}' } else { character })
        .collect()
}

/// The prompt of zenz-v3 for a reading in katakana and the text to its left.
#[must_use]
pub fn prompt(reading: &str, left_context: &str) -> String {
    let skipped = left_context.chars().count().saturating_sub(CONTEXT_LENGTH);
    let context = left_context.chars().skip(skipped).collect::<String>();
    let mut prompt = String::new();
    if !context.is_empty() {
        prompt.push(CONTEXT);
        prompt.push_str(&context);
    }
    prompt.push(INPUT);
    prompt.push_str(reading);
    prompt.push(OUTPUT);
    preprocess(&prompt)
}

/// zenz: its network and its tokenizer.
#[derive(Debug, Clone)]
pub struct Zenz<'a> {
    /// The network.
    pub model: Gpt2<'a>,
    /// The tokenizer.
    pub tokenizer: Tokenizer,
}

fn read(directory: &Path, name: &str) -> Result<Vec<u8>, Error> {
    let path = directory.join(name);
    std::fs::read(&path).map_err(|error| Error::Io(path, error))
}

impl Zenz<'static> {
    /// The model of a directory as Hugging Face publishes zenz: `config.json`,
    /// `model.safetensors`, `vocab.json`, `merges.txt` and
    /// `special_tokens_map.json`; its products computed in `precision`.
    ///
    /// # Errors
    ///
    /// Where a file cannot be read, or is not what it should be.
    pub fn load(directory: &Path, precision: Precision) -> Result<Self, Error> {
        let config = read(directory, "config.json")?;
        let config = json::parse(&config).map_err(|error| Error::Json("config.json", error))?;
        let config = config
            .as_object()
            .ok_or_else(|| Error::Config("a config.json that is not an object".to_owned()))?;
        let config = Config::from_json(config)?;
        let weights = read(directory, "model.safetensors")?;
        let weights = SafeTensors::parse(&weights).map_err(Error::SafeTensors)?;
        let model = Gpt2::from_safetensors(&weights, config, precision)?;
        let merges = read(directory, "merges.txt")?;
        let merges =
            String::from_utf8(merges).map_err(|_| Error::Tokenizer("a merges.txt that is not UTF-8".to_owned()))?;
        let tokenizer = Tokenizer::from_files(
            &read(directory, "vocab.json")?,
            &merges,
            &read(directory, "special_tokens_map.json")?,
        )?;
        if tokenizer.len() != config.vocabulary {
            return Err(Error::Tokenizer(format!(
                "a vocabulary of {} words for a model of {}",
                tokenizer.len(),
                config.vocabulary
            )));
        }
        Ok(Self { model, tokenizer })
    }
}

impl<'a> Zenz<'a> {
    /// The model of a model file of zenz's architecture and tokenizer: its
    /// weights read where they lie, its products in integers.
    ///
    /// # Errors
    ///
    /// Where the file is of another architecture or tokenizer, or its model or
    /// its tokenizer are not what they should be.
    pub fn from_file(file: &ModelFile<'a>) -> Result<Self, Error> {
        if file.architecture() != ARCHITECTURE || file.tokenizer() != TOKENIZER {
            return Err(Error::Architecture(format!(
                "of the architecture {} and the tokenizer {}, not {ARCHITECTURE} and {TOKENIZER}",
                file.architecture(),
                file.tokenizer()
            )));
        }
        let model = Gpt2::from_file(file)?;
        let tokenizer = Tokenizer::from_vocabulary(&file.vocabulary())?;
        if tokenizer.len() != model.config().vocabulary {
            return Err(Error::Tokenizer(format!(
                "a vocabulary of {} words for a model of {}",
                tokenizer.len(),
                model.config().vocabulary
            )));
        }
        Ok(Self { model, tokenizer })
    }
}

impl Zenz<'_> {
    /// The model file of zenz in integers: its configuration, its tokenizer,
    /// its tensors as [`Gpt2::to_tensors`] gives them, and the metadata given.
    ///
    /// # Errors
    ///
    /// Where the model is in single precision, or there is more of something
    /// than the format counts.
    pub fn to_file(&self, metadata: &Metadata<'_>) -> Result<Vec<u8>, Error> {
        let tensors = self.model.to_tensors()?;
        let tensors: Vec<file::Tensor<'_>> = tensors.iter().map(FileTensor::tensor).collect();
        let words: Vec<&[u8]> = self.tokenizer.words().collect();
        let merges = self.tokenizer.merges();
        let configuration = self.model.config().to_bytes()?;
        let specials = self.tokenizer.specials();
        file::write(&Contents {
            architecture: ARCHITECTURE,
            tokenizer: TOKENIZER,
            metadata: *metadata,
            configuration: &configuration,
            words: &words,
            merges: &merges,
            specials: file::Specials {
                unknown: specials.unknown,
                end: specials.end,
                begin: specials.begin,
                padding: specials.padding,
            },
            tensors: &tensors,
        })
        .map_err(Error::Write)
    }

    /// The directory a path names: the path, or the directory of a file in it.
    #[must_use]
    pub fn directory_of(path: &Path) -> PathBuf {
        if path.is_file() {
            path.parent().map_or_else(PathBuf::new, Path::to_path_buf)
        } else {
            path.to_path_buf()
        }
    }
}

/// How a conversion ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stop {
    /// At the end of text.
    End,
    /// At a character of the tags, which is not part of the conversion.
    Tag,
    /// At the most tokens it was allowed.
    Limit,
}

/// A conversion of the model.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Conversion {
    /// The text it wrote.
    pub text: String,
    /// How it ended.
    pub stop: Stop,
    /// The tokens of the prompt.
    pub prompt_tokens: usize,
    /// The tokens it wrote, the end of text not counted.
    pub tokens: usize,
    /// The prompt's tokens, then those it wrote.
    pub sequence: Vec<u32>,
}

/// Whether bytes of UTF-8 hold a character of the tags, U+EE00 to U+EE0F, and
/// where: their encodings are EE B8 80 to EE B8 8F, which no other character's
/// bytes hold, UTF-8 telling the first byte of a character from the others.
#[must_use]
pub fn tag_in(bytes: &[u8]) -> Option<usize> {
    bytes
        .windows(3)
        .position(|window| window[0] == 0xee && window[1] == 0xb8 && (0x80..=0x8f).contains(&window[2]))
}

/// zenz converting one reading after another, its session kept between them.
#[derive(Debug)]
pub struct Converter<'m> {
    zenz: &'m Zenz<'m>,
    session: Session<'m>,
    logits: Vec<f32>,
    tokens: Vec<u32>,
}

impl<'m> Converter<'m> {
    /// A converter of the model with room for `capacity` positions: those of the
    /// longest prompt and its conversion.
    ///
    /// # Panics
    ///
    /// Where `capacity` is more than the positions of the model.
    #[must_use]
    pub fn new(zenz: &'m Zenz<'m>, capacity: usize) -> Self {
        Self {
            zenz,
            session: Session::new(&zenz.model, capacity),
            logits: vec![0.0; zenz.model.config().vocabulary],
            tokens: Vec::new(),
        }
    }

    /// The model's conversion of a reading in katakana, greedily, with at most
    /// `limit` tokens.
    ///
    /// # Errors
    ///
    /// Where the prompt and the tokens allowed pass the room of the session.
    pub fn convert(&mut self, reading: &str, left_context: &str, limit: usize) -> Result<Conversion, Error> {
        let tokenizer = &self.zenz.tokenizer;
        let end = tokenizer.specials().end;
        self.tokens.clear();
        tokenizer.encode(&prompt(reading, left_context), &mut self.tokens);
        let prompt_tokens = self.tokens.len();
        self.session.truncate(0);
        self.session.feed(&self.tokens, &mut self.logits)?;
        let mut written = Vec::new();
        let mut tokens = 0;
        let stop = loop {
            let next = argmax(&self.logits).and_then(|next| u32::try_from(next).ok());
            let Some(next) = next.filter(|&next| next != end) else {
                break Stop::End;
            };
            tokens += 1;
            self.tokens.push(next);
            written.extend_from_slice(tokenizer.piece(next));
            if let Some(at) = tag_in(&written) {
                written.truncate(at);
                break Stop::Tag;
            }
            if tokens >= limit {
                break Stop::Limit;
            }
            self.session.feed(&[next], &mut self.logits)?;
        };
        Ok(Conversion {
            text: String::from_utf8_lossy(&written).into_owned(),
            stop,
            prompt_tokens,
            tokens,
            sequence: self.tokens.clone(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prompts_are_zenz_v3s() {
        assert_eq!(prompt("カナ", ""), "\u{ee00}カナ\u{ee01}");
        assert_eq!(
            prompt("カナ", "左の 文\n脈"),
            "\u{ee02}左の\u{3000}文脈\u{ee00}カナ\u{ee01}"
        );
        let long = "あ".repeat(30) + &"い".repeat(30);
        let expected = format!("\u{ee02}{}{}\u{ee00}ヨミ\u{ee01}", "あ".repeat(10), "い".repeat(30));
        assert_eq!(prompt("ヨミ", &long), expected);
    }

    #[test]
    fn tags_are_found_in_bytes() {
        assert_eq!(tag_in("漢字\u{ee00}カ".as_bytes()), Some(6));
        assert_eq!(tag_in("\u{ee0f}".as_bytes()), Some(0));
        assert_eq!(tag_in("\u{ee10}\u{edff}漢".as_bytes()), None);
        assert_eq!(tag_in(&[0xee, 0xb8]), None);
    }
}
