// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later
//
// This file is part of KIRAL. The SPDX lines name its copyright holder and its
// license, in the machine-readable form of the REUSE specification: the GNU
// General Public License, version 3 or any later version
// (LICENSES/GPL-3.0-or-later.txt).

//! The parsers take any bytes: inputs made by changing valid ones at random --
//! bits flipped, bytes put in, taken out, repeated, the end cut off -- are read
//! or refused, never a panic. The inputs that once made a parser fail are kept
//! in fuzz/regressions/<parser>/, each named for what it holds, and read on
//! every run.

use std::path::Path;

use kiral_model::file::{self, Contents, Kind, Metadata, ModelFile, Panels, Specials, Tensor};
use kiral_model::json;
use kiral_model::safetensors::SafeTensors;

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

    /// A number below `bound`, which is not 0.
    fn below(&mut self, bound: usize) -> usize {
        let bound = u64::try_from(bound).expect("a small bound");
        usize::try_from(self.next() % bound).expect("below a usize")
    }

    fn byte(&mut self) -> u8 {
        self.next().to_le_bytes()[0]
    }
}

/// The input changed in one to four ways at random.
fn mutate(random: &mut Random, input: &[u8]) -> Vec<u8> {
    let mut bytes = input.to_vec();
    for _ in 0..=random.below(4) {
        let at = random.below(bytes.len() + 1);
        match random.below(6) {
            0 if at < bytes.len() => bytes[at] ^= 1 << random.below(8),
            1 => bytes.insert(at, random.byte()),
            2 if at < bytes.len() => {
                let end = (at + 1 + random.below(8)).min(bytes.len());
                bytes.drain(at..end);
            }
            3 if at < bytes.len() => {
                let end = (at + 1 + random.below(16)).min(bytes.len());
                let repeated = bytes[at..end].to_vec();
                bytes.splice(at..at, repeated);
            }
            4 => bytes.truncate(at),
            _ => {
                let token: &[u8] =
                    [&b"{"[..], b"}", b"[", b"]", b"\"", b"\\u", b",", b":", b"-0e", b"null"][random.below(10)];
                bytes.splice(at..at, token.iter().copied());
            }
        }
    }
    bytes
}

/// The inputs kept in fuzz/regressions/<parser>/, none where there are none.
fn regressions(parser: &str) -> Vec<Vec<u8>> {
    let directory = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../fuzz/regressions")
        .join(parser);
    let Ok(entries) = std::fs::read_dir(&directory) else {
        return Vec::new();
    };
    entries
        .map(|entry| std::fs::read(entry.expect("a directory entry").path()).expect("a regression input"))
        .collect()
}

fn safetensors_seed() -> Vec<u8> {
    let header = br#"{"__metadata__":{"format":"pt"},"w":{"dtype":"BF16","shape":[2,3],"data_offsets":[0,12]},"b":{"dtype":"F32","shape":[3],"data_offsets":[12,24]}}"#;
    let length = u64::try_from(header.len()).expect("a short header");
    [&length.to_le_bytes()[..], header, &[0x3f; 24]].concat()
}

/// A small model file: floats of both kinds and weights in panels, a
/// vocabulary of special words, bytes and merges.
fn model_file_seed() -> Vec<u8> {
    let panels = Panels {
        columns: 16,
        group: 64,
        bits: 8,
        fitted: false,
        rotation: 4,
    };
    let weights = vec![0x11; panels.length(5, 3).expect("a length")];
    let tensors = [
        Tensor {
            name: "e",
            kind: Kind::Bf16,
            shape: vec![2, 3],
            bytes: &[0x80, 0x3f, 0, 0x40, 0x40, 0x40, 0x80, 0x3f, 0, 0, 0, 0x80],
        },
        Tensor {
            name: "n",
            kind: Kind::F32,
            shape: vec![2],
            bytes: &[0, 0, 0x80, 0x3f, 0, 0, 0, 0xc0],
        },
        Tensor {
            name: "w",
            kind: Kind::Panels(panels),
            shape: vec![5, 3],
            bytes: &weights,
        },
    ];
    let contents = Contents {
        architecture: "gpt2-zenz",
        tokenizer: "byte-bpe",
        metadata: Metadata {
            name: "seed",
            license: "CC0-1.0",
            source: "here",
            notice: "none",
        },
        configuration: &[1, 0, 0, 0, 2, 0, 0, 0],
        words: &[b"", b"", b"a", b"b", b"ab"],
        merges: &[[2, 3, 4]],
        specials: Specials {
            unknown: 0,
            end: 1,
            begin: Some(1),
            padding: None,
        },
        tensors: &tensors,
    };
    file::write(&contents).expect("a model file")
}

#[test]
fn json_takes_any_bytes() {
    let seeds: [&[u8]; 5] = [
        br#"{"dtype":"BF16","shape":[768,2304],"data_offsets":[0,3538944]}"#,
        "{\"\\u00e3\\u0081\\u00ae\": 240, \"[UNK]\": 0, \"ア\": 1}".as_bytes(),
        br#"[[[[]]], {"a": [true, false, null]}, -0.5e-3, "\ud83d\ude00"]"#,
        br#"{"index": "7", "context_text": "", "expected_output": ["a", "b"]}"#,
        b"  12345678901234567890123  ",
    ];
    let mut random = Random(11);
    for seed in seeds {
        assert!(json::parse(seed).is_ok(), "a seed that is not JSON");
        for _ in 0..4000 {
            let _ = json::parse(&mutate(&mut random, seed));
        }
    }
    for input in regressions("json") {
        let _ = json::parse(&input);
    }
}

#[test]
fn safetensors_takes_any_bytes() {
    let seed = safetensors_seed();
    assert!(SafeTensors::parse(&seed).is_ok(), "a seed that is not a file");
    let mut random = Random(12);
    for _ in 0..20_000 {
        let input = mutate(&mut random, &seed);
        if let Ok(file) = SafeTensors::parse(&input) {
            for tensor in file.tensors() {
                let _ = file.tensor(&tensor.name).expect("a tensor listed").to_f32();
            }
        }
    }
    for input in regressions("safetensors") {
        let _ = SafeTensors::parse(&input);
    }
}

#[test]
fn model_files_take_any_bytes() {
    let seed = model_file_seed();
    assert!(
        ModelFile::parse(&seed).is_ok_and(|model| model.verify()),
        "a seed that is not a file"
    );
    let mut random = Random(13);
    for _ in 0..20_000 {
        let input = mutate(&mut random, &seed);
        if let Ok(model) = ModelFile::parse(&input) {
            for tensor in model.tensors() {
                let _ = tensor.floats().map(|floats| floats.to_vec());
            }
            let vocabulary = model.vocabulary();
            for word in 0..=vocabulary.len() {
                let _ = vocabulary.word(word);
            }
            let _ = vocabulary.merges().count();
            let _ = model.verify();
        }
    }
    for input in regressions("model-file") {
        let _ = ModelFile::parse(&input);
    }
}
