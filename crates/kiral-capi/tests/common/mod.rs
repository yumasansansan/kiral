// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later
//
// This file is part of KIRAL. The SPDX lines name its copyright holder and its
// license, in the machine-readable form of the REUSE specification: the GNU
// General Public License, version 3 or any later version
// (LICENSES/GPL-3.0-or-later.txt).

//! What the C ABI's tests share: a small dictionary file, and the model file of
//! a small zenz of random weights, as the engine's tests make them.

#![allow(dead_code, reason = "each test file takes what it needs of these")]

use kiral_dict::file::{Unknown, build};
use kiral_dict::mozc::Matrix;
use kiral_dict::{Metadata, Word};
use kiral_kernels::quantized::Scheme;
use kiral_model::file;
use kiral_model::json;
use kiral_model::safetensors::SafeTensors;
use kiral_nn::gpt2::{Config, Gpt2, Precision};
use kiral_nn::tokenizer::Tokenizer;
use kiral_nn::zenz::Zenz;

/// `SplitMix64`: a fixed stream of numbers, the same on every system.
pub struct Random(pub u64);

impl Random {
    pub fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    pub fn below(&mut self, bound: usize) -> usize {
        usize::try_from(self.next() % u64::try_from(bound).expect("small")).expect("small")
    }

    /// A value in [-1, 1) of 16 significant bits.
    pub fn unit(&mut self) -> f32 {
        let bits = u16::try_from(self.next() >> 48).expect("16 bits");
        f32::from(bits) / 32_768.0 - 1.0
    }
}

/// A dictionary of a few words over か and な, of random costs.
pub fn dictionary(random: &mut Random) -> Vec<u8> {
    let words: Vec<Word> = [
        ("か", "甲"),
        ("な", "乙"),
        ("かな", "仮名"),
        ("かな", "金"),
        ("なか", "中"),
    ]
    .iter()
    .map(|&(reading, surface)| Word {
        reading: reading.to_owned(),
        surface: surface.to_owned(),
        lid: 1,
        rid: 1,
        cost: u16::try_from(random.below(5000)).expect("small"),
        flags: 0,
    })
    .collect();
    let matrix = Matrix {
        ids: 2,
        costs: vec![0; 4],
    };
    let metadata = Metadata {
        name: "test",
        license: "CC0-1.0",
        source: "made by the test",
        notice: "",
    };
    build(&words, &matrix, Unknown { id: 1, cost: 8000 }, &metadata).expect("a dictionary")
}

/// The model file of a small zenz of random weights.
pub fn model_file(random: &mut Random) -> Vec<u8> {
    let metadata = file::Metadata {
        name: "test",
        license: "CC0-1.0",
        source: "made by the test",
        notice: "",
    };
    zenz(random).to_file(&metadata).expect("a model file")
}

/// The special words of the byte-level vocabulary: the unknown word and the
/// end of text.
pub const UNKNOWN: u32 = 0;
pub const END: u32 = 1;

/// The words of the byte-level vocabulary: two special words, then each byte
/// as a word of its own, then the words made of the bytes of each character of
/// `joined`, a merge for each byte after the first.
pub fn vocabulary(joined: &str) -> (Vec<Vec<u8>>, Vec<[u32; 3]>) {
    let mut words: Vec<Vec<u8>> = vec![Vec::new(), Vec::new()];
    words.extend((0..=255u8).map(|byte| vec![byte]));
    let mut merges = Vec::new();
    let id = |words: &Vec<Vec<u8>>, piece: &[u8]| {
        u32::try_from(words.iter().position(|word| word == piece).expect("a word")).expect("small")
    };
    for character in joined.chars() {
        let mut buffer = [0; 4];
        let bytes = character.encode_utf8(&mut buffer).as_bytes();
        for end in 2..=bytes.len() {
            let left = id(&words, &bytes[..end - 1]);
            let right = id(&words, &bytes[end - 1..end]);
            words.push(bytes[..end].to_vec());
            let made = u32::try_from(words.len() - 1).expect("small");
            merges.push([left, right, made]);
        }
    }
    (words, merges)
}

/// The tokenizer of [`vocabulary`], read from a model file of it.
pub fn tokenizer(joined: &str) -> Tokenizer {
    let (words, merges) = vocabulary(joined);
    let words: Vec<&[u8]> = words.iter().map(Vec::as_slice).collect();
    let bytes = file::write(&file::Contents {
        architecture: "test",
        tokenizer: "byte-bpe",
        metadata: file::Metadata {
            name: "",
            license: "",
            source: "",
            notice: "",
        },
        configuration: &[],
        words: &words,
        merges: &merges,
        specials: file::Specials {
            unknown: UNKNOWN,
            end: END,
            begin: None,
            padding: None,
        },
        tensors: &[],
    })
    .expect("a model file");
    let model = file::ModelFile::parse(&bytes).expect("a model file");
    Tokenizer::from_vocabulary(&model.vocabulary()).expect("a tokenizer")
}

const LAYERS: usize = 2;
const HEADS: usize = 2;
const WIDTH: usize = 16;
pub const POSITIONS: usize = 256;

/// zenz of a small GPT-2 of random weights, in integers, and the byte-level
/// tokenizer with no merges.
pub fn zenz(random: &mut Random) -> Zenz<'static> {
    let tokenizer = tokenizer("");
    let vocabulary = tokenizer.len();
    let mut tensors: Vec<(String, Vec<usize>, Vec<f32>)> = Vec::new();
    let mut add = |name: String, shape: Vec<usize>, scale: f32, offset: f32| {
        let count = shape.iter().product();
        let values = (0..count).map(|_| offset + scale * random.unit()).collect();
        tensors.push((name, shape, values));
    };
    add("transformer.wte.weight".to_owned(), vec![vocabulary, WIDTH], 0.5, 0.0);
    add("transformer.wpe.weight".to_owned(), vec![POSITIONS, WIDTH], 0.25, 0.0);
    for layer in 0..LAYERS {
        let name = |part: &str| format!("transformer.h.{layer}.{part}");
        add(name("ln_1.weight"), vec![WIDTH], 0.25, 1.0);
        add(name("ln_1.bias"), vec![WIDTH], 0.25, 0.0);
        add(name("attn.c_attn.weight"), vec![WIDTH, 3 * WIDTH], 0.25, 0.0);
        add(name("attn.c_attn.bias"), vec![3 * WIDTH], 0.25, 0.0);
        add(name("attn.c_proj.weight"), vec![WIDTH, WIDTH], 0.25, 0.0);
        add(name("attn.c_proj.bias"), vec![WIDTH], 0.25, 0.0);
        add(name("ln_2.weight"), vec![WIDTH], 0.25, 1.0);
        add(name("ln_2.bias"), vec![WIDTH], 0.25, 0.0);
        add(name("mlp.c_fc.weight"), vec![WIDTH, 4 * WIDTH], 0.25, 0.0);
        add(name("mlp.c_fc.bias"), vec![4 * WIDTH], 0.25, 0.0);
        add(name("mlp.c_proj.weight"), vec![4 * WIDTH, WIDTH], 0.125, 0.0);
        add(name("mlp.c_proj.bias"), vec![WIDTH], 0.25, 0.0);
    }
    add("transformer.ln_f.weight".to_owned(), vec![WIDTH], 0.25, 1.0);
    add("transformer.ln_f.bias".to_owned(), vec![WIDTH], 0.25, 0.0);
    let mut entries = Vec::new();
    let mut data = Vec::new();
    for (name, shape, values) in &tensors {
        let begin = data.len();
        data.extend(values.iter().flat_map(|value| value.to_le_bytes()));
        let shape = shape.iter().map(ToString::to_string).collect::<Vec<_>>().join(",");
        entries.push(format!(
            r#""{name}":{{"dtype":"F32","shape":[{shape}],"data_offsets":[{begin},{}]}}"#,
            data.len()
        ));
    }
    let header = format!("{{{}}}", entries.join(","));
    let length = u64::try_from(header.len()).expect("a short header");
    let bytes = [&length.to_le_bytes()[..], header.as_bytes(), &data].concat();
    let config = format!(
        r#"{{"model_type": "gpt2", "activation_function": "gelu_new", "n_layer": {LAYERS}, "n_head": {HEADS},
            "n_embd": {WIDTH}, "n_positions": {POSITIONS}, "vocab_size": {vocabulary},
            "layer_norm_epsilon": 1e-05}}"#
    );
    let config = json::parse(config.as_bytes()).expect("JSON");
    let config = Config::from_json(config.as_object().expect("an object")).expect("a configuration");
    let model = Gpt2::from_safetensors(
        &SafeTensors::parse(&bytes).expect("a file"),
        config,
        Precision::Integers(Scheme::INT8),
    )
    .expect("a model");
    Zenz { model, tokenizer }
}
