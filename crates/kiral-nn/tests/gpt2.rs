// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later
//
// This file is part of KIRAL. The SPDX lines name its copyright holder and its
// license, in the machine-readable form of the REUSE specification: the GNU
// General Public License, version 3 or any later version
// (LICENSES/GPL-3.0-or-later.txt).

//! GPT-2 in single precision against the same network written plainly in
//! double precision, on a small one of random weights made into a safetensors
//! file here: two blocks of width 16 and two heads. The logits are compared
//! after the tokens are fed all at once, one at a time, and in two parts with
//! the cache cut back between, which takes the cache of keys and values and the
//! positions through every way the session is used. The model in integers,
//! written to a model file and read back where it lies, computes the same
//! logits.

use kiral_kernels::quantized::{Rotation, Scheme};
use kiral_kernels::vector::argmax;
use kiral_model::file::{self, Contents, Kind, Metadata, ModelFile, Panels, Specials, Tensor};
use kiral_model::json;
use kiral_model::safetensors::SafeTensors;
use kiral_nn::gpt2::{CONFIG_BYTES, Config, FileTensor, Gpt2, Precision, Session};
use kiral_nn::zenz::Zenz;

const LAYERS: usize = 2;
const HEADS: usize = 2;
const WIDTH: usize = 16;
const POSITIONS: usize = 32;
const VOCABULARY: usize = 11;

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

    /// A value in [-1, 1) of 16 significant bits, exact in single precision.
    fn unit(&mut self) -> f32 {
        let bits = u16::try_from(self.next() >> 48).expect("16 bits");
        f32::from(bits) / 32_768.0 - 1.0
    }
}

/// The tensors of the model, by Hugging Face's names, with their shapes and
/// values: weights of magnitudes that keep the activations moderate, gains
/// near 1.
fn tensors(random: &mut Random) -> Vec<(String, Vec<usize>, Vec<f32>)> {
    let mut tensors = Vec::new();
    let mut add = |name: String, shape: Vec<usize>, scale: f32, offset: f32| {
        let count = shape.iter().product();
        let values = (0..count).map(|_| offset + scale * random.unit()).collect();
        tensors.push((name, shape, values));
    };
    add("transformer.wte.weight".to_owned(), vec![VOCABULARY, WIDTH], 0.5, 0.0);
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
    tensors
}

/// The tensors as a safetensors file of single precision.
fn file(tensors: &[(String, Vec<usize>, Vec<f32>)]) -> Vec<u8> {
    let mut entries = Vec::new();
    let mut data = Vec::new();
    for (name, shape, values) in tensors {
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
    [&length.to_le_bytes()[..], header.as_bytes(), &data].concat()
}

fn config() -> Config {
    let text = format!(
        r#"{{"model_type": "gpt2", "activation_function": "gelu_new", "n_layer": {LAYERS}, "n_head": {HEADS},
            "n_embd": {WIDTH}, "n_positions": {POSITIONS}, "vocab_size": {VOCABULARY}, "n_inner": null,
            "layer_norm_epsilon": 1e-05, "scale_attn_weights": true}}"#
    );
    let value = json::parse(text.as_bytes()).expect("JSON");
    Config::from_json(value.as_object().expect("an object")).expect("a configuration")
}

type Tensors = [(String, Vec<usize>, Vec<f32>)];

/// A tensor in double precision.
fn get(tensors: &Tensors, name: &str) -> Vec<f64> {
    let (_, _, values) = tensors.iter().find(|(given, _, _)| given == name).expect("a tensor");
    values.iter().copied().map(f64::from).collect()
}

fn count(value: usize) -> f64 {
    f64::from(u32::try_from(value).expect("small"))
}

fn norm(x: &[f64], gain: &[f64], bias: &[f64]) -> Vec<f64> {
    let mean = x.iter().sum::<f64>() / count(x.len());
    let variance = x.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / count(x.len());
    let scale = 1.0 / (variance + 1e-5).sqrt();
    x.iter()
        .zip(gain)
        .zip(bias)
        .map(|((x, g), b)| (x - mean) * scale * g + b)
        .collect()
}

fn linear(x: &[f64], weight: &[f64], bias: &[f64]) -> Vec<f64> {
    let outputs = bias.len();
    (0..outputs)
        .map(|j| {
            bias[j]
                + x.iter()
                    .enumerate()
                    .map(|(i, x)| x * weight[i * outputs + j])
                    .sum::<f64>()
        })
        .collect()
}

fn gelu(x: f64) -> f64 {
    0.5 * x * (1.0 + ((2.0 / std::f64::consts::PI).sqrt() * (x + 0.044_715 * x * x * x)).tanh())
}

/// What the heads of position `t` find, its queries, keys and values those of
/// `projected`.
fn attend(projected: &[Vec<f64>], t: usize) -> Vec<f64> {
    let head = WIDTH / HEADS;
    let mut found = vec![0.0; WIDTH];
    for h in 0..HEADS {
        let query = &projected[t][h * head..][..head];
        let scores: Vec<f64> = (0..=t)
            .map(|s| {
                let key = &projected[s][WIDTH + h * head..][..head];
                query.iter().zip(key).map(|(q, k)| q * k).sum::<f64>() / count(head).sqrt()
            })
            .collect();
        let largest = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let weights: Vec<f64> = scores.iter().map(|s| (s - largest).exp()).collect();
        let total: f64 = weights.iter().sum();
        for (s, weight) in weights.iter().enumerate() {
            for i in 0..head {
                found[h * head + i] += weight / total * projected[s][2 * WIDTH + h * head + i];
            }
        }
    }
    found
}

/// The network in double precision, as plainly as it can be written: the
/// logits after each of the tokens.
fn reference(tensors: &Tensors, tokens: &[usize]) -> Vec<Vec<f64>> {
    let wte = get(tensors, "transformer.wte.weight");
    let wpe = get(tensors, "transformer.wpe.weight");
    let mut stream: Vec<Vec<f64>> = tokens
        .iter()
        .enumerate()
        .map(|(position, &token)| {
            (0..WIDTH)
                .map(|i| wte[token * WIDTH + i] + wpe[position * WIDTH + i])
                .collect()
        })
        .collect();
    for layer in 0..LAYERS {
        let get = |part: &str| get(tensors, &format!("transformer.h.{layer}.{part}"));
        let projected: Vec<Vec<f64>> = stream
            .iter()
            .map(|x| {
                let normalized = norm(x, &get("ln_1.weight"), &get("ln_1.bias"));
                linear(&normalized, &get("attn.c_attn.weight"), &get("attn.c_attn.bias"))
            })
            .collect();
        for (t, x) in stream.iter_mut().enumerate() {
            let attended = linear(
                &attend(&projected, t),
                &get("attn.c_proj.weight"),
                &get("attn.c_proj.bias"),
            );
            for (x, a) in x.iter_mut().zip(attended) {
                *x += a;
            }
        }
        for x in &mut stream {
            let normalized = norm(x, &get("ln_2.weight"), &get("ln_2.bias"));
            let expanded = linear(&normalized, &get("mlp.c_fc.weight"), &get("mlp.c_fc.bias"));
            let activated: Vec<f64> = expanded.into_iter().map(gelu).collect();
            let contracted = linear(&activated, &get("mlp.c_proj.weight"), &get("mlp.c_proj.bias"));
            for (x, c) in x.iter_mut().zip(contracted) {
                *x += c;
            }
        }
    }
    let (gain, bias) = (
        get(tensors, "transformer.ln_f.weight"),
        get(tensors, "transformer.ln_f.bias"),
    );
    stream
        .iter()
        .map(|x| {
            let hidden = norm(x, &gain, &bias);
            (0..VOCABULARY)
                .map(|v| hidden.iter().zip(&wte[v * WIDTH..][..WIDTH]).map(|(h, e)| h * e).sum())
                .collect()
        })
        .collect()
}

/// Whether logits of single precision are those of double precision: within
/// 1e-4 of the logits' largest magnitude. Single precision through these two
/// blocks of width 16 comes within 2.7e-7 of it (measured on x86-64-v2, in the
/// dev and the release profile), and other orders of its sums would stay near
/// that; a mistake in the computation moves logits by their own size.
fn agree(logits: &[f32], exact: &[f64]) -> bool {
    let scale = exact.iter().map(|value| value.abs()).fold(0.0, f64::max);
    logits
        .iter()
        .zip(exact)
        .all(|(&logit, exact)| (f64::from(logit) - exact).abs() <= 1e-4 * scale)
}

#[test]
fn the_network_is_gpt2_in_every_way_it_is_fed() {
    let mut random = Random(21);
    let tensors = tensors(&mut random);
    let bytes = file(&tensors);
    let file = SafeTensors::parse(&bytes).expect("a file");
    let model = Gpt2::from_safetensors(&file, config(), Precision::Single).expect("a model");
    let tokens = [3usize, 7, 1, 10, 0, 7, 4, 9];
    let exact = reference(&tensors, &tokens);
    let ids = tokens.map(|token| u32::try_from(token).expect("small"));
    let mut logits = vec![0.0f32; VOCABULARY];

    let mut session = Session::new(&model, POSITIONS);
    session.feed(&ids, &mut logits).expect("fed");
    assert!(agree(&logits, &exact[tokens.len() - 1]), "all at once: {logits:?}");
    assert_eq!(session.len(), tokens.len());

    let mut session = Session::new(&model, tokens.len());
    for (position, &id) in ids.iter().enumerate() {
        session.feed(&[id], &mut logits).expect("fed");
        assert!(
            agree(&logits, &exact[position]),
            "one at a time, at {position}: {logits:?}"
        );
    }
    assert!(session.feed(&[0], &mut logits).is_err(), "a position past the room");

    session.truncate(3);
    session.feed(&ids[3..], &mut logits).expect("fed");
    assert!(
        agree(&logits, &exact[tokens.len() - 1]),
        "after the cache was cut back: {logits:?}"
    );
    assert!(session.feed(&[], &mut logits).is_err());
    let mut session = Session::new(&model, POSITIONS);
    assert!(
        session.feed(&[11], &mut logits).is_err(),
        "a token out of the vocabulary"
    );
}

/// The largest error of logits against those of double precision, as a share
/// of the logits' largest magnitude.
fn error_of(logits: &[f32], exact: &[f64]) -> f64 {
    let scale = exact.iter().map(|value| value.abs()).fold(0.0, f64::max);
    logits
        .iter()
        .zip(exact)
        .map(|(&logit, exact)| (f64::from(logit) - exact).abs() / scale)
        .fold(0.0, f64::max)
}

#[test]
fn every_positions_logits_are_those_of_one_at_a_time() {
    let mut random = Random(24);
    let tensors = tensors(&mut random);
    let bytes = file(&tensors);
    let file = SafeTensors::parse(&bytes).expect("a file");
    let tokens = [5usize, 2, 9, 9, 0, 3, 8];
    let exact = reference(&tensors, &tokens);
    let ids = tokens.map(|token| u32::try_from(token).expect("small"));
    let rotated = Scheme {
        rotation: Rotation::Hadamard(16),
        ..Scheme::INT8
    };
    for precision in [
        Precision::Single,
        Precision::Integers(Scheme::INT8),
        Precision::Integers(rotated),
    ] {
        let model = Gpt2::from_safetensors(&file, config(), precision).expect("a model");
        assert_eq!(model.precision(), precision);
        let mut all = vec![0.0f32; tokens.len() * VOCABULARY];
        let mut session = Session::new(&model, POSITIONS);
        session.feed_all(&ids[..2], &mut all[..2 * VOCABULARY]).expect("fed");
        session.feed_all(&ids[2..], &mut all[2 * VOCABULARY..]).expect("fed");
        let mut one = vec![0.0f32; VOCABULARY];
        let mut session = Session::new(&model, POSITIONS);
        for (position, &id) in ids.iter().enumerate() {
            session.feed(&[id], &mut one).expect("fed");
            let row = &all[position * VOCABULARY..][..VOCABULARY];
            // A row is quantized alone either way, and its sums are taken in the
            // same order: the two agree to the roundings of attention's sums.
            let apart = error_of(row, &one.iter().copied().map(f64::from).collect::<Vec<_>>());
            assert!(apart <= 1e-5, "{precision:?} at {position}: {apart}");
            // Integers through these two blocks of width 16 come within 1.4e-2 of
            // the logits' largest magnitude, rotated or not (measured on
            // x86-64-v2, in the dev and the release profile; single precision
            // within 2.3e-7): a mistake in the products, as a weight in another
            // column or a rotation of one side only, moves logits by their own
            // size.
            let error = error_of(row, &exact[position]);
            let bound = if precision == Precision::Single { 1e-4 } else { 5e-2 };
            assert!(error <= bound, "{precision:?} at {position}: {error}");
        }
    }
}

#[test]
fn other_networks_are_refused() {
    let refused = |text: &str| {
        let value = json::parse(text.as_bytes()).expect("JSON");
        Config::from_json(value.as_object().expect("an object")).is_err()
    };
    let base =
        r#""n_layer": 2, "n_head": 2, "n_embd": 16, "n_positions": 32, "vocab_size": 11, "layer_norm_epsilon": 1e-05"#;
    assert!(!refused(&format!(
        r#"{{"model_type": "gpt2", "activation_function": "gelu_new", {base}}}"#
    )));
    assert!(refused(&format!(
        r#"{{"model_type": "llama", "activation_function": "gelu_new", {base}}}"#
    )));
    assert!(refused(&format!(
        r#"{{"model_type": "gpt2", "activation_function": "relu", {base}}}"#
    )));
    assert!(refused(&format!(
        r#"{{"model_type": "gpt2", "activation_function": "gelu_new", "scale_attn_by_inverse_layer_idx": true, {base}}}"#
    )));
    assert!(refused(&format!(
        r#"{{"model_type": "gpt2", "activation_function": "gelu_new", "n_inner": 100, {base}}}"#
    )));
    assert!(refused(
        r#"{"model_type": "gpt2", "activation_function": "gelu_new", "n_layer": 2, "n_head": 3, "n_embd": 16,
            "n_positions": 32, "vocab_size": 11, "layer_norm_epsilon": 1e-05}"#
    ));
}

#[test]
fn a_missing_or_misshapen_tensor_is_refused() {
    let mut random = Random(22);
    let mut tensors = tensors(&mut random);
    tensors.retain(|(name, _, _)| name != "transformer.h.1.mlp.c_fc.bias");
    let bytes = file(&tensors);
    let error = Gpt2::from_safetensors(
        &SafeTensors::parse(&bytes).expect("a file"),
        config(),
        Precision::Integers(Scheme::INT8),
    )
    .expect_err("refused");
    assert!(error.to_string().contains("transformer.h.1.mlp.c_fc.bias"), "{error}");
    let mut tensors = self::tensors(&mut random);
    tensors[0].1 = vec![WIDTH, VOCABULARY];
    let bytes = file(&tensors);
    assert!(
        Gpt2::from_safetensors(
            &SafeTensors::parse(&bytes).expect("a file"),
            config(),
            Precision::Single
        )
        .is_err()
    );
}

#[test]
fn a_configuration_is_its_bytes() {
    let config = config();
    let bytes = config.to_bytes().expect("its bytes");
    assert_eq!(Config::from_bytes(&bytes).expect("a configuration"), config);
    assert!(Config::from_bytes(&bytes[..CONFIG_BYTES - 1]).is_err());
    let refused = |at: usize, value: [u8; 4]| {
        let mut changed = bytes;
        changed[at..at + 4].copy_from_slice(&value);
        Config::from_bytes(&changed).is_err()
    };
    assert!(refused(0, 0u32.to_le_bytes()), "no blocks");
    assert!(refused(4, 3u32.to_le_bytes()), "three heads of a width of 16");
    assert!(refused(20, 0f32.to_le_bytes()), "an epsilon of 0");
    assert!(refused(20, f32::NAN.to_le_bytes()), "an epsilon that is no number");
}

/// The tensors, each value cut to bfloat16, as zenz-v3.1's are.
fn in_bfloat16(tensors: &Tensors) -> Vec<(String, Vec<usize>, Vec<f32>)> {
    tensors
        .iter()
        .map(|(name, shape, values)| {
            let cut = values
                .iter()
                .map(|value| f32::from_bits(value.to_bits() & 0xffff_0000))
                .collect();
            (name.clone(), shape.clone(), cut)
        })
        .collect()
}

/// A model file of the model's tensors, of zenz's architecture and tokenizer,
/// with a vocabulary of two special words and a byte for each other word.
fn model_file(architecture: &str, config: &Config, parts: &[FileTensor<'_>]) -> Vec<u8> {
    let tensors: Vec<Tensor<'_>> = parts.iter().map(FileTensor::tensor).collect();
    let words: Vec<Vec<u8>> = (0..VOCABULARY)
        .map(|word| {
            if word < 2 {
                Vec::new()
            } else {
                vec![u8::try_from(word).expect("a byte")]
            }
        })
        .collect();
    let words: Vec<&[u8]> = words.iter().map(Vec::as_slice).collect();
    file::write(&Contents {
        architecture,
        tokenizer: "byte-bpe",
        metadata: Metadata {
            name: "test",
            license: "GPL-3.0-or-later",
            source: "random weights",
            notice: "",
        },
        configuration: &config.to_bytes().expect("its bytes"),
        words: &words,
        merges: &[],
        specials: Specials {
            unknown: 0,
            end: 1,
            begin: None,
            padding: None,
        },
        tensors: &tensors,
    })
    .expect("a model file")
}

#[test]
fn a_model_file_computes_as_the_model_it_was_written_from() {
    let mut random = Random(25);
    let random_tensors = tensors(&mut random);
    let ids = [4u32, 10, 0, 6, 6, 2, 9];
    let schemes = [
        Scheme::INT8,
        Scheme {
            weight_bits: 4,
            rotation: Rotation::Hadamard(16),
            fitted: true,
        },
    ];
    // Random values of 16 significant bits are written in single precision;
    // cut to bfloat16, in bfloat16.
    for (tensors, floats) in [
        (random_tensors.clone(), Kind::F32),
        (in_bfloat16(&random_tensors), Kind::Bf16),
    ] {
        let bytes = file(&tensors);
        let safetensors = SafeTensors::parse(&bytes).expect("a file");
        for scheme in schemes {
            let model = Gpt2::from_safetensors(&safetensors, config(), Precision::Integers(scheme)).expect("a model");
            let parts = model.to_tensors().expect("its tensors");
            let panels = parts.iter().filter(|part| matches!(part.kind, Kind::Panels(_))).count();
            assert_eq!(panels, 4 * LAYERS + 1);
            assert!(
                parts
                    .iter()
                    .all(|part| matches!(part.kind, Kind::Panels(_)) || part.kind == floats)
            );
            let written = model_file("gpt2-zenz", model.config(), &parts);
            let parsed = ModelFile::parse(&written).expect("a model file");
            assert!(parsed.verify());
            let read = Gpt2::from_file(&parsed).expect("a model");
            assert_eq!(read.config(), model.config());
            assert_eq!(read.precision(), Precision::Integers(scheme));
            assert_eq!(read.to_tensors().expect("its tensors"), parts);
            let mut expected = vec![0.0f32; ids.len() * VOCABULARY];
            let mut logits = vec![f32::NAN; ids.len() * VOCABULARY];
            Session::new(&model, POSITIONS)
                .feed_all(&ids, &mut expected)
                .expect("fed");
            Session::new(&read, POSITIONS).feed_all(&ids, &mut logits).expect("fed");
            assert_eq!(logits, expected, "{scheme:?}, floats in {floats:?}");
            let zenz = Zenz::from_file(&parsed).expect("zenz");
            assert_eq!(zenz.tokenizer.len(), VOCABULARY);
        }
    }
    let bytes = file(&random_tensors);
    let single = Gpt2::from_safetensors(
        &SafeTensors::parse(&bytes).expect("a file"),
        config(),
        Precision::Single,
    )
    .expect("a model");
    assert!(single.to_tensors().is_err(), "a model in single precision written");
}

#[test]
fn a_model_file_of_another_model_is_refused() {
    let mut random = Random(26);
    let tensors = tensors(&mut random);
    let bytes = file(&tensors);
    let safetensors = SafeTensors::parse(&bytes).expect("a file");
    let model = Gpt2::from_safetensors(&safetensors, config(), Precision::Integers(Scheme::INT8)).expect("a model");
    let parts = model.to_tensors().expect("its tensors");
    let refused = |parts: &[FileTensor<'_>]| {
        let written = model_file("gpt2-zenz", model.config(), parts);
        let error = Gpt2::from_file(&ModelFile::parse(&written).expect("a model file")).expect_err("refused");
        error.to_string()
    };
    let without_head: Vec<FileTensor<'_>> = parts
        .iter()
        .filter(|part| part.name != "lm_head.weight")
        .cloned()
        .collect();
    assert!(refused(&without_head).contains("lm_head.weight"));
    let first = "transformer.h.0.attn.c_attn.weight";
    let replaced = |kind: Kind, bytes: Vec<u8>| -> Vec<FileTensor<'_>> {
        parts
            .iter()
            .map(|part| {
                if part.name == first {
                    FileTensor {
                        kind,
                        bytes: bytes.clone().into(),
                        ..part.clone()
                    }
                } else {
                    part.clone()
                }
            })
            .collect()
    };
    let floats = replaced(Kind::F32, vec![0; 4 * WIDTH * 3 * WIDTH]);
    assert!(refused(&floats).contains("not weights in panels"));
    let wide = Panels {
        columns: 32,
        group: 64,
        bits: 8,
        fitted: false,
        rotation: 0,
    };
    let length = wide.length(WIDTH, 3 * WIDTH).expect("a length");
    let other_layout = replaced(Kind::Panels(wide), vec![0; length]);
    assert!(refused(&other_layout).contains("another layout"));
    let written = model_file("gpt2-other", model.config(), &parts);
    let parsed = ModelFile::parse(&written).expect("a model file");
    assert!(Zenz::from_file(&parsed).is_err(), "another architecture");
}

#[test]
fn the_likeliest_tokens_from_a_row_are_those_of_a_feed_of_all_and_a_feeding_stops_where_asked() {
    let mut random = Random(28);
    let tensors = tensors(&mut random);
    let bytes = file(&tensors);
    let file = SafeTensors::parse(&bytes).expect("a file");
    let ids = [3u32, 7, 1, 10, 0, 7];
    for precision in [Precision::Single, Precision::Integers(Scheme::INT8)] {
        let model = Gpt2::from_safetensors(&file, config(), precision).expect("a model");
        let mut all = vec![0.0f32; ids.len() * VOCABULARY];
        Session::new(&model, POSITIONS).feed_all(&ids, &mut all).expect("fed");
        let mut session = Session::new(&model, POSITIONS);
        let mut likeliest = vec![None; 2];
        let mut blocks = 0;
        let mut stop_at_second = || {
            blocks += 1;
            blocks > 1
        };
        assert!(
            !session
                .feed_likeliest(&ids, 4, &mut likeliest, &mut stop_at_second)
                .expect("fed"),
            "stopped"
        );
        assert!(session.is_empty(), "nothing kept of a feeding stopped");
        assert_eq!(likeliest, [None, None], "nothing written by a feeding stopped");
        assert!(
            session
                .feed_likeliest(&ids, 4, &mut likeliest, &mut || false)
                .expect("fed")
        );
        assert_eq!(session.len(), ids.len());
        // The same rows of the same computation, the output layer left out for
        // the others and its argmax taken as it is summed.
        let expected: Vec<Option<usize>> = all[4 * VOCABULARY..]
            .as_chunks::<VOCABULARY>()
            .0
            .iter()
            .map(|row| argmax(row))
            .collect();
        assert_eq!(likeliest, expected, "{precision:?}");
    }
}
