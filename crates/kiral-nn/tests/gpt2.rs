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
//! positions through every way the session is used.

use kiral_kernels::quantized::{Rotation, Scheme};
use kiral_model::json;
use kiral_model::safetensors::SafeTensors;
use kiral_nn::gpt2::{Config, Gpt2, Precision, Session};

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
