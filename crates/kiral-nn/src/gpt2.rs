// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later
//
// This file is part of KIRAL. The SPDX lines name its copyright holder and its
// license, in the machine-readable form of the REUSE specification: the GNU
// General Public License, version 3 or any later version
// (LICENSES/GPL-3.0-or-later.txt).

//! GPT-2 in single precision, as Hugging Face's `GPT2LMHeadModel` computes it:
//! zenz's network, and the reference that its quantized forms are measured
//! against.
//!
//! Each block normalizes the residual stream, attends with the heads over the
//! cache of keys and values -- each position to itself and those before it --
//! adds the projection of what it found to the stream, normalizes again, and
//! adds the projection of an expansion through GELU. The logits are the
//! normalized stream's dot products with the token embedding, which the output
//! layer shares. The linear layers' weights are packed for the kernel that
//! computes them when they are read.

use kiral_kernels::activation::gelu_tanh;
use kiral_kernels::attention::attention;
use kiral_kernels::linear::{Packed, linear};
use kiral_kernels::norm::layer_norm;
use kiral_kernels::vector::{add, logits};
use kiral_model::json::{Object, Value};
use kiral_model::safetensors::SafeTensors;

use crate::Error;

/// The sizes of a GPT-2, as its `config.json` gives them.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Config {
    /// The blocks (`n_layer`).
    pub layers: usize,
    /// The heads of each attention (`n_head`).
    pub heads: usize,
    /// The width of the residual stream (`n_embd`).
    pub width: usize,
    /// The positions it has embeddings for (`n_positions`).
    pub positions: usize,
    /// The words of its vocabulary (`vocab_size`).
    pub vocabulary: usize,
    /// The epsilon of its layer normalizations (`layer_norm_epsilon`).
    pub epsilon: f32,
}

impl Config {
    /// The sizes of a `config.json`, checked to be those of the GPT-2 this
    /// computes: `gelu_new`, attention scaled by the square root of a head's
    /// width and by nothing else, an expansion of four times the width.
    ///
    /// # Errors
    ///
    /// Where a size is missing or out of range, or the model is another.
    pub fn from_json(config: &Object) -> Result<Self, Error> {
        let size = |name: &str| {
            config
                .get(name)
                .and_then(Value::as_u64)
                .and_then(|value| usize::try_from(value).ok())
                .filter(|&value| value > 0)
                .ok_or_else(|| Error::Config(format!("no {name} that is a positive integer")))
        };
        let text = |name: &str| config.get(name).and_then(Value::as_str);
        let flag = |name: &str| config.get(name).and_then(Value::as_bool);
        if text("model_type") != Some("gpt2") {
            return Err(Error::Config("a model_type other than gpt2".to_owned()));
        }
        if text("activation_function") != Some("gelu_new") {
            return Err(Error::Config("an activation_function other than gelu_new".to_owned()));
        }
        if flag("scale_attn_weights") == Some(false)
            || flag("scale_attn_by_inverse_layer_idx") == Some(true)
            || flag("reorder_and_upcast_attn") == Some(true)
        {
            return Err(Error::Config(
                "attention scaled otherwise than by a head's width".to_owned(),
            ));
        }
        let width = size("n_embd")?;
        match config.get("n_inner") {
            None | Some(Value::Null) => {}
            Some(inner) if inner.as_u64().and_then(|inner| usize::try_from(inner).ok()) == Some(4 * width) => {}
            Some(_) => return Err(Error::Config("an expansion other than four times the width".to_owned())),
        }
        let heads = size("n_head")?;
        if width % heads != 0 {
            return Err(Error::Config(format!(
                "{heads} heads that do not divide the width {width}"
            )));
        }
        let epsilon = config
            .get("layer_norm_epsilon")
            .and_then(Value::as_f64)
            .filter(|epsilon| *epsilon > 0.0 && *epsilon < 1.0)
            .ok_or_else(|| Error::Config("no layer_norm_epsilon between 0 and 1".to_owned()))?;
        Ok(Self {
            layers: size("n_layer")?,
            heads,
            width,
            positions: size("n_positions")?,
            vocabulary: size("vocab_size")?,
            epsilon: kiral_model::half::f64_to_f32(epsilon),
        })
    }

    /// The width of a head.
    #[must_use]
    pub fn head_width(&self) -> usize {
        self.width / self.heads
    }
}

/// A layer normalization's gain and bias.
#[derive(Debug, Clone)]
struct Norm {
    gain: Vec<f32>,
    bias: Vec<f32>,
}

impl Norm {
    fn apply(&self, input: &[f32], epsilon: f32, output: &mut [f32]) {
        layer_norm(input, &self.gain, &self.bias, epsilon, output);
    }
}

/// A linear layer: its weights, packed, and its bias.
#[derive(Debug, Clone)]
struct Linear {
    weight: Packed,
    bias: Vec<f32>,
}

impl Linear {
    fn apply(&self, input: &[f32], output: &mut [f32]) {
        linear(input, &self.weight, &self.bias, output);
    }
}

/// The room the blocks compute in, for the rows fed at once.
#[derive(Debug)]
struct Room {
    normalized: Vec<f32>,
    projected: Vec<f32>,
    found: Vec<f32>,
    expanded: Vec<f32>,
    scores: Vec<f32>,
}

impl Room {
    fn new(config: &Config, rows: usize, positions: usize) -> Self {
        let width = config.width;
        Self {
            normalized: vec![0.0; rows * width],
            projected: vec![0.0; rows * 3 * width],
            found: vec![0.0; rows * width],
            expanded: vec![0.0; rows * 4 * width],
            scores: vec![0.0; positions],
        }
    }
}

/// A block of the decoder.
#[derive(Debug, Clone)]
struct Block {
    attention_norm: Norm,
    /// The width to the queries, keys and values of every head.
    attention: Linear,
    /// What the heads found, back to the width.
    projection: Linear,
    expansion_norm: Norm,
    /// The width to four times the width.
    expansion: Linear,
    /// Four times the width back to the width.
    contraction: Linear,
}

/// A GPT-2's weights, in single precision.
#[derive(Debug, Clone)]
pub struct Gpt2 {
    config: Config,
    token_embedding: Vec<f32>,
    position_embedding: Vec<f32>,
    blocks: Vec<Block>,
    final_norm: Norm,
}

impl Gpt2 {
    /// The model of a safetensors file of Hugging Face's names, its weights
    /// converted to single precision. An output layer of its own (`lm_head`) is
    /// taken only where it is the token embedding.
    ///
    /// # Errors
    ///
    /// Where a tensor is missing, of another shape than the configuration's, or
    /// not of floats.
    pub fn from_safetensors(file: &SafeTensors<'_>, config: Config) -> Result<Self, Error> {
        let tensor = |name: &str, shape: &[usize]| -> Result<Vec<f32>, Error> {
            let tensor = file
                .tensor(name)
                .ok_or_else(|| Error::Tensor(format!("no tensor {name}")))?;
            if tensor.shape != shape {
                return Err(Error::Tensor(format!(
                    "{name} of shape {:?}, not {shape:?}",
                    tensor.shape
                )));
            }
            tensor
                .to_f32()
                .map_err(|error| Error::Tensor(format!("{name}: {error}")))
        };
        let width = config.width;
        let norm = |name: &str| -> Result<Norm, Error> {
            Ok(Norm {
                gain: tensor(&format!("{name}.weight"), &[width])?,
                bias: tensor(&format!("{name}.bias"), &[width])?,
            })
        };
        let linear = |name: &str, inputs: usize, outputs: usize| -> Result<Linear, Error> {
            Ok(Linear {
                weight: Packed::new(&tensor(&format!("{name}.weight"), &[inputs, outputs])?, inputs, outputs),
                bias: tensor(&format!("{name}.bias"), &[outputs])?,
            })
        };
        let blocks = (0..config.layers)
            .map(|layer| {
                let name = |part: &str| format!("transformer.h.{layer}.{part}");
                Ok(Block {
                    attention_norm: norm(&name("ln_1"))?,
                    attention: linear(&name("attn.c_attn"), width, 3 * width)?,
                    projection: linear(&name("attn.c_proj"), width, width)?,
                    expansion_norm: norm(&name("ln_2"))?,
                    expansion: linear(&name("mlp.c_fc"), width, 4 * width)?,
                    contraction: linear(&name("mlp.c_proj"), 4 * width, width)?,
                })
            })
            .collect::<Result<Vec<_>, Error>>()?;
        let token_embedding = tensor("transformer.wte.weight", &[config.vocabulary, width])?;
        if file.tensor("lm_head.weight").is_some()
            && tensor("lm_head.weight", &[config.vocabulary, width])? != token_embedding
        {
            return Err(Error::Tensor(
                "an output layer other than the token embedding".to_owned(),
            ));
        }
        Ok(Self {
            config,
            token_embedding,
            position_embedding: tensor("transformer.wpe.weight", &[config.positions, width])?,
            blocks,
            final_norm: norm("transformer.ln_f")?,
        })
    }

    /// Its sizes.
    #[must_use]
    pub fn config(&self) -> &Config {
        &self.config
    }
}

/// A sequence that a model computes, a token after another: the cache of the
/// keys and values of its positions, and the room to compute in.
#[derive(Debug)]
pub struct Session<'m> {
    model: &'m Gpt2,
    /// The positions whose keys and values are in the cache.
    length: usize,
    /// The positions the cache has room for.
    capacity: usize,
    /// The keys of each block and head, a position's after another.
    keys: Vec<f32>,
    /// The values, laid out as the keys.
    values: Vec<f32>,
}

impl<'m> Session<'m> {
    /// A session of the model with room for `capacity` positions, at most the
    /// model's.
    ///
    /// # Panics
    ///
    /// Where `capacity` is more than the positions of the model.
    #[must_use]
    pub fn new(model: &'m Gpt2, capacity: usize) -> Self {
        let config = model.config;
        assert!(
            capacity <= config.positions,
            "room for {capacity} positions of {}",
            config.positions
        );
        let cache = config.layers * capacity * config.width;
        Self {
            model,
            length: 0,
            capacity,
            keys: vec![0.0; cache],
            values: vec![0.0; cache],
        }
    }

    /// The positions computed.
    #[must_use]
    pub fn len(&self) -> usize {
        self.length
    }

    /// Whether no position has been computed.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.length == 0
    }

    /// Forgets the positions from `length` on, to compute others after the
    /// ones kept.
    pub fn truncate(&mut self, length: usize) {
        self.length = self.length.min(length);
    }

    /// The index in the cache of a block's head's row of a position.
    fn row(&self, layer: usize, head: usize, position: usize) -> usize {
        let config = self.model.config;
        ((layer * config.heads + head) * self.capacity + position) * config.head_width()
    }

    /// Computes the tokens at the next positions, and writes the logits of the
    /// last of them.
    ///
    /// # Errors
    ///
    /// Where there are no tokens, a token is out of the vocabulary, or the
    /// positions would pass the room of the session.
    ///
    /// # Panics
    ///
    /// Where `output` is not a logit for each word of the vocabulary.
    pub fn feed(&mut self, tokens: &[u32], output: &mut [f32]) -> Result<(), Error> {
        let model = self.model;
        let config = model.config;
        let rows = tokens.len();
        if rows == 0 {
            return Err(Error::Input("no tokens".to_owned()));
        }
        if self.length + rows > self.capacity {
            return Err(Error::Input(format!(
                "{} positions, past the room for {}",
                self.length + rows,
                self.capacity
            )));
        }
        assert_eq!(
            output.len(),
            config.vocabulary,
            "logits of another length than the vocabulary"
        );
        let mut stream = self.embed(tokens)?;
        let mut room = Room::new(&config, rows, self.length + rows);
        for (layer, block) in model.blocks.iter().enumerate() {
            self.block(layer, block, &mut stream, &mut room);
        }
        self.length += rows;
        let mut hidden = vec![0.0f32; config.width];
        model
            .final_norm
            .apply(&stream[(rows - 1) * config.width..], config.epsilon, &mut hidden);
        logits(&hidden, &model.token_embedding, output);
        Ok(())
    }

    /// The residual stream of the tokens at the next positions: their tokens'
    /// and their positions' embeddings.
    fn embed(&self, tokens: &[u32]) -> Result<Vec<f32>, Error> {
        let model = self.model;
        let config = model.config;
        let width = config.width;
        let mut stream = vec![0.0f32; tokens.len() * width];
        for (row, (&token, embedded)) in tokens.iter().zip(stream.chunks_exact_mut(width)).enumerate() {
            let token = usize::try_from(token)
                .ok()
                .filter(|&token| token < config.vocabulary)
                .ok_or_else(|| Error::Input(format!("the token {token}, out of the vocabulary")))?;
            embedded.copy_from_slice(&model.token_embedding[token * width..][..width]);
            add(
                embedded,
                &model.position_embedding[(self.length + row) * width..][..width],
            );
        }
        Ok(stream)
    }

    /// A block of the decoder on the stream of the rows fed, whose keys and
    /// values it caches.
    fn block(&mut self, layer: usize, block: &Block, stream: &mut [f32], room: &mut Room) {
        let epsilon = self.model.config.epsilon;
        block.attention_norm.apply(stream, epsilon, &mut room.normalized);
        block.attention.apply(&room.normalized, &mut room.projected);
        self.cache(layer, &room.projected);
        self.attend(layer, &room.projected, &mut room.scores, &mut room.found);
        block.projection.apply(&room.found, &mut room.normalized);
        add(stream, &room.normalized);
        block.expansion_norm.apply(stream, epsilon, &mut room.normalized);
        block.expansion.apply(&room.normalized, &mut room.expanded);
        gelu_tanh(&mut room.expanded);
        block.contraction.apply(&room.expanded, &mut room.normalized);
        add(stream, &room.normalized);
    }

    /// Puts the keys and values of the rows' queries, keys and values in the
    /// cache, at the rows' positions.
    fn cache(&mut self, layer: usize, projected: &[f32]) {
        let config = self.model.config;
        let (width, head_width) = (config.width, config.head_width());
        for (row, parts) in projected.chunks_exact(3 * width).enumerate() {
            for head in 0..config.heads {
                let at = self.row(layer, head, self.length + row);
                let part = head * head_width;
                self.keys[at..][..head_width].copy_from_slice(&parts[width + part..][..head_width]);
                self.values[at..][..head_width].copy_from_slice(&parts[2 * width + part..][..head_width]);
            }
        }
    }

    /// The attention of each row's queries, head by head, over the positions up
    /// to the row's own.
    fn attend(&self, layer: usize, projected: &[f32], scores: &mut [f32], found: &mut [f32]) {
        let config = self.model.config;
        let (width, head_width) = (config.width, config.head_width());
        let scale =
            1.0 / kiral_model::half::f64_to_f32(f64::from(u32::try_from(head_width).expect("a head's width")).sqrt());
        for (row, (parts, found)) in projected
            .chunks_exact(3 * width)
            .zip(found.chunks_exact_mut(width))
            .enumerate()
        {
            let positions = self.length + row + 1;
            for head in 0..config.heads {
                let first = self.row(layer, head, 0);
                let cached = positions * head_width;
                attention(
                    &parts[head * head_width..][..head_width],
                    &self.keys[first..][..cached],
                    &self.values[first..][..cached],
                    scale,
                    &mut scores[..positions],
                    &mut found[head * head_width..][..head_width],
                );
            }
        }
    }
}
