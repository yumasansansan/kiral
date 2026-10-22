// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later
//
// This file is part of KIRAL. The SPDX lines name its copyright holder and its
// license, in the machine-readable form of the REUSE specification: the GNU
// General Public License, version 3 or any later version
// (LICENSES/GPL-3.0-or-later.txt).

//! GPT-2, as Hugging Face's `GPT2LMHeadModel` computes it: zenz's network, in
//! single precision, the reference, and with its products in integers.
//!
//! Each block normalizes the residual stream, attends with the heads over the
//! cache of keys and values -- each position to itself and those before it --
//! adds the projection of what it found to the stream, normalizes again, and
//! adds the projection of an expansion through GELU. The logits are the
//! normalized stream's dot products with the token embedding, which the output
//! layer shares. The linear layers' weights are packed for the kernel that
//! computes them when they are read.
//!
//! In integers ([`Precision::Integers`]), the linear layers and the output
//! layer take their weights quantized to bytes, a scale for each output's group
//! of 64 inputs, and their inputs quantized as they come, a scale and a zero
//! point for each row's group, by a scheme that may rotate them first
//! ([`kiral_kernels::quantized`]); the embeddings, the normalizations, GELU,
//! attention and its cache, and the residual stream stay in single precision.
//!
//! A model is read from a safetensors file of Hugging Face's names
//! ([`Gpt2::from_safetensors`]), its weights converted, and packed or
//! quantized, as they are read; or from a model file ([`Gpt2::from_file`]),
//! which holds a model in integers with its weights quantized and packed
//! already: the linear layers' and the output layer's weights and the
//! embeddings are read where they lie, an embedding's row converted when it is
//! looked up, and only the normalizations' and the biases' floats, 0.5 MB of
//! zenz's 100, are copied. [`Gpt2::to_tensors`] gives a model in integers'
//! tensors for its file.

use std::borrow::Cow;

use kiral_kernels::activation::gelu_tanh;
use kiral_kernels::attention::attention;
use kiral_kernels::linear::{Packed, linear};
use kiral_kernels::norm::layer_norm;
use kiral_kernels::quantized::{self, Rotation, Rows, Scheme};
use kiral_kernels::vector::{add, logits};
use kiral_model::file::{Floats, Kind, ModelFile, Panels, Tensor};
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

    /// The configuration of a model file, as [`Config::to_bytes`] lays it out,
    /// its sizes checked as [`Config::from_json`] checks them.
    ///
    /// # Errors
    ///
    /// Where the bytes are not [`CONFIG_BYTES`], a size is 0, the heads do not
    /// divide the width, or the epsilon is not between 0 and 1.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, Error> {
        let numbers: &[u8; CONFIG_BYTES] = bytes
            .try_into()
            .map_err(|_| Error::Config(format!("{} bytes of configuration, not {CONFIG_BYTES}", bytes.len())))?;
        let numbers = numbers.as_chunks::<4>().0;
        let names = ["blocks", "heads", "width", "positions", "vocabulary"];
        let size = |at: usize| {
            usize::try_from(u32::from_le_bytes(numbers[at]))
                .ok()
                .filter(|&size| size > 0)
                .ok_or_else(|| Error::Config(format!("no {} that is a positive integer", names[at])))
        };
        let config = Self {
            layers: size(0)?,
            heads: size(1)?,
            width: size(2)?,
            positions: size(3)?,
            vocabulary: size(4)?,
            epsilon: f32::from_le_bytes(numbers[5]),
        };
        if !config.width.is_multiple_of(config.heads) {
            return Err(Error::Config(format!(
                "{} heads that do not divide the width {}",
                config.heads, config.width
            )));
        }
        if !(config.epsilon > 0.0 && config.epsilon < 1.0) {
            return Err(Error::Config("no epsilon between 0 and 1".to_owned()));
        }
        Ok(config)
    }

    /// The configuration's bytes in a model file: the blocks, the heads, the
    /// width, the positions and the vocabulary, four bytes each, and the
    /// epsilon in single precision.
    ///
    /// # Errors
    ///
    /// Where a size passes 32 bits.
    pub fn to_bytes(&self) -> Result<[u8; CONFIG_BYTES], Error> {
        let mut bytes = [0; CONFIG_BYTES];
        let sizes = [self.layers, self.heads, self.width, self.positions, self.vocabulary];
        for (number, size) in bytes.as_chunks_mut::<4>().0.iter_mut().zip(sizes) {
            *number = u32::try_from(size)
                .map_err(|_| Error::Config(format!("the size {size}, past 32 bits")))?
                .to_le_bytes();
        }
        bytes[20..].copy_from_slice(&self.epsilon.to_le_bytes());
        Ok(bytes)
    }

    /// The width of a head.
    #[must_use]
    pub fn head_width(&self) -> usize {
        self.width / self.heads
    }
}

/// The bytes of a configuration in a model file.
pub const CONFIG_BYTES: usize = 24;

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

/// What the products of a model's linear layers and output layer are computed
/// in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Precision {
    /// Single precision: the reference.
    Single,
    /// Integers: the weights quantized to signed bytes, the inputs to unsigned
    /// bytes with a zero point, each in groups of 64 with a scale of their own,
    /// by a scheme.
    Integers(Scheme),
}

/// A linear layer's weights, packed for the kernel of its precision.
#[derive(Debug, Clone)]
enum Weights<'a> {
    Single(Packed),
    Int8(quantized::Packed<'a>),
}

/// A linear layer: its weights and its bias.
#[derive(Debug, Clone)]
struct Linear<'a> {
    weight: Weights<'a>,
    bias: Vec<f32>,
}

impl Linear<'_> {
    /// `output = input weight + bias`, `rows` the room the input is quantized
    /// in, for weights in integers.
    fn apply(&self, input: &[f32], rows: &mut Rows, output: &mut [f32]) {
        match &self.weight {
            Weights::Single(weight) => linear(input, weight, &self.bias, output),
            Weights::Int8(weight) => {
                rows.quantize(input, weight.depth(), weight.scheme().rotation);
                quantized::linear(rows, weight, &self.bias, output);
            }
        }
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
    quantized: Rows,
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
            quantized: Rows::new(),
        }
    }
}

/// A block of the decoder.
#[derive(Debug, Clone)]
struct Block<'a> {
    attention_norm: Norm,
    /// The width to the queries, keys and values of every head.
    attention: Linear<'a>,
    /// What the heads found, back to the width.
    projection: Linear<'a>,
    expansion_norm: Norm,
    /// The width to four times the width.
    expansion: Linear<'a>,
    /// Four times the width back to the width.
    contraction: Linear<'a>,
}

/// An embedding: a row of floats for each token, or each position.
#[derive(Debug, Clone)]
enum Table<'a> {
    /// Converted to single precision when it was read.
    Single(Vec<f32>),
    /// Read where it lies in a model file, a row converted when it is looked
    /// up.
    File(Floats<'a>),
}

impl Table<'_> {
    /// Writes the row `index` of `output.len()` floats into `output`.
    fn row(&self, index: usize, output: &mut [f32]) {
        let width = output.len();
        match self {
            Self::Single(values) => output.copy_from_slice(&values[index * width..][..width]),
            Self::File(floats) => floats.read(index * width, output),
        }
    }

    /// Every float, in single precision.
    fn to_vec(&self) -> Vec<f32> {
        match self {
            Self::Single(values) => values.clone(),
            Self::File(floats) => floats.to_vec(),
        }
    }
}

/// A GPT-2's weights: its own, or read where they lie in a model file.
#[derive(Debug, Clone)]
pub struct Gpt2<'a> {
    config: Config,
    precision: Precision,
    token_embedding: Table<'a>,
    position_embedding: Table<'a>,
    blocks: Vec<Block<'a>>,
    final_norm: Norm,
    /// The output layer in integers, at eight bits, the token embedding's rows
    /// its columns; none in single precision, which takes the embedding as it
    /// is.
    head: Option<quantized::Packed<'a>>,
    /// The output layer's bias, which GPT-2 has not: zeros.
    head_bias: Vec<f32>,
}

/// The name of a block's tensor, Hugging Face's.
fn block_name(layer: usize, part: &str) -> String {
    format!("transformer.h.{layer}.{part}")
}

impl Gpt2<'static> {
    /// The model of a safetensors file of Hugging Face's names, its weights
    /// converted to single precision, and the linear layers' and the output
    /// layer's quantized for [`Precision::Integers`]. An output layer of its
    /// own (`lm_head`) is taken only where it is the token embedding.
    ///
    /// # Errors
    ///
    /// Where a tensor is missing, of another shape than the configuration's, or
    /// not of floats.
    pub fn from_safetensors(file: &SafeTensors<'_>, config: Config, precision: Precision) -> Result<Self, Error> {
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
            let weight = tensor(&format!("{name}.weight"), &[inputs, outputs])?;
            Ok(Linear {
                weight: match precision {
                    Precision::Single => Weights::Single(Packed::new(&weight, inputs, outputs)),
                    Precision::Integers(scheme) => {
                        Weights::Int8(quantized::Packed::quantize(inputs, outputs, scheme, |k, j| {
                            weight[k * outputs + j]
                        }))
                    }
                },
                bias: tensor(&format!("{name}.bias"), &[outputs])?,
            })
        };
        let blocks = (0..config.layers)
            .map(|layer| {
                let name = |part: &str| block_name(layer, part);
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
        // The output layer's weights stay at eight bits whatever the blocks'
        // are: its errors move the logits themselves. With the blocks' at four
        // bits, rotated in blocks of 256 and fitted, the conversions' positions
        // of AJIMEE-Bench whose likeliest token agrees with single precision's
        // went from 4372 to 4376 of 4396 with it at eight, for 2.3 MB more.
        let head = match precision {
            Precision::Single => None,
            Precision::Integers(scheme) => {
                let scheme = Scheme {
                    weight_bits: 8,
                    ..scheme
                };
                Some(quantized::Packed::quantize(width, config.vocabulary, scheme, |k, v| {
                    token_embedding[v * width + k]
                }))
            }
        };
        Ok(Self {
            config,
            precision,
            token_embedding: Table::Single(token_embedding),
            position_embedding: Table::Single(tensor("transformer.wpe.weight", &[config.positions, width])?),
            blocks,
            final_norm: norm("transformer.ln_f")?,
            head,
            head_bias: vec![0.0; config.vocabulary],
        })
    }
}

/// The scheme of weights in panels of the kernels' layout; none for another
/// layout.
fn scheme_of(panels: &Panels) -> Option<Scheme> {
    if usize::from(panels.columns) != quantized::PANEL || usize::from(panels.group) != quantized::GROUP {
        return None;
    }
    Some(Scheme {
        weight_bits: u32::from(panels.bits),
        rotation: match panels.rotation {
            0 => Rotation::None,
            order => Rotation::Hadamard(usize::try_from(order).ok()?),
        },
        fitted: panels.fitted,
    })
}

/// The layout and scheme of weights packed by the kernels, as a model file
/// gives them.
fn panels_of(scheme: Scheme) -> Result<Panels, Error> {
    let failed = || Error::Config(format!("a scheme a model file does not hold: {scheme:?}"));
    Ok(Panels {
        columns: u16::try_from(quantized::PANEL).map_err(|_| failed())?,
        group: u16::try_from(quantized::GROUP).map_err(|_| failed())?,
        bits: u8::try_from(scheme.weight_bits).map_err(|_| failed())?,
        fitted: scheme.fitted,
        rotation: match scheme.rotation {
            Rotation::None => 0,
            Rotation::Hadamard(order) => u32::try_from(order).map_err(|_| failed())?,
        },
    })
}

/// A tensor of a model as a model file holds it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FileTensor<'m> {
    /// Its name, Hugging Face's.
    pub name: String,
    /// What its bytes hold.
    pub kind: Kind,
    /// Its shape.
    pub shape: Vec<usize>,
    /// Its bytes: packed weights' where they lie, or floats'.
    pub bytes: Cow<'m, [u8]>,
}

impl<'m> FileTensor<'m> {
    /// Weights in panels, as the kernels packed them, where they lie.
    fn panels(name: String, weight: &'m quantized::Packed<'_>) -> Result<Self, Error> {
        Ok(Self {
            name,
            kind: Kind::Panels(panels_of(weight.scheme())?),
            shape: vec![weight.depth(), weight.width()],
            bytes: Cow::Borrowed(weight.bytes()),
        })
    }

    /// The tensor as the writer of model files takes it.
    #[must_use]
    pub fn tensor(&self) -> Tensor<'_> {
        Tensor {
            name: &self.name,
            kind: self.kind,
            shape: self.shape.clone(),
            bytes: &self.bytes,
        }
    }

    /// Floats as a model file holds them: in bfloat16 where every one of them
    /// is one, which loses nothing, and in single precision otherwise.
    fn floats(name: String, shape: Vec<usize>, values: &[f32]) -> Self {
        // A bfloat16 is a float of single precision whose lower 16 bits are 0.
        let (kind, bytes) = if values.iter().all(|value| value.to_bits().trailing_zeros() >= 16) {
            let bytes = values
                .iter()
                .flat_map(|value| {
                    let [low, high, _, _] = (value.to_bits() >> 16).to_le_bytes();
                    [low, high]
                })
                .collect();
            (Kind::Bf16, bytes)
        } else {
            (Kind::F32, values.iter().flat_map(|value| value.to_le_bytes()).collect())
        };
        Self {
            name,
            kind,
            shape,
            bytes: Cow::Owned(bytes),
        }
    }
}

impl<'a> Gpt2<'a> {
    /// The model of a model file: its linear layers' and output layer's
    /// weights and its embeddings read where they lie, its normalizations and
    /// biases converted to single precision. Its precision is integers, by the
    /// scheme of its first block's first layer.
    ///
    /// # Errors
    ///
    /// Where the configuration is not one, or a tensor is missing, of another
    /// shape than the configuration's or of another type than the format gives
    /// it, or in panels of another layout than the kernels'.
    pub fn from_file(file: &ModelFile<'a>) -> Result<Self, Error> {
        let config = Config::from_bytes(file.configuration())?;
        let width = config.width;
        let tensor = |name: &str, shape: &[usize]| -> Result<&Tensor<'a>, Error> {
            let tensor = file
                .tensor(name)
                .ok_or_else(|| Error::Tensor(format!("no tensor {name}")))?;
            if tensor.shape != shape {
                return Err(Error::Tensor(format!(
                    "{name} of shape {:?}, not {shape:?}",
                    tensor.shape
                )));
            }
            Ok(tensor)
        };
        let floats = |name: &str, shape: &[usize]| -> Result<Floats<'a>, Error> {
            tensor(name, shape)?
                .floats()
                .ok_or_else(|| Error::Tensor(format!("{name}, not floats")))
        };
        let norm = |name: &str| -> Result<Norm, Error> {
            Ok(Norm {
                gain: floats(&format!("{name}.weight"), &[width])?.to_vec(),
                bias: floats(&format!("{name}.bias"), &[width])?.to_vec(),
            })
        };
        let panels = |name: &str, inputs: usize, outputs: usize| -> Result<quantized::Packed<'a>, Error> {
            let tensor = tensor(name, &[inputs, outputs])?;
            let Kind::Panels(panels) = tensor.kind else {
                return Err(Error::Tensor(format!("{name}, not weights in panels")));
            };
            let scheme = scheme_of(&panels)
                .ok_or_else(|| Error::Tensor(format!("{name}, in panels of another layout than the kernels'")))?;
            quantized::Packed::from_bytes(inputs, outputs, scheme, tensor.bytes)
                .ok_or_else(|| Error::Tensor(format!("{name}, not the bytes of its panels")))
        };
        let linear = |name: &str, inputs: usize, outputs: usize| -> Result<Linear<'a>, Error> {
            Ok(Linear {
                weight: Weights::Int8(panels(&format!("{name}.weight"), inputs, outputs)?),
                bias: floats(&format!("{name}.bias"), &[outputs])?.to_vec(),
            })
        };
        let blocks = (0..config.layers)
            .map(|layer| {
                let name = |part: &str| block_name(layer, part);
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
        let first = panels(&block_name(0, "attn.c_attn.weight"), width, 3 * width)?;
        Ok(Self {
            config,
            precision: Precision::Integers(first.scheme()),
            token_embedding: Table::File(floats("transformer.wte.weight", &[config.vocabulary, width])?),
            position_embedding: Table::File(floats("transformer.wpe.weight", &[config.positions, width])?),
            blocks,
            final_norm: norm("transformer.ln_f")?,
            head: Some(panels("lm_head.weight", width, config.vocabulary)?),
            head_bias: vec![0.0; config.vocabulary],
        })
    }

    /// The model's tensors as a model file holds them, by Hugging Face's names,
    /// in the order the file lays them out: the embeddings, the blocks', the
    /// last normalization's, and the output layer's, as [`Gpt2::from_file`]
    /// reads them. The weights of the linear layers and of the output layer are
    /// the packed bytes, where they lie; the other tensors' floats are in
    /// bfloat16 where every one of them is one, as zenz-v3.1's are, and in
    /// single precision otherwise.
    ///
    /// # Errors
    ///
    /// Where the model is in single precision: a model file holds a model in
    /// integers.
    pub fn to_tensors(&self) -> Result<Vec<FileTensor<'_>>, Error> {
        let single = || Error::Config("a model in single precision, which a model file does not hold".to_owned());
        let Some(head) = &self.head else {
            return Err(single());
        };
        let Config {
            width,
            positions,
            vocabulary,
            ..
        } = self.config;
        let mut tensors = vec![
            FileTensor::floats(
                "transformer.wte.weight".to_owned(),
                vec![vocabulary, width],
                &self.token_embedding.to_vec(),
            ),
            FileTensor::floats(
                "transformer.wpe.weight".to_owned(),
                vec![positions, width],
                &self.position_embedding.to_vec(),
            ),
        ];
        let norm = |tensors: &mut Vec<FileTensor<'_>>, name: &str, norm: &Norm| {
            tensors.push(FileTensor::floats(format!("{name}.weight"), vec![width], &norm.gain));
            tensors.push(FileTensor::floats(format!("{name}.bias"), vec![width], &norm.bias));
        };
        for (layer, block) in self.blocks.iter().enumerate() {
            let name = |part: &str| block_name(layer, part);
            norm(&mut tensors, &name("ln_1"), &block.attention_norm);
            for (part, linear) in [("attn.c_attn", &block.attention), ("attn.c_proj", &block.projection)] {
                let Weights::Int8(weight) = &linear.weight else {
                    return Err(single());
                };
                tensors.push(FileTensor::panels(name(&format!("{part}.weight")), weight)?);
                tensors.push(FileTensor::floats(
                    name(&format!("{part}.bias")),
                    vec![linear.bias.len()],
                    &linear.bias,
                ));
            }
            norm(&mut tensors, &name("ln_2"), &block.expansion_norm);
            for (part, linear) in [("mlp.c_fc", &block.expansion), ("mlp.c_proj", &block.contraction)] {
                let Weights::Int8(weight) = &linear.weight else {
                    return Err(single());
                };
                tensors.push(FileTensor::panels(name(&format!("{part}.weight")), weight)?);
                tensors.push(FileTensor::floats(
                    name(&format!("{part}.bias")),
                    vec![linear.bias.len()],
                    &linear.bias,
                ));
            }
        }
        norm(&mut tensors, "transformer.ln_f", &self.final_norm);
        tensors.push(FileTensor::panels("lm_head.weight".to_owned(), head)?);
        Ok(tensors)
    }
}

impl Gpt2<'_> {
    /// Its sizes.
    #[must_use]
    pub fn config(&self) -> &Config {
        &self.config
    }

    /// What its products are computed in.
    #[must_use]
    pub fn precision(&self) -> Precision {
        self.precision
    }
}

/// A sequence that a model computes, a token after another: the cache of the
/// keys and values of its positions, and the room to compute in.
#[derive(Debug)]
pub struct Session<'m> {
    model: &'m Gpt2<'m>,
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
    pub fn new(model: &'m Gpt2<'m>, capacity: usize) -> Self {
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
        let config = self.model.config;
        assert_eq!(
            output.len(),
            config.vocabulary,
            "logits of another length than the vocabulary"
        );
        let (stream, mut room) = self.forward(tokens)?;
        let last = &stream[(tokens.len() - 1) * config.width..];
        self.head(last, &mut room, output);
        Ok(())
    }

    /// Computes the tokens at the next positions, and writes the logits after
    /// each of them, a row of the vocabulary's for each token: what checking a
    /// draft's tokens in one decode needs.
    ///
    /// # Errors
    ///
    /// Where there are no tokens, a token is out of the vocabulary, or the
    /// positions would pass the room of the session.
    ///
    /// # Panics
    ///
    /// Where `output` is not a row of logits for each token.
    pub fn feed_all(&mut self, tokens: &[u32], output: &mut [f32]) -> Result<(), Error> {
        assert_eq!(
            output.len(),
            tokens.len() * self.model.config.vocabulary,
            "logits that are not a row of the vocabulary's for each token"
        );
        let (stream, mut room) = self.forward(tokens)?;
        self.head(&stream, &mut room, output);
        Ok(())
    }

    /// The residual stream after the blocks, of the tokens at the next
    /// positions, whose keys and values it caches; and the room it was
    /// computed in.
    fn forward(&mut self, tokens: &[u32]) -> Result<(Vec<f32>, Room), Error> {
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
        let mut stream = self.embed(tokens)?;
        let mut room = Room::new(&config, rows, self.length + rows);
        for (layer, block) in model.blocks.iter().enumerate() {
            self.block(layer, block, &mut stream, &mut room);
        }
        self.length += rows;
        Ok((stream, room))
    }

    /// The logits of rows of the stream: each normalized, then its products
    /// with the token embedding.
    fn head(&self, stream: &[f32], room: &mut Room, output: &mut [f32]) {
        let model = self.model;
        let config = model.config;
        let mut hidden = vec![0.0f32; stream.len()];
        model.final_norm.apply(stream, config.epsilon, &mut hidden);
        if let Some(head) = &model.head {
            room.quantized.quantize(&hidden, config.width, head.scheme().rotation);
            quantized::linear(&room.quantized, head, &model.head_bias, output);
        } else {
            let Table::Single(embedding) = &model.token_embedding else {
                unreachable!("a model in single precision, which a model file does not hold, read from one");
            };
            for (hidden, output) in hidden
                .chunks_exact(config.width)
                .zip(output.chunks_exact_mut(config.vocabulary))
            {
                logits(hidden, embedding, output);
            }
        }
    }

    /// The residual stream of the tokens at the next positions: their tokens'
    /// and their positions' embeddings.
    fn embed(&self, tokens: &[u32]) -> Result<Vec<f32>, Error> {
        let model = self.model;
        let config = model.config;
        let width = config.width;
        let mut stream = vec![0.0f32; tokens.len() * width];
        let mut position = vec![0.0f32; width];
        for (row, (&token, embedded)) in tokens.iter().zip(stream.chunks_exact_mut(width)).enumerate() {
            let token = usize::try_from(token)
                .ok()
                .filter(|&token| token < config.vocabulary)
                .ok_or_else(|| Error::Input(format!("the token {token}, out of the vocabulary")))?;
            model.token_embedding.row(token, embedded);
            model.position_embedding.row(self.length + row, &mut position);
            add(embedded, &position);
        }
        Ok(stream)
    }

    /// A block of the decoder on the stream of the rows fed, whose keys and
    /// values it caches.
    fn block(&mut self, layer: usize, block: &Block<'_>, stream: &mut [f32], room: &mut Room) {
        let epsilon = self.model.config.epsilon;
        block.attention_norm.apply(stream, epsilon, &mut room.normalized);
        block
            .attention
            .apply(&room.normalized, &mut room.quantized, &mut room.projected);
        self.cache(layer, &room.projected);
        self.attend(layer, &room.projected, &mut room.scores, &mut room.found);
        block
            .projection
            .apply(&room.found, &mut room.quantized, &mut room.normalized);
        add(stream, &room.normalized);
        block.expansion_norm.apply(stream, epsilon, &mut room.normalized);
        block
            .expansion
            .apply(&room.normalized, &mut room.quantized, &mut room.expanded);
        gelu_tanh(&mut room.expanded);
        block
            .contraction
            .apply(&room.expanded, &mut room.quantized, &mut room.normalized);
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
