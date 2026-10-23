// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later
//
// This file is part of KIRAL. The SPDX lines name its copyright holder and its
// license, in the machine-readable form of the REUSE specification: the GNU
// General Public License, version 3 or any later version
// (LICENSES/GPL-3.0-or-later.txt).

//! The linear layers in integers: `output = input weight + bias`, the weights
//! and the inputs quantized to bytes.
//!
//! The weights are signed and symmetric, a weight `w = s q` with `|q| <= 127`
//! at eight bits (`2^(b - 1) - 1` at `b`); the inputs unsigned with a zero
//! point, `x = s (q - z)` with `0 <= q <= 255` -- `127` for the form of AVX2,
//! whose sums of two products in a word would pass 32767 otherwise
//! ([`crate::Form::input_max`]) -- the range of each group of them from its
//! least to its greatest, 0 within it: the inputs of a block's
//! contraction, after GELU, lie mostly on one side of 0, where a symmetric
//! scale would waste half its range. A scale holds for a group of [`GROUP`]
//! numbers of a sum, a weight's for its column's group and an input's for its
//! row's, so that one large number costs only its own group its precision. The
//! products of a group are summed exactly in 32 bits -- four products of a byte
//! by a byte to a lane, as VNNI's `vpdpbusd` sums them -- and added to the
//! output in single precision, scaled:
//!
//! ```text
//! output[i][j] += sa[i] sb[j] sum(qa qb) - oa[i] ob[j]
//! ```
//!
//! `sa` and `sb` the scales of row `i`'s and column `j`'s group, `oa` the
//! input's offset `sa z` and `ob` the weight's `sb sum(qb)` over the group:
//! `sa sb sum((qa - z) qb) = sa sb sum(qa qb) - (sa z)(sb sum(qb))`.
//!
//! The weights are packed once ([`Packed`]), in panels of [`PANEL`] columns, a
//! group after another: the group's bytes quad by quad, sixteen quads of
//! sixteen columns of four bytes, the four bytes of a column's quad side by
//! side -- a vector of `vpdpbusd` -- then the panel's sixteen scales and sixteen
//! offsets. They are packed when a model is converted, and read where they lie
//! in its file ([`Packed::from_bytes`]). The inputs are quantized as they come,
//! each row's groups in one piece ([`Rows`]).
//!
//! A scheme ([`Scheme`]) may rotate the depth first, as ConvRot does for the
//! diffusion transformers (Huang et al., 2025): each block of the depth of a
//! power of four's length, the weights' when they are packed and the inputs'
//! as they are quantized, by a regular Hadamard matrix of its order --
//! orthogonal, so that the products are unchanged, and spreading a large number
//! over its block, so that its group's scale holds the others less coarsely.
//! A regular Hadamard matrix, each of whose rows and columns sums to the square
//! root of its order, is taken rather than Sylvester's, whose first column of
//! ones adds a row's numbers of one sign up into one. The scheme may also
//! quantize the weights to fewer bits than eight, each still held in a byte,
//! and take each group's scale as the one of least squared error rather than
//! that of its greatest magnitude.
//!
//! The portable form keeps a panel's four products of each column of a quad in
//! lanes of their own, down the quads of a group, and adds a column's four at
//! the group's end: the compiler makes vector products of them
//! (`pmovsxbd` and `pmaddwd` on x86-64-v2), where adding the four within each
//! quad would have it shuffle the weights first.

use std::borrow::Cow;

use crate::{Form, multiply_add};

/// The numbers of a sum that a scale holds for: sixteen quads.
pub const GROUP: usize = 64;

/// The quads of a group.
pub const QUADS: usize = GROUP / 4;

/// The columns of a panel: a quad of each, 64 bytes, is a line of the caches.
pub const PANEL: usize = 16;

/// The greatest integer of an input, a byte's; the form of AVX2 takes inputs of
/// seven bits ([`crate::Form::input_max`]).
pub const INPUT_MAX: f32 = 255.0;

/// The bytes of a group of a panel: its quads, then its scales and offsets.
pub const GROUP_BYTES: usize = QUADS * PANEL * 4 + 2 * PANEL * 4;

/// The columns of a panel a tile sums at once, half of it: their products are
/// 32 lanes, eight registers of SSE, which leave the weights and the input
/// room among its sixteen, where a whole panel's 64 lanes would be spilled to
/// the stack at every quad.
pub const STRIP: usize = 8;

/// The rows of the input a tile takes at once: one, as a second row's lanes
/// spill again (two rows of a strip were the slower on a P-core, by 7 to 10%,
/// with 4, 16 and 40 rows of a layer of 768 by 3072).
pub const ROWS: usize = 1;

/// How a scheme rotates the depth before it quantizes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Rotation {
    /// Not at all.
    None,
    /// Each block of this many numbers of the depth, a power of four, by the
    /// regular Hadamard matrix of its order ([`rotate`]).
    Hadamard(usize),
}

/// How a layer's weights and inputs are quantized.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Scheme {
    /// The bits of a weight's integer, from 2 to 8, held in a byte whatever
    /// they are.
    pub weight_bits: u32,
    /// The rotation of the depth, the weights' and the inputs'.
    pub rotation: Rotation,
    /// Whether a weight group's scale is the one of least squared error among
    /// those that clip its greatest magnitude by up to a half, rather than that
    /// of its greatest magnitude.
    pub fitted: bool,
}

impl Scheme {
    /// Eight bits a weight, no rotation, the scale of a group's greatest
    /// magnitude.
    pub const INT8: Self = Self {
        weight_bits: 8,
        rotation: Rotation::None,
        fitted: false,
    };

    /// The greatest magnitude of a weight's integer: 2^(bits - 1) - 1.
    #[must_use]
    pub fn weight_max(&self) -> f32 {
        f32::from(u8::MAX >> (9 - self.weight_bits.clamp(2, 8)))
    }
}

/// A linear layer's weights, quantized and packed for [`linear`]: packed here,
/// or read where they lie in a model file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Packed<'a> {
    depth: usize,
    width: usize,
    groups: usize,
    scheme: Scheme,
    /// The panels, each its groups one after another.
    bytes: Cow<'a, [u8]>,
}

impl Packed<'static> {
    /// The weights of a layer of `depth` inputs and `width` outputs, `weight(k,
    /// j)` the weight of input `k` in output `j`, quantized by a scheme: each
    /// column rotated, then each column's group to its scale, a weight to the
    /// nearest integer of it, ties to even, within the scheme's greatest
    /// magnitude. The groups and panels past the layer's depth and width are
    /// zeros of no scale.
    ///
    /// # Panics
    ///
    /// Where `depth` or `width` is 0, or the scheme's weights are not of 2 to
    /// 8 bits, or its rotation not of a power of four.
    #[must_use]
    pub fn quantize(depth: usize, width: usize, scheme: Scheme, weight: impl Fn(usize, usize) -> f32) -> Self {
        assert!(
            depth > 0 && width > 0,
            "a linear layer of {depth} inputs and {width} outputs"
        );
        assert!(
            (2..=8).contains(&scheme.weight_bits),
            "weights of {} bits",
            scheme.weight_bits
        );
        let groups = depth.div_ceil(GROUP);
        let panels = width.div_ceil(PANEL);
        let largest = scheme.weight_max();
        let mut bytes = vec![0u8; panels * groups * GROUP_BYTES];
        let mut column_values = vec![0.0f32; depth];
        for (panel, panel_bytes) in bytes.chunks_exact_mut(groups * GROUP_BYTES).enumerate() {
            for column in 0..PANEL.min(width - panel * PANEL) {
                let j = panel * PANEL + column;
                for (k, value) in column_values.iter_mut().enumerate() {
                    *value = weight(k, j);
                }
                if let Rotation::Hadamard(order) = scheme.rotation {
                    rotate(&mut column_values, order);
                }
                for (group, group_bytes) in panel_bytes.as_chunks_mut::<GROUP_BYTES>().0.iter_mut().enumerate() {
                    let values = &column_values[group * GROUP..depth.min((group + 1) * GROUP)];
                    let scale = weight_scale(values, largest, scheme.fitted);
                    let inverse = if scale > 0.0 { 1.0 / scale } else { 0.0 };
                    let (quads, floats) = group_bytes.split_at_mut(QUADS * PANEL * 4);
                    let (scales, offsets) = floats.split_at_mut(PANEL * 4);
                    let mut sum = 0i32;
                    for (e, &value) in values.iter().enumerate() {
                        let q = small((value * inverse).round_ties_even().clamp(-largest, largest));
                        sum += i32::from(q);
                        quads[(e / 4 * PANEL + column) * 4 + e % 4] = q.cast_unsigned();
                    }
                    scales[column * 4..][..4].copy_from_slice(&scale.to_le_bytes());
                    offsets[column * 4..][..4].copy_from_slice(&(scale * exact(sum)).to_le_bytes());
                }
            }
        }
        Self {
            depth,
            width,
            groups,
            scheme,
            bytes: Cow::Owned(bytes),
        }
    }
}

impl<'a> Packed<'a> {
    /// The weights of a layer of `depth` inputs and `width` outputs that a
    /// scheme quantized and packed into `bytes`, read where they lie: the
    /// panels of [`PANEL`] columns, each its groups of [`GROUP_BYTES`], as
    /// [`Packed::quantize`] lays them out. None where the bytes are of another
    /// length than the layer's panels, either size is 0, or the scheme's weights
    /// are not of 2 to 8 bits or its rotation not of a power of four.
    #[must_use]
    pub fn from_bytes(depth: usize, width: usize, scheme: Scheme, bytes: &'a [u8]) -> Option<Self> {
        let rotation = match scheme.rotation {
            Rotation::None => true,
            Rotation::Hadamard(order) => order.is_power_of_two() && order.trailing_zeros().is_multiple_of(2),
        };
        let groups = depth.div_ceil(GROUP);
        let length = width
            .div_ceil(PANEL)
            .checked_mul(groups)
            .and_then(|count| count.checked_mul(GROUP_BYTES));
        (depth > 0 && width > 0 && (2..=8).contains(&scheme.weight_bits) && rotation && length == Some(bytes.len()))
            .then_some(Self {
                depth,
                width,
                groups,
                scheme,
                bytes: Cow::Borrowed(bytes),
            })
    }

    /// The scheme the weights were quantized by.
    #[must_use]
    pub fn scheme(&self) -> Scheme {
        self.scheme
    }

    /// The inputs of the layer.
    #[must_use]
    pub fn depth(&self) -> usize {
        self.depth
    }

    /// The outputs of the layer.
    #[must_use]
    pub fn width(&self) -> usize {
        self.width
    }

    /// The packed bytes: the panels of [`PANEL`] columns, each the layer's
    /// groups of [`GROUP_BYTES`].
    #[must_use]
    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }
}

/// The scale of a group of weights: that of its greatest magnitude, or, where
/// `fitted`, the one of least squared error among those that clip it by up to
/// a half, in steps of a fiftieth.
fn weight_scale(values: &[f32], largest: f32, fitted: bool) -> f32 {
    let greatest = values.iter().fold(0.0f32, |greatest, value| greatest.max(value.abs()));
    let mut best = greatest / largest;
    if fitted && best > 0.0 {
        let error = |scale: f32| -> f32 {
            let inverse = 1.0 / scale;
            values
                .iter()
                .map(|&value| {
                    let error = value - scale * (value * inverse).round_ties_even().clamp(-largest, largest);
                    error * error
                })
                .sum()
        };
        let mut least = error(best);
        for step in 1..=25u8 {
            let scale = greatest * (1.0 - f32::from(step) / 50.0) / largest;
            let error = error(scale);
            if error < least {
                (best, least) = (scale, error);
            }
        }
    }
    best
}

/// Rotates each block of `order` values, a power of four, by the regular
/// Hadamard matrix of that order divided by the square root of the order, so
/// that it is orthogonal; the values left after the last whole block in blocks
/// of the largest powers of four that fit. The matrix is the Kronecker power of
///
/// ```text
///  1  1  1 -1
///  1  1 -1  1
///  1 -1  1  1
/// -1  1  1  1
/// ```
///
/// each of whose rows and columns sums to 2, so that each of the power's sums
/// to the square root of its order; it is its own transpose and its own
/// inverse. It is applied a factor at a time, on the digits of the values'
/// indices in base four: a block of four becomes `s - 2 x[3 - i]`, `s` the
/// block's sum.
///
/// # Panics
///
/// Where `order` is not a power of four.
pub fn rotate(values: &mut [f32], order: usize) {
    assert!(
        order.is_power_of_two() && order.trailing_zeros().is_multiple_of(2),
        "a rotation of order {order}, not a power of four"
    );
    let mut rest = values;
    let mut order = order;
    while !rest.is_empty() {
        while order > rest.len() {
            order /= 4;
        }
        let (block, after) = rest.split_at_mut(order);
        hadamard(block);
        rest = after;
    }
}

/// A block of a power of four's length times the regular Hadamard matrix of its
/// order, divided by the square root of the order.
fn hadamard(block: &mut [f32]) {
    let order = block.len();
    let mut stride = 1;
    while stride < order {
        for start in (0..order).step_by(4 * stride) {
            for offset in start..start + stride {
                let x = [
                    block[offset],
                    block[offset + stride],
                    block[offset + 2 * stride],
                    block[offset + 3 * stride],
                ];
                let sum = (x[0] + x[1]) + (x[2] + x[3]);
                for (i, &other) in x.iter().rev().enumerate() {
                    block[offset + i * stride] = sum - 2.0 * other;
                }
            }
        }
        stride *= 4;
    }
    // 1 / sqrt(order), a power of two: exact.
    let scale = 1.0 / f32::from(u16::try_from(order.isqrt()).expect("a block's order below 2^32"));
    for value in block {
        *value *= scale;
    }
}

/// Rows of inputs quantized for [`linear`]: each row's groups of [`GROUP`]
/// bytes in one piece, and each row's groups' scales and offsets, a group's
/// scale and offset side by side.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Rows {
    depth: usize,
    groups: usize,
    rotation: Option<usize>,
    /// The greatest integer of an input: [`INPUT_MAX`], or 127.
    largest: f32,
    bytes: Vec<u8>,
    /// Row by row, group by group, the scale and then the offset.
    floats: Vec<f32>,
    /// A row rotated.
    rotated: Vec<f32>,
}

impl Rows {
    /// No rows.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Quantizes rows of `depth` inputs in place of those held, for weights of
    /// a rotation, for the form of the kernels in effect ([`crate::form`]): as
    /// [`Rows::quantize_in`] does.
    ///
    /// # Panics
    ///
    /// As [`Rows::quantize_in`].
    pub fn quantize(&mut self, input: &[f32], depth: usize, rotation: Rotation) {
        self.quantize_in(crate::form(), input, depth, rotation);
    }

    /// Quantizes rows of `depth` inputs in place of those held, for weights of
    /// a rotation and the products of a form: each row rotated, then each group
    /// to the range from its least to its greatest input, 0 within it, an input
    /// to the nearest integer of the scale, ties to even, plus the zero point,
    /// the integer 0 stands for, up to the form's greatest integer
    /// ([`crate::Form::input_max`]). The group past the depth holds the zero
    /// point.
    ///
    /// # Panics
    ///
    /// Where `depth` is 0, `input` is not whole rows of it, or the rotation is
    /// not of a power of four.
    pub fn quantize_in(&mut self, form: Form, input: &[f32], depth: usize, rotation: Rotation) {
        assert!(depth > 0, "rows of no inputs");
        assert_eq!(input.len() % depth, 0, "an input that is not whole rows of {depth}");
        let groups = depth.div_ceil(GROUP);
        let rows = input.len() / depth;
        let largest = form.input_max();
        self.depth = depth;
        self.groups = groups;
        self.largest = largest;
        self.rotation = match rotation {
            Rotation::None => None,
            Rotation::Hadamard(order) => Some(order),
        };
        self.bytes.resize(rows * groups * GROUP, 0);
        self.floats.resize(rows * groups * 2, 0.0);
        for (row, input) in input.chunks_exact(depth).enumerate() {
            let input = if let Some(order) = self.rotation {
                self.rotated.clear();
                self.rotated.extend_from_slice(input);
                rotate(&mut self.rotated, order);
                &self.rotated
            } else {
                input
            };
            for group in 0..groups {
                let values = &input[group * GROUP..depth.min((group + 1) * GROUP)];
                let at = row * groups + group;
                let bytes = &mut self.bytes[at * GROUP..][..GROUP];
                let (scale, offset) = quantize_group(values, bytes, largest);
                self.floats[2 * at] = scale;
                self.floats[2 * at + 1] = offset;
            }
        }
    }

    /// The greatest integer of an input of the rows: [`INPUT_MAX`], or 127 for
    /// the form of AVX2.
    #[must_use]
    pub fn input_max(&self) -> f32 {
        self.largest
    }

    /// The rotation the rows were quantized after.
    #[must_use]
    pub fn rotation(&self) -> Rotation {
        self.rotation.map_or(Rotation::None, Rotation::Hadamard)
    }

    /// The rows held.
    #[must_use]
    pub fn len(&self) -> usize {
        self.floats.len().checked_div(2 * self.groups).unwrap_or(0)
    }

    /// Whether no rows are held.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.floats.is_empty()
    }

    /// The inputs of a row.
    #[must_use]
    pub fn depth(&self) -> usize {
        self.depth
    }

    /// The bytes of the rows, each its groups of [`GROUP`] one after another.
    #[must_use]
    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// The scale of a row's group.
    ///
    /// # Panics
    ///
    /// Where the row or the group is not held.
    #[must_use]
    pub fn scale(&self, row: usize, group: usize) -> f32 {
        assert!(group < self.groups, "group {group} of {}", self.groups);
        self.floats[2 * (row * self.groups + group)]
    }

    /// The offset of a row's group: its scale times its zero point.
    ///
    /// # Panics
    ///
    /// Where the row or the group is not held.
    #[must_use]
    pub fn offset(&self, row: usize, group: usize) -> f32 {
        assert!(group < self.groups, "group {group} of {}", self.groups);
        self.floats[2 * (row * self.groups + group) + 1]
    }
}

/// 2^23: a float from 0 to 2^23 added to it leaves its nearest integer, ties to
/// even, in the sum's lowest bits.
const ROUNDER: f32 = 8_388_608.0;

/// Quantizes a group of inputs, at most [`GROUP`], into its bytes; returns the
/// group's scale and offset. The least and the greatest are taken in lanes of
/// sixteen, which the compiler keeps in vectors, and then among the lanes, by
/// comparisons, which leave a NaN out as `min` and `max` do, in one
/// instruction each; an input's integer is its value, scaled, plus the zero
/// point plus 2^23, which rounds it, kept within 0 and 255, and its lowest byte
/// taken. So a group took a third of the time it took rounded
/// by `round_ties_even` and converted by `as`, saturating, with the same bytes
/// (20 rows of 3072: 0.0377 ms against 0.1084 on a P-core of an i7-1360P at
/// 4.7 GHz, the medians of repeated calls).
fn quantize_group(values: &[f32], bytes: &mut [u8], largest: f32) -> (f32, f32) {
    let mut least = [0.0f32; 16];
    let mut greatest = [0.0f32; 16];
    let (chunks, rest) = values.as_chunks::<16>();
    for chunk in chunks {
        for ((least, greatest), &value) in least.iter_mut().zip(greatest.iter_mut()).zip(chunk) {
            *least = if value < *least { value } else { *least };
            *greatest = if value > *greatest { value } else { *greatest };
        }
    }
    for &value in rest {
        least[0] = if value < least[0] { value } else { least[0] };
        greatest[0] = if value > greatest[0] { value } else { greatest[0] };
    }
    let least = least
        .into_iter()
        .fold(0.0f32, |least, value| if value < least { value } else { least });
    let greatest = greatest.into_iter().fold(
        0.0f32,
        |greatest, value| if value > greatest { value } else { greatest },
    );
    let scale = (greatest - least) / largest;
    let inverse = if scale > 0.0 { 1.0 / scale } else { 0.0 };
    let zero = (-least * inverse).round_ties_even();
    let shift = zero + ROUNDER;
    let (inputs, past) = bytes.split_at_mut(values.len());
    for (byte, &value) in inputs.iter_mut().zip(values) {
        *byte = lowest_byte(multiply_add(value, inverse, shift).clamp(ROUNDER, ROUNDER + largest));
    }
    past.fill(lowest_byte(shift));
    (scale, scale * zero)
}

/// The lowest byte of a float's bits: its integer, where it is 2^23 plus an
/// integer from 0 to 255.
#[inline]
fn lowest_byte(value: f32) -> u8 {
    value.to_bits().to_le_bytes()[0]
}

/// `output = input weight + bias`, row by row, by the form of the kernels in
/// effect ([`crate::form`]).
///
/// `input` holds rows of the layer's depth, `bias` a value for each of its
/// outputs, and `output` a row of them for each row of `input`.
///
/// # Panics
///
/// Where the lengths do not fit the layer's, or the inputs were rotated
/// otherwise than the weights.
pub fn linear(input: &Rows, weight: &Packed<'_>, bias: &[f32], output: &mut [f32]) {
    linear_in(crate::form(), input, weight, bias, output);
}

/// `output = input weight + bias`, row by row, by a form of the kernels.
///
/// # Panics
///
/// As [`linear`]; and where the form is one this build or this processor has
/// not got.
pub fn linear_in(form: Form, input: &Rows, weight: &Packed<'_>, bias: &[f32], output: &mut [f32]) {
    let rows = check(form, input, weight, bias);
    let width = weight.width;
    assert_eq!(
        output.len(),
        rows * width,
        "an output that is not {rows} rows of {width}"
    );
    sums_in(form, input, weight, bias, |row, column, sums| {
        output[row * width + column..][..sums.len()].copy_from_slice(sums);
    });
}

/// The likeliest output of each row of `input weight + bias`, by the form of
/// the kernels in effect ([`crate::form`]): the index of its largest value,
/// the first of those equal to it, or none for a row of NaNs alone, as
/// [`crate::vector::argmax`] takes it. The outputs are summed as [`linear`]
/// sums them, a tile of a panel's columns at a time, and each row keeps only
/// its largest so far: the output layer of a model whose verification needs
/// only the likeliest token, its logits never written out.
///
/// # Panics
///
/// As [`linear`], `output` holding a value for each row of `input`.
pub fn likeliest(input: &Rows, weight: &Packed<'_>, bias: &[f32], output: &mut [Option<usize>]) {
    likeliest_in(crate::form(), input, weight, bias, output);
}

/// The likeliest output of each row of `input weight + bias`, by a form of the
/// kernels.
///
/// # Panics
///
/// As [`likeliest`]; and where the form is one this build or this processor
/// has not got.
pub fn likeliest_in(form: Form, input: &Rows, weight: &Packed<'_>, bias: &[f32], output: &mut [Option<usize>]) {
    let rows = check(form, input, weight, bias);
    assert_eq!(
        output.len(),
        rows,
        "an output that is not a value for each of {rows} rows"
    );
    output.fill(None);
    let mut largest = vec![0.0f32; rows];
    // The columns of a row come in their order, so that a later value equal to
    // the largest leaves the first.
    sums_in(form, input, weight, bias, |row, column, sums| {
        for (j, &sum) in sums.iter().enumerate() {
            if sum > largest[row] || (output[row].is_none() && !sum.is_nan()) {
                largest[row] = sum;
                output[row] = Some(column + j);
            }
        }
    });
}

/// Checks that the input, the weights and the bias fit together, and the
/// inputs' integers the form's; returns the rows of the input.
fn check(form: Form, input: &Rows, weight: &Packed<'_>, bias: &[f32]) -> usize {
    let Packed {
        depth, width, scheme, ..
    } = *weight;
    assert_eq!(input.depth, depth, "inputs of another depth than the layer's {depth}");
    assert_eq!(
        input.rotation(),
        scheme.rotation,
        "inputs rotated otherwise than the weights"
    );
    assert_eq!(bias.len(), width, "a bias of another width than the layer's {width}");
    assert!(
        input.is_empty() || input.largest <= form.input_max(),
        "inputs up to {}, past the {form} form's {}",
        input.largest,
        form.input_max()
    );
    input.len()
}

/// The outputs of `input weight + bias`, by a form of the kernels, given to
/// `each` a row's run of columns at a time: `each(row, column, sums)`, the
/// sums of the row's outputs from `column` on. Each row's runs come in the
/// order of their columns.
fn sums_in(form: Form, input: &Rows, weight: &Packed<'_>, bias: &[f32], each: impl FnMut(usize, usize, &[f32])) {
    #[cfg(target_arch = "x86_64")]
    use crate::x86_64::{Products, TILE_ROWS, qgemm};
    match form {
        Form::Portable => portable(input, weight, bias, each),
        #[cfg(target_arch = "x86_64")]
        Form::Avx2 => tiled::<TILE_ROWS>(input, weight, bias, each, |rows, groups, a, floats, panel, tile| {
            qgemm(Products::Avx2, rows, groups, a, floats, panel, tile);
        }),
        #[cfg(target_arch = "x86_64")]
        Form::AvxVnni => tiled::<TILE_ROWS>(input, weight, bias, each, |rows, groups, a, floats, panel, tile| {
            qgemm(Products::AvxVnni, rows, groups, a, floats, panel, tile);
        }),
        #[cfg(target_arch = "x86_64")]
        Form::Avx512Vnni => tiled::<TILE_ROWS>(input, weight, bias, each, |rows, groups, a, floats, panel, tile| {
            qgemm(Products::Avx512Vnni, rows, groups, a, floats, panel, tile);
        }),
        #[cfg(target_arch = "aarch64")]
        Form::Dotprod => {
            tiled::<{ crate::aarch64::TILE_ROWS }>(input, weight, bias, each, crate::aarch64::qgemm_dotprod);
        }
        // The forms of another architecture than the build's.
        _ => panic!("the {form} kernels on another architecture than theirs"),
    }
}

/// The portable form of [`sums_in`]: strips of [`STRIP`] columns, a row at a
/// time.
fn portable(input: &Rows, weight: &Packed<'_>, bias: &[f32], mut each: impl FnMut(usize, usize, &[f32])) {
    let Packed {
        width,
        groups,
        ref bytes,
        ..
    } = *weight;
    let rows = input.len();
    let row_bytes = groups * GROUP;
    for (panel, panel_bytes) in bytes.chunks_exact(groups * GROUP_BYTES).enumerate() {
        for strip in (0..PANEL).step_by(STRIP) {
            let column = panel * PANEL + strip;
            if column >= width {
                break;
            }
            let columns = STRIP.min(width - column);
            let mut first = [0.0f32; STRIP];
            first[..columns].copy_from_slice(&bias[column..][..columns]);
            let tile_of = |row: usize, take: usize| Tile {
                bytes: &input.bytes[row * row_bytes..][..take * row_bytes],
                floats: &input.floats[row * groups * 2..][..take * groups * 2],
                groups,
            };
            let mut give = |row: usize, sums: &[[f32; STRIP]]| {
                for (r, sums) in sums.iter().enumerate() {
                    each(row + r, column, &sums[..columns]);
                }
            };
            // Whole tiles, then the rows they leave one at a time.
            let tiled = rows / ROWS * ROWS;
            for row in (0..tiled).step_by(ROWS) {
                give(row, &tile::<ROWS>(&tile_of(row, ROWS), panel_bytes, strip, &first));
            }
            for row in tiled..rows {
                give(row, &tile::<1>(&tile_of(row, 1), panel_bytes, strip, &first));
            }
        }
    }
}

/// A form in assembly of [`sums_in`]: tiles of up to `T` rows by a panel's
/// sixteen columns, each started from the bias in a tile of its own and summed
/// by the form's kernel, `kernel(rows, groups, a, a_floats, panel, tile)`.
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
fn tiled<const T: usize>(
    input: &Rows,
    weight: &Packed<'_>,
    bias: &[f32],
    mut each: impl FnMut(usize, usize, &[f32]),
    mut kernel: impl FnMut(usize, usize, &[u8], &[f32], &[u8], &mut [[f32; PANEL]; T]),
) {
    let Packed {
        width,
        groups,
        ref bytes,
        ..
    } = *weight;
    let rows = input.len();
    let row_bytes = groups * GROUP;
    let mut tile = [[0.0f32; PANEL]; T];
    for (panel, panel_bytes) in bytes.chunks_exact(groups * GROUP_BYTES).enumerate() {
        let column = panel * PANEL;
        let columns = PANEL.min(width - column);
        let mut row = 0;
        while row < rows {
            let take = T.min(rows - row);
            for sums in &mut tile[..take] {
                sums[..columns].copy_from_slice(&bias[column..][..columns]);
                sums[columns..].fill(0.0);
            }
            kernel(
                take,
                groups,
                &input.bytes[row * row_bytes..][..take * row_bytes],
                &input.floats[row * groups * 2..][..take * groups * 2],
                panel_bytes,
                &mut tile,
            );
            for (r, sums) in tile[..take].iter().enumerate() {
                each(row + r, column, &sums[..columns]);
            }
            row += take;
        }
    }
}

/// The quantized rows a tile takes.
#[derive(Debug, Clone, Copy)]
struct Tile<'a> {
    bytes: &'a [u8],
    floats: &'a [f32],
    groups: usize,
}

/// The sums of `R` rows with the [`STRIP`] columns of a panel from `strip`,
/// from `first`: the tile's values of the output. Not inlined: inlined into its
/// caller, it was vectorized along the quads instead, its bytes gathered one at
/// a time, and a row of a layer of 768 by 3072 took 0.43 ms on a P-core
/// against 0.26.
#[inline(never)]
fn tile<const R: usize>(a: &Tile<'_>, panel: &[u8], strip: usize, first: &[f32; STRIP]) -> [[f32; STRIP]; R] {
    let mut sums = [*first; R];
    let row_bytes = a.groups * GROUP;
    for (group, group_bytes) in panel.as_chunks::<GROUP_BYTES>().0.iter().enumerate() {
        let (quads, floats) = group_bytes.split_at(QUADS * PANEL * 4);
        let mut lanes = [[[0i32; 4]; STRIP]; R];
        for (q, weights) in quads.as_chunks::<{ PANEL * 4 }>().0.iter().enumerate() {
            let weights = &weights.as_chunks::<4>().0[strip..][..STRIP];
            for (r, lanes) in lanes.iter_mut().enumerate() {
                let at = r * row_bytes + group * GROUP + 4 * q;
                let x: [i32; 4] = std::array::from_fn(|t| i32::from(a.bytes[at + t]));
                for (lanes, weights) in lanes.iter_mut().zip(weights) {
                    for ((lane, &x), &weight) in lanes.iter_mut().zip(&x).zip(weights) {
                        *lane += x * i32::from(weight.cast_signed());
                    }
                }
            }
        }
        let (scales, offsets) = floats.split_at(PANEL * 4);
        let scales = floats_of(scales);
        let offsets = floats_of(offsets);
        for (r, (sums, lanes)) in sums.iter_mut().zip(&lanes).enumerate() {
            let scale = a.floats[2 * (r * a.groups + group)];
            let offset = a.floats[2 * (r * a.groups + group) + 1];
            for (((sum, lanes), &weight_scale), &weight_offset) in
                sums.iter_mut().zip(lanes).zip(&scales[strip..]).zip(&offsets[strip..])
            {
                let products = exact(lanes[0] + lanes[1] + lanes[2] + lanes[3]);
                *sum = multiply_add(scale * weight_scale, products, *sum);
                *sum = multiply_add(-offset, weight_offset, *sum);
            }
        }
    }
    sums
}

/// The sixteen floats of a panel's scales or offsets.
#[inline]
fn floats_of(bytes: &[u8]) -> [f32; PANEL] {
    let (chunks, _) = bytes.as_chunks::<4>();
    std::array::from_fn(|column| f32::from_le_bytes(chunks[column]))
}

/// A sum of a group's products, exact in single precision: at most 64 products
/// of 255 by 128, below 2^24.
#[inline]
#[expect(clippy::as_conversions, reason = "a sum below 2^24 converts exactly")]
#[expect(clippy::cast_precision_loss, reason = "a sum below 2^24 converts exactly")]
fn exact(sum: i32) -> f32 {
    sum as f32
}

/// An integer of a weight, rounded and within -127 to 127.
#[inline]
#[expect(clippy::as_conversions, reason = "an integer within -127 to 127 converts exactly")]
#[expect(
    clippy::cast_possible_truncation,
    reason = "an integer within -127 to 127 converts exactly"
)]
fn small(value: f32) -> i8 {
    value as i8
}
