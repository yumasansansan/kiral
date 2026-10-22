// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later
//
// This file is part of KIRAL. The SPDX lines name its copyright holder and its
// license, in the machine-readable form of the REUSE specification: the GNU
// General Public License, version 3 or any later version
// (LICENSES/GPL-3.0-or-later.txt).

//! The linear layers in integers. The rotations are checked to be orthogonal,
//! regular and their own inverses; the quantization of the weights and of the
//! inputs against the rule it follows, each number within half a step of its
//! scale of what it stands for, read back from the packed bytes by the layout
//! the documentation gives; the products against double precision from the
//! same integers, scales and offsets, held to `gamma(n)` times the sum of the
//! magnitudes added, `n` the most roundings a term goes through; and against
//! the products of the numbers before they were quantized, held to what the
//! quantization's errors allow. Each under several schemes, at depths and
//! widths that are not multiples of a group, a panel or a rotation, and with
//! every number of rows a tile of each form may take; the products in every
//! form the processor runs. Packed weights read back from their bytes, as a
//! model file holds them, are the same weights.

use kiral_cpu::Features;
use kiral_kernels::Form;
use kiral_kernels::quantized::{
    GROUP, GROUP_BYTES, INPUT_MAX, PANEL, Packed, QUADS, Rotation, Rows, Scheme, linear, linear_in, rotate,
};

/// The unit roundoff of single precision, 2^-24.
const U: f64 = 1.0 / 16_777_216.0;

/// `n u / (1 - n u)`: the bound of the relative error of `n` roundings.
fn gamma(n: usize) -> f64 {
    let nu = f64::from(u32::try_from(n).expect("a small count")) * U;
    nu / (1.0 - nu)
}

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

    /// A value in [-1, 1) of 16 significant bits.
    fn unit(&mut self) -> f32 {
        let bits = u16::try_from(self.next() >> 48).expect("16 bits");
        f32::from(bits) / 32_768.0 - 1.0
    }

    fn below(&mut self, bound: usize) -> usize {
        usize::try_from(self.next() % u64::try_from(bound).expect("small")).expect("small")
    }

    /// Values of ranges of every kind: a stretch of them all positive, all
    /// negative, mixed, zeros, or one large among small ones.
    fn values(&mut self, count: usize) -> Vec<f32> {
        let mut values = Vec::with_capacity(count);
        while values.len() < count {
            let stretch = (1 + self.below(90)).min(count - values.len());
            let kind = self.below(5);
            for _ in 0..stretch {
                let unit = self.unit();
                values.push(match kind {
                    0 => unit.abs() * 4.0,
                    1 => -unit.abs() * 0.5,
                    2 => unit * 2.0,
                    3 => 0.0,
                    _ => unit * 0.01,
                });
            }
            if kind == 4 {
                let at = values.len() - 1 - self.below(stretch);
                values[at] = 30.0 * self.unit();
            }
        }
        values
    }
}

/// The shapes taken: depth, width and rows.
const SHAPES: [(usize, usize, usize); 7] = [
    (1, 1, 1),
    (63, 17, 1),
    (64, 16, 5),
    (65, 15, 2),
    (100, 40, 9),
    (200, 33, 4),
    (768, 48, 3),
];

/// The schemes taken.
const SCHEMES: [Scheme; 4] = [
    Scheme::INT8,
    Scheme {
        weight_bits: 4,
        rotation: Rotation::Hadamard(16),
        fitted: true,
    },
    Scheme {
        weight_bits: 8,
        rotation: Rotation::Hadamard(256),
        fitted: false,
    },
    Scheme {
        weight_bits: 3,
        rotation: Rotation::Hadamard(4),
        fitted: false,
    },
];

/// A weight read back from the packed bytes: its integer, its group's scale and
/// its group's offset, by the layout the documentation gives.
fn weight_of(packed: &Packed, k: usize, j: usize) -> (i8, f32, f32) {
    let groups = packed.depth().div_ceil(GROUP);
    let (panel, column) = (j / PANEL, j % PANEL);
    let (group, e) = (k / GROUP, k % GROUP);
    let base = (panel * groups + group) * GROUP_BYTES;
    let bytes = packed.bytes();
    let q = bytes[base + (e / 4 * PANEL + column) * 4 + e % 4].cast_signed();
    let float = |at: usize| f32::from_le_bytes(bytes[at..][..4].try_into().expect("four bytes"));
    let floats = base + QUADS * PANEL * 4;
    (q, float(floats + column * 4), float(floats + (PANEL + column) * 4))
}

/// Values rotated as a scheme rotates them.
fn rotated(values: &[f32], rotation: Rotation) -> Vec<f32> {
    let mut values = values.to_vec();
    if let Rotation::Hadamard(order) = rotation {
        rotate(&mut values, order);
    }
    values
}

#[test]
fn rotations_are_orthogonal_regular_and_their_own_inverses() {
    for order in [4usize, 16, 64, 256] {
        // The columns of the matrix, the rotations of the unit vectors.
        let columns: Vec<Vec<f32>> = (0..order)
            .map(|i| {
                let mut unit = vec![0.0f32; order];
                unit[i] = 1.0;
                rotate(&mut unit, order);
                unit
            })
            .collect();
        let root = f64::from(u32::try_from(order).expect("small")).sqrt();
        for (i, column) in columns.iter().enumerate() {
            // Each entry is plus or minus 1 / sqrt(order), and each column sums to
            // plus or minus 1: sqrt(order) entries of the Hadamard matrix's.
            for &entry in column {
                assert!((f64::from(entry).abs() * root - 1.0).abs() < 1e-6, "{order}: {entry}");
            }
            let sum: f64 = column.iter().copied().map(f64::from).sum();
            assert!((sum.abs() - 1.0).abs() < 1e-5, "order {order}, column {i}: {sum}");
            for (j, other) in columns.iter().enumerate() {
                let dot: f64 = column
                    .iter()
                    .zip(other)
                    .map(|(&a, &b)| f64::from(a) * f64::from(b))
                    .sum();
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (dot - expected).abs() < 1e-5,
                    "order {order}: columns {i} and {j}: {dot}"
                );
                // Symmetric.
                assert_eq!(column[j], other[i]);
            }
        }
    }
    // Twice is once undone, the values left after the whole blocks too.
    let mut random = Random(19);
    for (length, order) in [(100, 64), (7, 4), (300, 256), (1, 16)] {
        let values = random.values(length);
        let mut twice = values.clone();
        rotate(&mut twice, order);
        rotate(&mut twice, order);
        for (&value, &back) in values.iter().zip(&twice) {
            assert!((value - back).abs() <= 1e-5 * (1.0 + value.abs()), "{length}, {order}");
        }
    }
}

/// The squared error of a group of weights at a scale.
fn squared_error(values: &[f32], scale: f32, largest: f32) -> f64 {
    values
        .iter()
        .map(|&value| {
            let q = if scale > 0.0 {
                (value / scale).round_ties_even().clamp(-largest, largest)
            } else {
                0.0
            };
            (f64::from(value) - f64::from(scale) * f64::from(q)).powi(2)
        })
        .sum()
}

#[test]
fn weights_are_their_groups_steps() {
    let mut random = Random(17);
    for scheme in SCHEMES {
        let largest = scheme.weight_max();
        for (depth, width, _) in SHAPES {
            let weights = random.values(depth * width);
            let packed = Packed::quantize(depth, width, scheme, |k, j| weights[k * width + j]);
            assert_eq!(packed.scheme(), scheme);
            let groups = depth.div_ceil(GROUP);
            assert_eq!(packed.bytes().len(), width.div_ceil(PANEL) * groups * GROUP_BYTES);
            for j in 0..width {
                let column: Vec<f32> = (0..depth).map(|k| weights[k * width + j]).collect();
                let column = rotated(&column, scheme.rotation);
                for group in 0..groups {
                    let ks = group * GROUP..depth.min((group + 1) * GROUP);
                    let values = &column[ks.clone()];
                    let greatest = values.iter().fold(0.0f32, |greatest, value| greatest.max(value.abs()));
                    let (_, scale, offset) = weight_of(&packed, ks.start, j);
                    let context = format!("{scheme:?}, {depth}x{width}: column {j}, group {group}");
                    if scheme.fitted {
                        // Of least error among the scales tried, that of the
                        // greatest magnitude among them.
                        assert!(
                            scale <= greatest / largest && scale >= 0.49 * greatest / largest,
                            "{context}"
                        );
                        assert!(
                            squared_error(values, scale, largest)
                                <= squared_error(values, greatest / largest, largest) * (1.0 + 1e-6),
                            "{context}"
                        );
                    } else {
                        assert_eq!(scale, greatest / largest, "{context}");
                    }
                    let mut sum = 0i32;
                    for (k, &value) in ks.clone().zip(values) {
                        let (q, _, _) = weight_of(&packed, k, j);
                        assert!(f32::from(q).abs() <= largest, "{context}");
                        let error = (f64::from(scale) * f64::from(q) - f64::from(value)).abs();
                        if value.abs() <= scale * largest {
                            assert!(
                                error <= f64::from(scale) * (0.5 + 1e-5),
                                "{context}: weight {k} off by {error}"
                            );
                        } else {
                            assert_eq!(f32::from(q).abs(), largest, "{context}: weight {k} not clipped");
                        }
                        sum += i32::from(q);
                    }
                    assert_eq!(
                        offset,
                        scale * f32::from(i16::try_from(sum).expect("a sum of 64 weights")),
                        "{context}"
                    );
                    // The rest of the group stands for zeros.
                    for k in depth..(group + 1) * GROUP {
                        assert_eq!(weight_of(&packed, k, j).0, 0);
                    }
                }
            }
            // The columns past the width are zeros of no scale.
            for j in width..width.div_ceil(PANEL) * PANEL {
                for k in 0..depth {
                    assert_eq!(weight_of(&packed, k, j), (0, 0.0, 0.0));
                }
            }
        }
    }
}

#[test]
fn inputs_are_their_groups_steps() {
    let mut random = Random(23);
    let mut rows = Rows::new();
    assert!(rows.is_empty());
    for rotation in [Rotation::None, Rotation::Hadamard(64)] {
        for (depth, _, count) in SHAPES {
            let mut input = random.values(depth * count);
            // A row of zeros, which has no scale.
            if count > 1 {
                input[depth..2 * depth].fill(0.0);
            }
            rows.quantize(&input, depth, rotation);
            assert_eq!((rows.len(), rows.depth(), rows.rotation()), (count, depth, rotation));
            let groups = depth.div_ceil(GROUP);
            for (row, input) in input.chunks_exact(depth).enumerate() {
                let input = rotated(input, rotation);
                for group in 0..groups {
                    let values = &input[group * GROUP..depth.min((group + 1) * GROUP)];
                    let least = values.iter().copied().fold(0.0f32, f32::min);
                    let greatest = values.iter().copied().fold(0.0f32, f32::max);
                    let at = row * groups + group;
                    let scale = rows.scale(row, group);
                    assert_eq!(scale, (greatest - least) / INPUT_MAX);
                    // The zero point as the rule takes it: by the scale's inverse.
                    let zero = if scale > 0.0 {
                        (-least * (1.0 / scale)).round_ties_even()
                    } else {
                        0.0
                    };
                    assert_eq!(
                        rows.offset(row, group),
                        scale * zero,
                        "the offset of row {row}'s group {group}"
                    );
                    let bytes = &rows.bytes()[at * GROUP..][..GROUP];
                    for (e, &value) in values.iter().enumerate() {
                        let error =
                            (f64::from(scale) * (f64::from(bytes[e]) - f64::from(zero)) - f64::from(value)).abs();
                        assert!(
                            error <= f64::from(scale) * (0.5 + 1e-4),
                            "{rotation:?}, {depth}: row {row}'s input {} off by {error} of a step {scale}",
                            group * GROUP + e
                        );
                    }
                    // The rest of the group holds the zero point.
                    for &byte in &bytes[values.len()..] {
                        assert_eq!(f32::from(byte), zero);
                    }
                }
            }
        }
    }
}

/// The products in double precision from the integers, scales and offsets the
/// kernel takes, and the sum of the magnitudes added: `bias + sum over groups of
/// (sa sb sum(qa qb) - oa ob)`.
fn exact(rows: &Rows, packed: &Packed, bias: &[f32], row: usize, j: usize) -> (f64, f64) {
    let groups = packed.depth().div_ceil(GROUP);
    let mut value = f64::from(bias[j]);
    let mut magnitude = value.abs();
    for group in 0..groups {
        let at = row * groups + group;
        let mut products = 0i64;
        for e in 0..GROUP {
            let k = group * GROUP + e;
            let q = if k < packed.depth() {
                weight_of(packed, k, j).0
            } else {
                0
            };
            products += i64::from(rows.bytes()[at * GROUP + e]) * i64::from(q);
        }
        let (_, weight_scale, weight_offset) = weight_of(packed, group * GROUP, j);
        let products = f64::from(i32::try_from(products).expect("a group's products"));
        let scaled = f64::from(rows.scale(row, group)) * f64::from(weight_scale) * products;
        let offset = f64::from(rows.offset(row, group)) * f64::from(weight_offset);
        value += scaled - offset;
        magnitude += scaled.abs() + offset.abs();
    }
    (value, magnitude)
}

/// The forms of the kernels the processor runs, the portable first.
fn forms() -> Vec<Form> {
    let forms = Form::available(Features::detect());
    assert_eq!(forms.first(), Some(&Form::Portable));
    forms
}

#[test]
fn products_are_those_of_double_precision() {
    let mut random = Random(29);
    let mut rows = Rows::new();
    for scheme in SCHEMES {
        for (depth, width, count) in SHAPES {
            let weights = random.values(depth * width);
            let packed = Packed::quantize(depth, width, scheme, |k, j| weights[k * width + j]);
            rows.quantize(&random.values(depth * count), depth, scheme.rotation);
            let bias = random.values(width);
            for form in forms() {
                let mut output = vec![f32::NAN; count * width];
                linear_in(form, &rows, &packed, &bias, &mut output);
                // A group's term goes through the product of the scales, its
                // product with the sum and its addition, each rounded, and two
                // more additions for each group after it; the bias through two
                // for each group.
                let n = 2 * depth.div_ceil(GROUP) + 3;
                for row in 0..count {
                    for j in 0..width {
                        let (value, magnitude) = exact(&rows, &packed, &bias, row, j);
                        let error = (f64::from(output[row * width + j]) - value).abs();
                        assert!(
                            error <= gamma(n) * magnitude,
                            "{form}, {scheme:?}, {depth}x{width}: row {row}, column {j} off by {error}, bound {}",
                            gamma(n) * magnitude
                        );
                    }
                }
            }
        }
    }
    rows.quantize(&[], 3, Rotation::None);
    assert!(rows.is_empty());
    let packed = Packed::quantize(3, 2, Scheme::INT8, |_, _| 1.0);
    for form in forms() {
        linear_in(form, &rows, &packed, &[0.0; 2], &mut []);
    }
    // The form in effect is one of them.
    assert!(forms().contains(&kiral_kernels::form()));
}

#[test]
fn every_number_of_rows_takes_its_tile() {
    // Up to two tiles of the largest, and every number of rows left after one.
    let mut random = Random(37);
    let (depth, width) = (130, 35);
    let weights = random.values(depth * width);
    let packed = Packed::quantize(depth, width, Scheme::INT8, |k, j| weights[k * width + j]);
    let bias = random.values(width);
    let mut rows = Rows::new();
    for count in 1..=13 {
        rows.quantize(&random.values(depth * count), depth, Rotation::None);
        for form in forms() {
            let mut output = vec![f32::NAN; count * width];
            linear_in(form, &rows, &packed, &bias, &mut output);
            let n = 2 * depth.div_ceil(GROUP) + 3;
            for row in 0..count {
                for j in 0..width {
                    let (value, magnitude) = exact(&rows, &packed, &bias, row, j);
                    let error = (f64::from(output[row * width + j]) - value).abs();
                    assert!(
                        error <= gamma(n) * magnitude,
                        "{form}, {count} rows: row {row}, column {j}"
                    );
                }
            }
        }
    }
}

#[test]
fn weights_read_where_they_lie_are_those_packed() {
    // The packed bytes, as a model file holds them, read back without a copy:
    // the same weights, and the same products in every form; bytes of another
    // length than the layer's panels, or a scheme out of range, refused.
    let mut random = Random(41);
    let mut rows = Rows::new();
    let (depth, width, count) = (100, 40, 9);
    for scheme in SCHEMES {
        let weights = random.values(depth * width);
        let packed = Packed::quantize(depth, width, scheme, |k, j| weights[k * width + j]);
        let bytes = packed.bytes();
        let read = Packed::from_bytes(depth, width, scheme, bytes).expect("the bytes of the layer's panels");
        assert_eq!(read, packed);
        rows.quantize(&random.values(depth * count), depth, scheme.rotation);
        let bias = random.values(width);
        for form in forms() {
            let mut packed_output = vec![0.0f32; count * width];
            let mut read_output = vec![f32::NAN; count * width];
            linear_in(form, &rows, &packed, &bias, &mut packed_output);
            linear_in(form, &rows, &read, &bias, &mut read_output);
            assert_eq!(read_output, packed_output, "{form}, {scheme:?}");
        }
        assert!(Packed::from_bytes(depth, width, scheme, &bytes[1..]).is_none());
        assert!(Packed::from_bytes(depth + GROUP, width, scheme, bytes).is_none());
        assert!(Packed::from_bytes(depth, width + PANEL, scheme, bytes).is_none());
        assert!(Packed::from_bytes(0, width, scheme, &[]).is_none());
        for refused in [
            Scheme {
                weight_bits: 9,
                ..scheme
            },
            Scheme {
                weight_bits: 1,
                ..scheme
            },
            Scheme {
                rotation: Rotation::Hadamard(8),
                ..scheme
            },
        ] {
            assert!(
                Packed::from_bytes(depth, width, refused, bytes).is_none(),
                "{refused:?}"
            );
        }
    }
}

#[test]
fn products_are_near_those_of_the_numbers_quantized() {
    let mut random = Random(31);
    let mut rows = Rows::new();
    for scheme in SCHEMES {
        for (depth, width, count) in SHAPES {
            let weights = random.values(depth * width);
            let input = random.values(depth * count);
            let packed = Packed::quantize(depth, width, scheme, |k, j| weights[k * width + j]);
            rows.quantize(&input, depth, scheme.rotation);
            let bias = random.values(width);
            let mut output = vec![0.0f32; count * width];
            linear(&rows, &packed, &bias, &mut output);
            for row in 0..count {
                let x = rotated(&input[row * depth..][..depth], scheme.rotation);
                for j in 0..width {
                    let column: Vec<f32> = (0..depth).map(|k| weights[k * width + j]).collect();
                    let w = rotated(&column, scheme.rotation);
                    // The rotation leaves the products as they were; each of the
                    // rotated ones is off by the input times the weight's error,
                    // and by at most half an input's step times the weight that
                    // stands in.
                    let (mut value, mut allowed, mut size) = (f64::from(bias[j]), 0.0, f64::from(bias[j]).abs());
                    for k in 0..depth {
                        let original = f64::from(input[row * depth + k]) * f64::from(weights[k * width + j]);
                        value += original;
                        size += original.abs();
                        let (q, weight_scale, _) = weight_of(&packed, k, j);
                        let standing = f64::from(weight_scale) * f64::from(q);
                        let input_scale = f64::from(rows.scale(row, k / GROUP));
                        allowed += f64::from(x[k]).abs() * (standing - f64::from(w[k])).abs()
                            + input_scale * 0.501 * standing.abs();
                    }
                    let error = (f64::from(output[row * width + j]) - value).abs();
                    assert!(
                        error <= allowed * 1.001 + 1e-5 * size,
                        "{scheme:?}, {depth}x{width}: row {row}, column {j} off by {error}, allowed {allowed}"
                    );
                }
            }
        }
    }
}
