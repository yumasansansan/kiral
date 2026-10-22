// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later
//
// This file is part of KIRAL. The SPDX lines name its copyright holder and its
// license, in the machine-readable form of the REUSE specification: the GNU
// General Public License, version 3 or any later version
// (LICENSES/GPL-3.0-or-later.txt).

//! The kernels in single precision against the same computations in double
//! precision, from the same inputs. Each value's error is held to what the
//! analysis of its roundings allows, as a multiple of the unit roundoff `u` of
//! single precision: for a sum, `gamma(n) = n u / (1 - n u)` times the sum of
//! the magnitudes it adds up, `n` the most roundings any term goes through. The
//! lengths are chosen not to be multiples of the vectors' widths, nor of the
//! linear layers' tiles.

use kiral_kernels::activation::{gelu_tanh, softmax};
use kiral_kernels::attention::attention;
use kiral_kernels::linear::{COLUMNS, Packed, ROWS, linear};
use kiral_kernels::norm::layer_norm;
use kiral_kernels::vector::{add, argmax, dot, logits};
use kiral_kernels::{LANES, sum};

/// The unit roundoff of single precision, 2^-24.
const U: f64 = 1.0 / 16_777_216.0;

/// `n u / (1 - n u)`: the bound of the relative error of `n` roundings.
fn gamma(n: usize) -> f64 {
    let nu = f64::from(u32::try_from(n).expect("a small count")) * U;
    nu / (1.0 - nu)
}

/// A double that holds a value of single precision, as single precision.
#[expect(clippy::as_conversions, reason = "the values given are exact in single precision")]
#[expect(
    clippy::cast_possible_truncation,
    reason = "the values given are exact in single precision"
)]
fn single(value: f64) -> f32 {
    let narrowed = value as f32;
    assert_eq!(f64::from(narrowed), value, "{value} is not a value of single precision");
    narrowed
}

/// A double rounded to the nearest value of single precision.
#[expect(
    clippy::as_conversions,
    reason = "the rounding to nearest that the test compares with"
)]
#[expect(
    clippy::cast_possible_truncation,
    reason = "the rounding to nearest that the test compares with"
)]
fn rounded(value: f64) -> f32 {
    value as f32
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

    /// A value in [-1, 1) of 24 significant bits, exact in single precision.
    fn unit(&mut self) -> f32 {
        let bits = f64::from(u32::try_from(self.next() >> 40).expect("24 bits"));
        single(bits / 8_388_608.0 - 1.0)
    }

    /// `count` values in [-scale, scale), `scale` a power of two.
    fn values(&mut self, count: usize, scale: f32) -> Vec<f32> {
        (0..count).map(|_| self.unit() * scale).collect()
    }
}

fn wide(values: &[f32]) -> Vec<f64> {
    values.iter().copied().map(f64::from).collect()
}

#[test]
fn linear_layers_are_within_their_bound() {
    let mut random = Random(1);
    let depths = [1, 3, 16, 17, 64, 77];
    let widths = [
        1,
        7,
        COLUMNS - 1,
        COLUMNS,
        COLUMNS + 1,
        2 * COLUMNS + 3,
        4 * COLUMNS,
        9 * COLUMNS + 5,
        61,
    ];
    let rows = [1, 2, ROWS - 1, ROWS, ROWS + 1, 2 * ROWS + 3];
    for depth in depths {
        for width in widths {
            for rows in rows {
                let input = random.values(rows * depth, 1.0);
                let weight = random.values(depth * width, 1.0);
                let bias = random.values(width, 1.0);
                let packed = Packed::new(&weight, depth, width);
                assert_eq!((packed.depth(), packed.width()), (depth, width));
                let mut output = vec![0.0f32; rows * width];
                linear(&input, &packed, &bias, &mut output);
                // The bias and `depth` products: in whatever order they are added, no
                // term goes through more than `depth + 1` roundings (a multiply-add's
                // one, or a product's and a sum's without fused ones).
                let bound = gamma(depth + 1);
                for row in 0..rows {
                    for column in 0..width {
                        let mut exact = f64::from(bias[column]);
                        let mut magnitude = exact.abs();
                        for k in 0..depth {
                            let product = f64::from(input[row * depth + k]) * f64::from(weight[k * width + column]);
                            exact += product;
                            magnitude += product.abs();
                        }
                        let error = (f64::from(output[row * width + column]) - exact).abs();
                        assert!(
                            error <= bound * magnitude,
                            "depth {depth}, width {width}, rows {rows}: ({row}, {column}) is {} for {exact}",
                            output[row * width + column]
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn dot_products_and_sums_are_within_their_bound() {
    let mut random = Random(2);
    let lengths = (0..=3 * LANES + 5).chain([255, 256, 257, 1000, 3072]);
    for length in lengths {
        let a = random.values(length, 1.0);
        let b = random.values(length, 1.0);
        let (exact, magnitude) = a.iter().zip(&b).fold((0.0f64, 0.0f64), |(sum, magnitude), (&a, &b)| {
            let product = f64::from(a) * f64::from(b);
            (sum + product, magnitude + product.abs())
        });
        // No term goes through more roundings than there are terms: the lanes' own,
        // the four of their tree, and the leftovers' (fewer than a chunk).
        let error = (f64::from(dot(&a, &b)) - exact).abs();
        assert!(error <= gamma(length) * magnitude, "a dot product of {length}");
        let exact: f64 = wide(&a).iter().sum();
        let magnitude: f64 = wide(&a).iter().map(|value| value.abs()).sum();
        let error = (f64::from(sum(&a)) - exact).abs();
        assert!(error <= gamma(length) * magnitude, "a sum of {length}");
    }
}

#[test]
fn additions_are_rounded_once() {
    let mut random = Random(3);
    for length in [0, 1, 15, 16, 17, 100] {
        let mut x = random.values(length, 1.0);
        let y = random.values(length, 0.25);
        // Two values of single precision within a factor of 2^24 add exactly in
        // double precision; rounded to single, the sum is the one rounding.
        let expected = x.iter().zip(&y).map(|(&x, &y)| rounded(f64::from(x) + f64::from(y)));
        let expected = expected.collect::<Vec<_>>();
        add(&mut x, &y);
        assert_eq!(x, expected);
    }
}

#[test]
fn the_largest_value_is_the_first_of_equals_and_never_nan() {
    assert_eq!(argmax(&[]), None);
    assert_eq!(argmax(&[f32::NAN, f32::NAN]), None);
    assert_eq!(argmax(&[1.0, 3.0, 3.0, 2.0]), Some(1));
    assert_eq!(argmax(&[f32::NAN, -1.0, f32::NAN, -0.5]), Some(3));
    assert_eq!(argmax(&[f32::NEG_INFINITY, f32::NEG_INFINITY]), Some(0));
    assert_eq!(argmax(&[-2.0, f32::INFINITY, f32::NAN]), Some(1));
}

#[test]
fn logits_are_dot_products_within_their_bound() {
    let mut random = Random(4);
    for (width, vocabulary) in [(1, 3), (17, 5), (64, 33), (768, 7)] {
        let hidden = random.values(width, 1.0);
        let embedding = random.values(width * vocabulary, 1.0);
        let mut output = vec![0.0f32; vocabulary];
        logits(&hidden, &embedding, &mut output);
        for (v, &logit) in output.iter().enumerate() {
            let row = &embedding[v * width..][..width];
            let (exact, magnitude) = hidden
                .iter()
                .zip(row)
                .fold((0.0f64, 0.0f64), |(sum, magnitude), (&h, &e)| {
                    let product = f64::from(h) * f64::from(e);
                    (sum + product, magnitude + product.abs())
                });
            assert!((f64::from(logit) - exact).abs() <= gamma(width) * magnitude);
        }
    }
}

/// The bound of the error of each value of a normalized row.
///
/// With `m` the mean of the magnitudes, the rounded mean is within
/// `D = (gamma(d) + u) m` of the exact one; the squared deviations from it sum to
/// `d (variance + delta^2)`, within `gamma(d + 3)`, and the variance is divided
/// out with one more rounding. The scale `1 / sqrt(variance + epsilon)` then
/// takes three roundings of its own and half the relative error of what it is
/// the root of, besides the factor `sqrt((s2 + e) / (s2 + delta^2 + e))` that
/// the mean's error makes it smaller by. So a normalized value
/// `(x - mean) scale` is within `|x - mean| scale (gamma(d + 4) / 2 + 6 u +
/// D^2 / (2 (s2 + e))) + D scale (1 + 2 u)` of the exact one, and the gain and
/// bias add at most two roundings of the result.
fn normalization_bound(x: &[f64], gain: &[f64], bias: &[f64], epsilon: f64) -> Vec<f64> {
    let d = x.len();
    let count = f64::from(u32::try_from(d).expect("a small width"));
    let mean = x.iter().sum::<f64>() / count;
    let variance = x.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / count;
    let magnitude = x.iter().map(|x| x.abs()).sum::<f64>() / count;
    let delta = (gamma(d) + U) * magnitude;
    let scale = 1.0 / (variance + epsilon).sqrt();
    let relative = gamma(d + 4) / 2.0 + 6.0 * U + delta * delta / (2.0 * (variance + epsilon));
    x.iter()
        .zip(gain)
        .zip(bias)
        .map(|((&x, &gain), &bias)| {
            let normalized = (x - mean) * scale;
            gain.abs() * ((x - mean).abs() * scale * relative + delta * scale * (1.0 + 2.0 * U))
                + 2.0 * U * ((normalized * gain).abs() + bias.abs())
        })
        .collect()
}

#[test]
fn layer_normalization_is_within_its_bound() {
    let mut random = Random(5);
    let epsilon = 1e-5f32;
    for width in [1, 2, 15, 16, 17, 33, 768] {
        for (offset, spread) in [(0.0, 1.0), (10.0, 1.0), (0.0, 0.0625)] {
            let rows = 3;
            let input = random
                .values(rows * width, spread)
                .into_iter()
                .map(|x| x + offset)
                .collect::<Vec<_>>();
            let gain = random.values(width, 2.0);
            let bias = random.values(width, 0.5);
            let mut output = vec![0.0f32; rows * width];
            layer_norm(&input, &gain, &bias, epsilon, &mut output);
            for row in 0..rows {
                let x = wide(&input[row * width..][..width]);
                let bounds = normalization_bound(&x, &wide(&gain), &wide(&bias), f64::from(epsilon));
                let count = f64::from(u32::try_from(width).expect("small"));
                let mean = x.iter().sum::<f64>() / count;
                let variance = x.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / count;
                let scale = 1.0 / (variance + f64::from(epsilon)).sqrt();
                for i in 0..width {
                    let exact = (x[i] - mean) * scale * f64::from(gain[i]) + f64::from(bias[i]);
                    let error = (f64::from(output[row * width + i]) - exact).abs();
                    assert!(
                        error <= bounds[i],
                        "width {width}, offset {offset}, spread {spread}: value {i} of row {row} off by {error}, \
                         beyond {}",
                        bounds[i]
                    );
                }
            }
        }
    }
}

#[test]
fn a_constant_row_normalizes_to_its_bias() {
    let gain = [1.5f32, -2.0, 0.25, 3.0, 1.0];
    let bias = [0.5f32, 0.25, -1.0, 0.0, 2.0];
    let mut output = [0.0f32; 5];
    layer_norm(&[0.75; 5], &gain, &bias, 1e-5, &mut output);
    assert_eq!(output, bias);
}

#[test]
fn gelu_is_within_its_bound() {
    let mut random = Random(6);
    let mut values = random.values(997, 8.0);
    values.extend([0.0, 1e-3, -1e-3, 0.5, -0.5, 3.0, -3.0, 40.0, -40.0]);
    let mut output = values.clone();
    gelu_tanh(&mut output);
    for (&x, &y) in values.iter().zip(&output) {
        let x = f64::from(x);
        let inner = (2.0 / std::f64::consts::PI).sqrt() * (x + 0.044_715 * x * x * x);
        let exact = 0.5 * x * (1.0 + inner.tanh());
        // The argument of tanh is within 5 u of its magnitude (five roundings, its
        // terms of one sign), and tanh's slope 1 - tanh^2 times it is at most
        // 0.45 times that, since z sech^2(z) <= 0.45; tanh adds its own error, a few
        // units in the last place, 1 + t one rounding, and the product with x / 2
        // one more: within 8 u |x| allows the library's tanh 9 units.
        let error = (f64::from(y) - exact).abs();
        assert!(error <= 8.0 * U * x.abs(), "GELU({x}) is {y}, {exact} exactly");
    }
}

/// The bound of the relative error of each weight of a softmax: each
/// exponential's argument is rounded once, which moves the exponential by its
/// magnitude times `u`, and the library's exponential adds up to two units in
/// the last place (`4 u`); the sum adds `gamma(n)`, and the division and the
/// product two roundings. A weight is off by its own exponential's error and at
/// most the largest of the others'.
fn softmax_bounds(x: &[f64]) -> Vec<f64> {
    let largest = x.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let exponential = |x: f64| U * ((x - largest).abs() + 4.0);
    let worst = x.iter().copied().map(exponential).fold(0.0, f64::max);
    let sum = gamma(x.len().max(1));
    x.iter().map(|&x| exponential(x) + worst + sum + 2.0 * U).collect()
}

fn exact_softmax(x: &[f64]) -> Vec<f64> {
    let largest = x.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let exponentials = x.iter().map(|x| (x - largest).exp()).collect::<Vec<_>>();
    let total: f64 = exponentials.iter().sum();
    exponentials.iter().map(|e| e / total).collect()
}

#[test]
fn softmax_is_within_its_bound() {
    let mut random = Random(7);
    softmax(&mut []);
    for length in [1, 2, 7, 16, 17, 100, 257] {
        let values = random.values(length, 8.0);
        let mut output = values.clone();
        softmax(&mut output);
        let x = wide(&values);
        let exact = exact_softmax(&x);
        for ((&y, exact), bound) in output.iter().zip(&exact).zip(softmax_bounds(&x)) {
            assert!(
                (f64::from(y) - exact).abs() <= bound * exact,
                "length {length}: {y} for {exact}"
            );
        }
    }
}

#[test]
fn attention_is_within_its_bound() {
    let mut random = Random(8);
    for (width, positions) in [(1, 1), (8, 3), (64, 1), (64, 17), (64, 200), (33, 9)] {
        let query = random.values(width, 1.0);
        let keys = random.values(positions * width, 1.0);
        let values = random.values(positions * width, 1.0);
        let scale = 0.125f32;
        let mut scores = vec![0.0f32; positions];
        let mut output = vec![0.0f32; width];
        attention(&query, &keys, &values, scale, &mut scores, &mut output);
        // Each score is a dot product, off by gamma(width) of its magnitudes times
        // the scale (a power of two, exact); a weight is off by its softmax's
        // relative error, and by its own score's error and at most the largest of
        // the others' as a relative error too; the output adds the weighted values
        // in `positions` multiply-adds.
        let q = wide(&query);
        let mut exact_scores = Vec::with_capacity(positions);
        let mut score_errors = Vec::with_capacity(positions);
        for key in keys.chunks_exact(width) {
            let k = wide(key);
            let product: f64 = q.iter().zip(&k).map(|(q, k)| q * k).sum();
            let magnitude: f64 = q.iter().zip(&k).map(|(q, k)| (q * k).abs()).sum();
            exact_scores.push(product * f64::from(scale));
            score_errors.push(gamma(width) * magnitude * f64::from(scale));
        }
        let weights = exact_softmax(&exact_scores);
        let worst_score = score_errors.iter().copied().fold(0.0, f64::max);
        let softmax_relative = softmax_bounds(&exact_scores);
        for i in 0..width {
            let mut exact = 0.0;
            let mut bound = 0.0;
            for j in 0..positions {
                let value = f64::from(values[j * width + i]);
                exact += weights[j] * value;
                let relative = softmax_relative[j] + score_errors[j] + worst_score + gamma(positions);
                bound += weights[j] * value.abs() * relative;
            }
            let error = (f64::from(output[i]) - exact).abs();
            assert!(
                error <= bound,
                "width {width}, positions {positions}: value {i} off by {error}, beyond {bound}"
            );
        }
    }
}
