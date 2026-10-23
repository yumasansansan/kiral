// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later
//
// This file is part of KIRAL. The SPDX lines name its copyright holder and its
// license, in the machine-readable form of the REUSE specification: the GNU
// General Public License, version 3 or any later version
// (LICENSES/GPL-3.0-or-later.txt).

//! Activations in single precision: GELU and softmax, and the exponential they
//! are computed with.
//!
//! The exponential is computed here rather than taken from the system's
//! library a value at a time, so that the compiler makes vector code of the
//! loops that take it. Taken from the library, GELU's hyperbolic tangent was
//! 16% of a key's refinement by zenz in integers on a P-core of an i7-1360P,
//! 3.0 of 19 ms on average (pinned, 4.4 GHz at the median), and the softmax's
//! exponentials some 0.5 ms more: GELU of 20 rows of 3072 took 0.246 ms there,
//! and takes 0.070 ms so; the softmaxes of a key's 2880 heads over 71
//! positions 0.49 ms, and 0.29 ms so (the medians of repeated calls, 4.6 GHz).

use crate::{multiply_add, sum};

/// sqrt(2 / pi), rounded to single precision.
const SQRT_2_OVER_PI: f32 = 0.797_884_6;

/// The coefficient of the cube in GPT-2's approximation of GELU.
const CUBE: f32 = 0.044_715;

/// 1.5 times 2^23: a float below 2^22 in magnitude added to it leaves its
/// nearest integer, ties to even, in the sum's lowest bits.
const ROUNDER: f32 = 12_582_912.0;

/// ln 2 in two parts, Cody and Waite's: the first of nine bits, so that its
/// product with an integer of eight is exact, and what it leaves of ln 2.
const LN_2_HIGH: f32 = 355.0 / 512.0;
const LN_2_LOW: f32 = -2.121_944_4e-4;

/// The largest argument taken: its exponential is below the largest float.
const HIGHEST: f32 = 88.376_26;

/// The least argument whose exponential is not taken as 0: its exponential is
/// the least normal float.
const LOWEST: f32 = -87.336_55;

/// The coefficients of `(e^r - 1 - r) / r^2` over `|r| <= ln 2 / 2`, from the
/// highest power's down: those of Cephes' `expf` (S. L. Moshier).
const POLYNOMIAL: [f32; 6] = [
    1.987_569_1e-4,
    1.398_199_9e-3,
    8.333_452e-3,
    4.166_579_6e-2,
    0.166_666_66,
    0.5,
];

/// The exponential of `x`, within 1.42 u of it relatively (u = 2^-24): `x`
/// less its nearest multiple of ln 2, `n ln 2`, in Cody and Waite's two parts;
/// the exponential of what is left, `r`, as `1 + r + r^2 p(r)` by Cephes'
/// polynomial; times `2^n`, built in the exponent's bits. Measured against
/// double precision on every float from -87.34 to 88.38, its error is at most
/// 1.37 u, and 1.42 u where the multiply-adds are fused. An argument above
/// 88.38 is taken as 88.38; one below -87.34, whose exponential is below the
/// least normal float, gives 0.
#[inline]
#[must_use]
pub fn exp(x: f32) -> f32 {
    let clamped = x.clamp(LOWEST, HIGHEST);
    let shifted = multiply_add(clamped, std::f32::consts::LOG2_E, ROUNDER);
    let multiple = shifted - ROUNDER;
    let rest = multiply_add(multiple, -LN_2_HIGH, clamped);
    let rest = multiply_add(multiple, -LN_2_LOW, rest);
    let polynomial = POLYNOMIAL[1..]
        .iter()
        .fold(POLYNOMIAL[0], |sum, &coefficient| multiply_add(sum, rest, coefficient));
    let exponential = multiply_add(polynomial, rest * rest, rest) + 1.0;
    // 2^n: n + 127 in the exponent's bits, n the lowest bits of the shifted sum
    // less those of the rounder, from -126 to 127.
    let power = f32::from_bits(shifted.to_bits().wrapping_sub(ROUNDER.to_bits()).wrapping_add(127) << 23);
    if x < LOWEST { 0.0 } else { exponential * power }
}

/// GELU as GPT-2 computes it (`gelu_new`), in place:
/// `x / 2 (1 + tanh(sqrt(2 / pi) (x + 0.044715 x^3)))`, as
/// `x / (1 + exp(-2 sqrt(2 / pi) (x + 0.044715 x^3)))`, which it equals, since
/// `(1 + tanh z) / 2 = 1 / (1 + e^-2z)`.
pub fn gelu_tanh(values: &mut [f32]) {
    for value in values {
        let x = *value;
        let inner = SQRT_2_OVER_PI * multiply_add(CUBE, x * x * x, x);
        *value = x / (1.0 + exp(-2.0 * inner));
    }
}

/// The softmax of the values, in place: `exp(x - max) / sum(exp(x - max))`.
/// Nothing is done to no values.
pub fn softmax(values: &mut [f32]) {
    let Some(largest) = values.iter().copied().reduce(f32::max) else {
        return;
    };
    for value in values.iter_mut() {
        *value = exp(*value - largest);
    }
    let scale = 1.0 / sum(values);
    for value in values.iter_mut() {
        *value *= scale;
    }
}
