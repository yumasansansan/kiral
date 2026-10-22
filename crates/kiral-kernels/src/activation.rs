// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later
//
// This file is part of KIRAL. The SPDX lines name its copyright holder and its
// license, in the machine-readable form of the REUSE specification: the GNU
// General Public License, version 3 or any later version
// (LICENSES/GPL-3.0-or-later.txt).

//! Activations in single precision: GELU and softmax.
//!
//! The hyperbolic tangent and the exponential are the system library's, one
//! value at a time: the reference computes them as precisely as the library
//! does, and the time they take is small beside the products'.

use crate::sum;

/// sqrt(2 / pi), rounded to single precision.
const SQRT_2_OVER_PI: f32 = 0.797_884_6;

/// The coefficient of the cube in GPT-2's approximation of GELU.
const CUBE: f32 = 0.044_715;

/// GELU as GPT-2 computes it (`gelu_new`), in place:
/// `x / 2 (1 + tanh(sqrt(2 / pi) (x + 0.044715 x^3)))`.
pub fn gelu_tanh(values: &mut [f32]) {
    for value in values {
        let x = *value;
        let inner = SQRT_2_OVER_PI * (x + CUBE * (x * x * x));
        *value = 0.5 * x * (1.0 + inner.tanh());
    }
}

/// The softmax of the values, in place: `exp(x - max) / sum(exp(x - max))`.
/// Nothing is done to no values.
pub fn softmax(values: &mut [f32]) {
    let Some(largest) = values.iter().copied().reduce(f32::max) else {
        return;
    };
    for value in values.iter_mut() {
        *value = (*value - largest).exp();
    }
    let scale = 1.0 / sum(values);
    for value in values.iter_mut() {
        *value *= scale;
    }
}
