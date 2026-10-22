// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later
//
// This file is part of KIRAL. The SPDX lines name its copyright holder and its
// license, in the machine-readable form of the REUSE specification: the GNU
// General Public License, version 3 or any later version
// (LICENSES/GPL-3.0-or-later.txt).

//! Layer normalization in single precision.

use crate::{LANES, length, multiply_add, sum, sum_lanes};

/// `output = (x - mean) / sqrt(variance + epsilon) gain + bias`, row by row:
/// the mean and the variance of each row of `input`, the variance the mean of
/// the squared deviations (the biased one, as layer normalization takes it),
/// each summed in lanes, the deviations from the mean as it was rounded.
///
/// # Panics
///
/// Where `gain` is empty or of another length than `bias`, or `input` and
/// `output` are not of one length of whole rows of it.
pub fn layer_norm(input: &[f32], gain: &[f32], bias: &[f32], epsilon: f32, output: &mut [f32]) {
    let width = gain.len();
    assert!(width > 0, "a normalization of empty rows");
    assert_eq!(bias.len(), width, "a gain and a bias of two lengths");
    assert_eq!(input.len() % width, 0, "an input that is not whole rows of {width}");
    assert_eq!(output.len(), input.len(), "an output of another length than the input");
    let count = length(width);
    for (x, y) in input.chunks_exact(width).zip(output.chunks_exact_mut(width)) {
        let mean = sum(x) / count;
        let variance = squared_deviations(x, mean) / count;
        let scale = 1.0 / (variance + epsilon).sqrt();
        for (((y, &x), &gain), &bias) in y.iter_mut().zip(x).zip(gain).zip(bias) {
            *y = multiply_add((x - mean) * scale, gain, bias);
        }
    }
}

/// The sum of the squares of the values' deviations from `mean`, in lanes.
fn squared_deviations(values: &[f32], mean: f32) -> f32 {
    let (chunks, rest) = values.as_chunks::<LANES>();
    let mut lanes = [0.0f32; LANES];
    for chunk in chunks {
        for (lane, &value) in lanes.iter_mut().zip(chunk) {
            let deviation = value - mean;
            *lane = multiply_add(deviation, deviation, *lane);
        }
    }
    let mut total = sum_lanes(lanes);
    for &value in rest {
        let deviation = value - mean;
        total = multiply_add(deviation, deviation, total);
    }
    total
}
