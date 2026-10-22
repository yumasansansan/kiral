// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later
//
// This file is part of KIRAL. The SPDX lines name its copyright holder and its
// license, in the machine-readable form of the REUSE specification: the GNU
// General Public License, version 3 or any later version
// (LICENSES/GPL-3.0-or-later.txt).

//! Vectors in single precision: dot products, sums, logits and the largest
//! value.

use crate::{LANES, multiply_add, sum_lanes};

/// The dot product of two vectors of one length: the products added in
/// [`LANES`] lanes, the lanes summed in a tree, then the products the lanes
/// leave over added one after another.
///
/// # Panics
///
/// Where the vectors' lengths differ.
#[must_use]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "a dot product of vectors of two lengths");
    let (a_chunks, a_rest) = a.as_chunks::<LANES>();
    let (b_chunks, b_rest) = b.as_chunks::<LANES>();
    let mut lanes = [0.0f32; LANES];
    for (a, b) in a_chunks.iter().zip(b_chunks) {
        for ((lane, &a), &b) in lanes.iter_mut().zip(a).zip(b) {
            *lane = multiply_add(a, b, *lane);
        }
    }
    let mut sum = sum_lanes(lanes);
    for (&a, &b) in a_rest.iter().zip(b_rest) {
        sum = multiply_add(a, b, sum);
    }
    sum
}

/// `x += y`, value by value: the residual stream.
///
/// # Panics
///
/// Where the vectors' lengths differ.
pub fn add(x: &mut [f32], y: &[f32]) {
    assert_eq!(x.len(), y.len(), "a sum of vectors of two lengths");
    for (x, &y) in x.iter_mut().zip(y) {
        *x += y;
    }
}

/// The logits of a hidden state: its dot product with each row of the
/// embedding, `output[v] = dot(hidden, embedding[v])`.
///
/// # Panics
///
/// Where the embedding is not a row of the hidden state's length for each
/// logit.
pub fn logits(hidden: &[f32], embedding: &[f32], output: &mut [f32]) {
    let width = hidden.len();
    assert!(width > 0, "logits of an empty hidden state");
    assert_eq!(
        embedding.len(),
        output.len() * width,
        "an embedding of other rows than the logits"
    );
    for (logit, row) in output.iter_mut().zip(embedding.chunks_exact(width)) {
        *logit = dot(hidden, row);
    }
}

/// The index of the largest value, the first of those equal to it: none where
/// there are no values but NaNs, which are never the largest.
#[must_use]
pub fn argmax(values: &[f32]) -> Option<usize> {
    let mut best: Option<(usize, f32)> = None;
    for (index, &value) in values.iter().enumerate() {
        if !value.is_nan() && best.is_none_or(|(_, largest)| value > largest) {
            best = Some((index, value));
        }
    }
    best.map(|(index, _)| index)
}
