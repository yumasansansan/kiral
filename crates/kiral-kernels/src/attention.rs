// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later
//
// This file is part of KIRAL. The SPDX lines name its copyright holder and its
// license, in the machine-readable form of the REUSE specification: the GNU
// General Public License, version 3 or any later version
// (LICENSES/GPL-3.0-or-later.txt).

//! Attention in single precision, over the cache of keys and values.

use crate::activation::softmax;
use crate::multiply_add;
use crate::vector::dot;

/// The attention of one head for one query over the positions of its cache:
/// `output = softmax(keys query scale) values`.
///
/// `keys` and `values` hold a row of the query's length for each position the
/// query attends to, a head's rows one after another; `scores` is a value for
/// each position, which the softmax is computed in. The output is the values'
/// rows added in the order of the positions, each times its weight.
///
/// # Panics
///
/// Where the lengths do not fit together so.
pub fn attention(query: &[f32], keys: &[f32], values: &[f32], scale: f32, scores: &mut [f32], output: &mut [f32]) {
    let width = query.len();
    assert!(width > 0, "an attention of empty rows");
    assert_eq!(output.len(), width, "an output of another length than the query");
    assert_eq!(
        keys.len(),
        scores.len() * width,
        "keys of other positions than the scores"
    );
    assert_eq!(values.len(), keys.len(), "values of other positions than the keys");
    for (score, key) in scores.iter_mut().zip(keys.chunks_exact(width)) {
        *score = dot(query, key) * scale;
    }
    softmax(scores);
    output.fill(0.0);
    for (&weight, value) in scores.iter().zip(values.chunks_exact(width)) {
        for (output, &value) in output.iter_mut().zip(value) {
            *output = multiply_add(weight, value, *output);
        }
    }
}
