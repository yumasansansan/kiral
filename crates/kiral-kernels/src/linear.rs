// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later
//
// This file is part of KIRAL. The SPDX lines name its copyright holder and its
// license, in the machine-readable form of the REUSE specification: the GNU
// General Public License, version 3 or any later version
// (LICENSES/GPL-3.0-or-later.txt).

//! The linear layers in single precision: `output = input weight + bias`.
//!
//! The output is computed in tiles of [`ROWS`] rows by [`COLUMNS`] columns, whose
//! sums stay in registers while the products of the whole depth are added to
//! them: each step takes one row of the tile's columns of the weights and
//! multiplies it by one value of each row of the input. The weights are packed
//! for it once ([`Packed`]): the columns in panels of a tile's width, each
//! panel's rows one after another, so that a tile reads its weights in order
//! rather than a row of the layer apart, across as many pages as the layer has
//! rows. The tiles of a panel are taken one after another down the rows of the
//! input, while the panel stays in the caches, the rows the tiles leave as one
//! tile of fewer rows. A single row -- a step of decoding -- takes four panels at
//! once instead, to keep as many sums in registers as a tile of rows does.

use crate::multiply_add;

/// The rows of a tile: twelve registers of sums, with two vectors of columns
/// each, of the sixteen of SSE and AVX (32 of AVX-512); 24 of NEON's 32, with
/// four.
pub const ROWS: usize = 6;

/// The columns of a tile: two vectors of the build's widest registers for each
/// row -- of AVX-512, AVX or SSE -- and four of NEON, which has 32 of them.
pub const COLUMNS: usize = if cfg!(target_feature = "avx512f") {
    32
} else if cfg!(any(target_feature = "avx", target_arch = "aarch64")) {
    16
} else {
    8
};

/// A linear layer's weights, packed for [`linear`].
#[derive(Debug, Clone, PartialEq)]
pub struct Packed {
    depth: usize,
    width: usize,
    /// The whole panels of [`COLUMNS`] columns, each its `depth` rows one after
    /// another.
    panels: Vec<f32>,
    /// The columns after the last whole panel, row by row.
    rest: Vec<f32>,
}

impl Packed {
    /// The weights of a layer of `depth` inputs, `weight` holding `depth` rows
    /// of `width` values: the layout of GPT-2's linear layers, inputs by outputs.
    ///
    /// # Panics
    ///
    /// Where `weight` is not `depth` rows of `width` values, or either is 0.
    #[must_use]
    pub fn new(weight: &[f32], depth: usize, width: usize) -> Self {
        assert!(
            depth > 0 && width > 0,
            "a linear layer of {depth} inputs and {width} outputs"
        );
        assert_eq!(
            weight.len(),
            depth * width,
            "weights that are not {depth} rows of {width}"
        );
        let tiled = width - width % COLUMNS;
        let mut panels = Vec::with_capacity(depth * tiled);
        for column in (0..tiled).step_by(COLUMNS) {
            for row in weight.chunks_exact(width) {
                panels.extend_from_slice(&row[column..][..COLUMNS]);
            }
        }
        let rest = weight
            .chunks_exact(width)
            .flat_map(|row| &row[tiled..])
            .copied()
            .collect();
        Self {
            depth,
            width,
            panels,
            rest,
        }
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
}

/// `output = input weight + bias`, row by row.
///
/// `input` is rows of the layer's depth, `bias` a value for each of its
/// outputs, and `output` a row of them for each row of `input`.
///
/// # Panics
///
/// Where the lengths do not fit the layer's.
pub fn linear(input: &[f32], weight: &Packed, bias: &[f32], output: &mut [f32]) {
    let Packed {
        depth,
        width,
        ref panels,
        ref rest,
    } = *weight;
    assert_eq!(bias.len(), width, "a bias of another width than the layer's {width}");
    assert_eq!(input.len() % depth, 0, "an input that is not whole rows of {depth}");
    let rows = input.len() / depth;
    assert_eq!(
        output.len(),
        rows * width,
        "an output that is not {rows} rows of {width}"
    );
    let panel_length = depth * COLUMNS;
    if rows == 1 {
        // A single row has as many sums in registers as a block of rows when it
        // takes several panels at once.
        let groups = panels.len() / (PANELS * panel_length);
        for group in 0..groups {
            let column = group * PANELS * COLUMNS;
            let sums = wide::<PANELS>(
                input,
                &panels[group * PANELS * panel_length..][..PANELS * panel_length],
                &bias[column..][..PANELS * COLUMNS],
            );
            for (p, sums) in sums.iter().enumerate() {
                output[column + p * COLUMNS..][..COLUMNS].copy_from_slice(sums);
            }
        }
        for (index, panel) in panels.chunks_exact(panel_length).enumerate().skip(groups * PANELS) {
            let column = index * COLUMNS;
            let bias: &[f32; COLUMNS] = bias[column..][..COLUMNS]
                .try_into()
                .expect("a panel's columns of the bias");
            let [sums] = tile::<1>(input, panel, bias);
            output[column..][..COLUMNS].copy_from_slice(&sums);
        }
    } else {
        // Blocks of rows, and the rows left as one block more: a panel at a time,
        // down the blocks while it stays in the caches.
        let blocked = rows - rows % ROWS;
        for (index, panel) in panels.chunks_exact(panel_length).enumerate() {
            let column = index * COLUMNS;
            let bias: &[f32; COLUMNS] = bias[column..][..COLUMNS]
                .try_into()
                .expect("a panel's columns of the bias");
            for row in (0..blocked).step_by(ROWS) {
                let sums = tile::<ROWS>(&input[row * depth..][..ROWS * depth], panel, bias);
                store(output, width, row, column, &sums);
            }
            let left = &input[blocked * depth..];
            // An arm for each number of rows a block can leave.
            const { assert!(ROWS == 6) };
            match rows - blocked {
                0 => {}
                1 => store(output, width, blocked, column, &tile::<1>(left, panel, bias)),
                2 => store(output, width, blocked, column, &tile::<2>(left, panel, bias)),
                3 => store(output, width, blocked, column, &tile::<3>(left, panel, bias)),
                4 => store(output, width, blocked, column, &tile::<4>(left, panel, bias)),
                _ => store(output, width, blocked, column, &tile::<5>(left, panel, bias)),
            }
        }
    }
    let tiled = width - width % COLUMNS;
    let left = width - tiled;
    for (input, output) in input.chunks_exact(depth).zip(output.chunks_exact_mut(width)) {
        for (column, (output, &bias)) in output[tiled..].iter_mut().zip(&bias[tiled..]).enumerate() {
            let weights = rest.iter().skip(column).step_by(left);
            *output = input
                .iter()
                .zip(weights)
                .fold(bias, |sum, (&x, &w)| multiply_add(x, w, sum));
        }
    }
}

/// The panels a single row takes at once.
const PANELS: usize = 4;

/// Stores a tile's sums in the output, from row `row` and column `column`.
#[inline]
fn store<const R: usize>(output: &mut [f32], width: usize, row: usize, column: usize, sums: &[[f32; COLUMNS]; R]) {
    for (r, sums) in sums.iter().enumerate() {
        output[(row + r) * width + column..][..COLUMNS].copy_from_slice(sums);
    }
}

/// The sums of `R` rows of the input, each of the depth, with a panel: the
/// tile's values of the output.
#[inline]
fn tile<const R: usize>(input: &[f32], panel: &[f32], bias: &[f32; COLUMNS]) -> [[f32; COLUMNS]; R] {
    let depth = input.len() / R;
    let inputs: [&[f32]; R] = std::array::from_fn(|r| &input[r * depth..][..depth]);
    let mut sums = [*bias; R];
    for (k, weights) in panel.as_chunks::<COLUMNS>().0.iter().enumerate() {
        for (sums, input) in sums.iter_mut().zip(&inputs) {
            let value = input[k];
            for (sum, &weight) in sums.iter_mut().zip(weights) {
                *sum = multiply_add(value, weight, *sum);
            }
        }
    }
    sums
}

/// The sums of one row of the input with `P` panels side by side, each read in
/// order: the row's values of `P` panels' columns.
#[inline]
fn wide<const P: usize>(input: &[f32], panels: &[f32], bias: &[f32]) -> [[f32; COLUMNS]; P] {
    let depth = input.len();
    let panels: [&[[f32; COLUMNS]]; P] = std::array::from_fn(|p| {
        panels[p * depth * COLUMNS..][..depth * COLUMNS]
            .as_chunks::<COLUMNS>()
            .0
    });
    let mut sums: [[f32; COLUMNS]; P] = std::array::from_fn(|p| {
        bias[p * COLUMNS..][..COLUMNS]
            .try_into()
            .expect("a panel's columns of the bias")
    });
    for (k, &value) in input.iter().enumerate() {
        for (sums, panel) in sums.iter_mut().zip(&panels) {
            for (sum, &weight) in sums.iter_mut().zip(&panel[k]) {
                *sum = multiply_add(value, weight, *sum);
            }
        }
    }
    sums
}
