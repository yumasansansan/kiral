// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later
//
// This file is part of KIRAL. The SPDX lines name its copyright holder and its
// license, in the machine-readable form of the REUSE specification: the GNU
// General Public License, version 3 or any later version
// (LICENSES/GPL-3.0-or-later.txt).

//! The kernels in assembly for AArch64 (`asm/aarch64/`), declared, and the safe
//! function that calls them: it checks, in every build, what its kernel reads
//! and writes and that the processor has the kernel's extensions, then makes the
//! call, the one unsafe operation of its block.
#![allow(
    unsafe_code,
    reason = "the calls into assembly, each the one operation of a block whose comment says why it is sound"
)]

use crate::quantized::{GROUP, GROUP_BYTES, PANEL};

/// A kernel of the products in integers for a number of rows
/// (`qgemm_dotprod.S`).
type Qgemm =
    unsafe extern "C" fn(groups: usize, a: *const u8, a_stride: usize, a_floats: *const f32, b: *const u8, c: *mut f32);

unsafe extern "C" {
    fn kiral_qgemm_dotprod_1(
        groups: usize,
        a: *const u8,
        a_stride: usize,
        a_floats: *const f32,
        b: *const u8,
        c: *mut f32,
    );
    fn kiral_qgemm_dotprod_2(
        groups: usize,
        a: *const u8,
        a_stride: usize,
        a_floats: *const f32,
        b: *const u8,
        c: *mut f32,
    );
    fn kiral_qgemm_dotprod_3(
        groups: usize,
        a: *const u8,
        a_stride: usize,
        a_floats: *const f32,
        b: *const u8,
        c: *mut f32,
    );
    fn kiral_qgemm_dotprod_4(
        groups: usize,
        a: *const u8,
        a_stride: usize,
        a_floats: *const f32,
        b: *const u8,
        c: *mut f32,
    );
}

/// The rows a tile of the kernels takes at most.
pub(crate) const TILE_ROWS: usize = 4;

/// The kernels by their number of rows, from one.
const DOTPROD: [Qgemm; TILE_ROWS] = [
    kiral_qgemm_dotprod_1,
    kiral_qgemm_dotprod_2,
    kiral_qgemm_dotprod_3,
    kiral_qgemm_dotprod_4,
];

/// Whether the processor has what the dot products' kernels run: NEON and
/// `FEAT_DotProd`, as std reports them (once, then from its cache).
fn has_dotprod() -> bool {
    std::arch::is_aarch64_feature_detected!("neon") && std::arch::is_aarch64_feature_detected!("dotprod")
}

/// Adds to the first `rows` rows of `tile` their products with `groups` groups
/// of a panel, by the dot products' kernel of that many rows: `a` the rows'
/// bytes, `groups` groups of 64 each, `a_floats` each group's scale and offset,
/// row by row, and `panel` the panel's groups, of [`GROUP_BYTES`].
///
/// # Panics
///
/// Where `rows` is not 1 to [`TILE_ROWS`], `groups` is 0, the slices are
/// shorter than the rows and groups they are said to hold, or the processor
/// lacks `FEAT_DotProd`.
pub(crate) fn qgemm_dotprod(
    rows: usize,
    groups: usize,
    a: &[u8],
    a_floats: &[f32],
    panel: &[u8],
    tile: &mut [[f32; PANEL]; TILE_ROWS],
) {
    assert!((1..=TILE_ROWS).contains(&rows), "a tile of {rows} rows");
    assert!(groups > 0, "products of no groups");
    let a_stride = groups * GROUP;
    assert!(
        a.len() >= rows * a_stride,
        "inputs shorter than {rows} rows of {groups} groups"
    );
    assert!(
        a_floats.len() >= rows * groups * 2,
        "fewer scales and offsets than {rows} rows of {groups} groups"
    );
    assert!(
        panel.len() >= groups * GROUP_BYTES,
        "a panel shorter than {groups} groups"
    );
    assert!(has_dotprod(), "the dot products' kernels on a processor without them");
    let kernel = DOTPROD[rows - 1];
    // SAFETY: checked above: the processor has NEON and FEAT_DotProd, which the
    // kernel executes; it reads `rows` rows of `a_stride` bytes from `a`, which
    // holds at least that many, the rows' `2 * groups` floats each, 8 bytes a
    // group, from `a_floats`, which holds them, and `groups` groups of
    // `GROUP_BYTES` from `panel`, which holds them; it reads and writes `rows`
    // rows of `PANEL` floats, 64 bytes apart, of `tile`, which has four. It
    // keeps to AAPCS64, as declared: it touches the registers the caller saves,
    // and v8, whose low half it puts back, never x18 nor sp, and keeps no
    // pointer after it returns.
    unsafe {
        kernel(
            groups,
            a.as_ptr(),
            a_stride,
            a_floats.as_ptr(),
            panel.as_ptr(),
            tile.as_mut_ptr().cast::<f32>(),
        );
    }
}
