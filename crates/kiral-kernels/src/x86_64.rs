// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later
//
// This file is part of KIRAL. The SPDX lines name its copyright holder and its
// license, in the machine-readable form of the REUSE specification: the GNU
// General Public License, version 3 or any later version
// (LICENSES/GPL-3.0-or-later.txt).

//! The kernels in assembly for x86-64 (`asm/x86_64/`), declared, and the safe
//! function that calls them: it checks, in every build, what its kernel reads
//! and writes and that the processor has the kernel's extensions, then makes the
//! call, the one unsafe operation of its block.
#![allow(
    unsafe_code,
    reason = "the calls into assembly, each the one operation of a block whose comment says why it is sound"
)]

use crate::quantized::{GROUP, GROUP_BYTES, PANEL};

/// A kernel of the products in integers for a number of rows
/// (`qgemm_avx2.S`, `qgemm_avx_vnni.S`, `qgemm_avx512_vnni.S`).
type Qgemm = unsafe extern "sysv64" fn(
    groups: usize,
    a: *const u8,
    a_stride: usize,
    a_floats: *const f32,
    b: *const u8,
    c: *mut f32,
);

unsafe extern "sysv64" {
    fn kiral_qgemm_avx2_1(
        groups: usize,
        a: *const u8,
        a_stride: usize,
        a_floats: *const f32,
        b: *const u8,
        c: *mut f32,
    );
    fn kiral_qgemm_avx2_2(
        groups: usize,
        a: *const u8,
        a_stride: usize,
        a_floats: *const f32,
        b: *const u8,
        c: *mut f32,
    );
    fn kiral_qgemm_avx2_3(
        groups: usize,
        a: *const u8,
        a_stride: usize,
        a_floats: *const f32,
        b: *const u8,
        c: *mut f32,
    );
    fn kiral_qgemm_avx2_4(
        groups: usize,
        a: *const u8,
        a_stride: usize,
        a_floats: *const f32,
        b: *const u8,
        c: *mut f32,
    );
    fn kiral_qgemm_avx2_5(
        groups: usize,
        a: *const u8,
        a_stride: usize,
        a_floats: *const f32,
        b: *const u8,
        c: *mut f32,
    );
    fn kiral_qgemm_avx2_6(
        groups: usize,
        a: *const u8,
        a_stride: usize,
        a_floats: *const f32,
        b: *const u8,
        c: *mut f32,
    );
    fn kiral_qgemm_avxvnni_1(
        groups: usize,
        a: *const u8,
        a_stride: usize,
        a_floats: *const f32,
        b: *const u8,
        c: *mut f32,
    );
    fn kiral_qgemm_avxvnni_2(
        groups: usize,
        a: *const u8,
        a_stride: usize,
        a_floats: *const f32,
        b: *const u8,
        c: *mut f32,
    );
    fn kiral_qgemm_avxvnni_3(
        groups: usize,
        a: *const u8,
        a_stride: usize,
        a_floats: *const f32,
        b: *const u8,
        c: *mut f32,
    );
    fn kiral_qgemm_avxvnni_4(
        groups: usize,
        a: *const u8,
        a_stride: usize,
        a_floats: *const f32,
        b: *const u8,
        c: *mut f32,
    );
    fn kiral_qgemm_avxvnni_5(
        groups: usize,
        a: *const u8,
        a_stride: usize,
        a_floats: *const f32,
        b: *const u8,
        c: *mut f32,
    );
    fn kiral_qgemm_avxvnni_6(
        groups: usize,
        a: *const u8,
        a_stride: usize,
        a_floats: *const f32,
        b: *const u8,
        c: *mut f32,
    );
    fn kiral_qgemm_avx512vnni_1(
        groups: usize,
        a: *const u8,
        a_stride: usize,
        a_floats: *const f32,
        b: *const u8,
        c: *mut f32,
    );
    fn kiral_qgemm_avx512vnni_2(
        groups: usize,
        a: *const u8,
        a_stride: usize,
        a_floats: *const f32,
        b: *const u8,
        c: *mut f32,
    );
    fn kiral_qgemm_avx512vnni_3(
        groups: usize,
        a: *const u8,
        a_stride: usize,
        a_floats: *const f32,
        b: *const u8,
        c: *mut f32,
    );
    fn kiral_qgemm_avx512vnni_4(
        groups: usize,
        a: *const u8,
        a_stride: usize,
        a_floats: *const f32,
        b: *const u8,
        c: *mut f32,
    );
    fn kiral_qgemm_avx512vnni_5(
        groups: usize,
        a: *const u8,
        a_stride: usize,
        a_floats: *const f32,
        b: *const u8,
        c: *mut f32,
    );
    fn kiral_qgemm_avx512vnni_6(
        groups: usize,
        a: *const u8,
        a_stride: usize,
        a_floats: *const f32,
        b: *const u8,
        c: *mut f32,
    );
}

/// The rows a tile of the kernels takes at most, of every form of x86-64.
pub(crate) const TILE_ROWS: usize = 6;

/// The kernels of the products in integers of x86-64, by their form.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Products {
    /// AVX2 without VNNI, on inputs of seven bits.
    Avx2,
    /// AVX-VNNI.
    AvxVnni,
    /// AVX-512 with VNNI.
    Avx512Vnni,
}

impl Products {
    /// The kernels by their number of rows, from one.
    fn kernels(self) -> [Qgemm; TILE_ROWS] {
        match self {
            Self::Avx2 => [
                kiral_qgemm_avx2_1,
                kiral_qgemm_avx2_2,
                kiral_qgemm_avx2_3,
                kiral_qgemm_avx2_4,
                kiral_qgemm_avx2_5,
                kiral_qgemm_avx2_6,
            ],
            Self::AvxVnni => [
                kiral_qgemm_avxvnni_1,
                kiral_qgemm_avxvnni_2,
                kiral_qgemm_avxvnni_3,
                kiral_qgemm_avxvnni_4,
                kiral_qgemm_avxvnni_5,
                kiral_qgemm_avxvnni_6,
            ],
            Self::Avx512Vnni => [
                kiral_qgemm_avx512vnni_1,
                kiral_qgemm_avx512vnni_2,
                kiral_qgemm_avx512vnni_3,
                kiral_qgemm_avx512vnni_4,
                kiral_qgemm_avx512vnni_5,
                kiral_qgemm_avx512vnni_6,
            ],
        }
    }

    /// Whether the processor has what the kernels run, as std reports it (once,
    /// then from its cache): AVX2 and FMA, and AVX-VNNI, or AVX-512 F, BW, VL
    /// and VNNI.
    fn runs(self) -> bool {
        use std::arch::is_x86_feature_detected;
        let avx2 = is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma");
        avx2 && match self {
            Self::Avx2 => true,
            Self::AvxVnni => is_x86_feature_detected!("avxvnni"),
            Self::Avx512Vnni => {
                is_x86_feature_detected!("avx512f")
                    && is_x86_feature_detected!("avx512bw")
                    && is_x86_feature_detected!("avx512vl")
                    && is_x86_feature_detected!("avx512vnni")
            }
        }
    }
}

/// Adds to the first `rows` rows of `tile` their products with `groups` groups
/// of a panel, by the kernel of that many rows of a form: `a` the rows' bytes,
/// `groups` groups of 64 each, `a_floats` each group's scale and offset, row by
/// row, and `panel` the panel's groups, of [`GROUP_BYTES`].
///
/// # Panics
///
/// Where `rows` is not 1 to [`TILE_ROWS`], `groups` is 0, the slices are
/// shorter than the rows and groups they are said to hold, or the processor
/// lacks the form's extensions.
pub(crate) fn qgemm(
    products: Products,
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
    assert!(
        products.runs(),
        "the {products:?} kernels on a processor without its extensions"
    );
    let kernel = products.kernels()[rows - 1];
    // SAFETY: checked above: the processor has the extensions the form's kernel
    // executes (AVX2 and FMA, with AVX-VNNI or AVX-512 F, BW, VL and VNNI for
    // theirs); it reads `rows` rows of `a_stride` bytes from `a`, which holds
    // at least that many, the rows' `2 * groups` floats each, 8 bytes a group,
    // from `a_floats`, which holds them, and `groups` groups of `GROUP_BYTES`
    // from `panel`, which holds them, and AVX2's 32 bytes of its own
    // constants; it reads and writes `rows` rows of `PANEL` floats, 64 bytes
    // apart, of `tile`, which has six. It keeps to System V's convention, as
    // declared, touches only registers the caller saves, leaves rsp alone, and
    // keeps no pointer after it returns.
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
