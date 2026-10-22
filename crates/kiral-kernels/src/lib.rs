// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later
//
// This file is part of KIRAL. The SPDX lines name its copyright holder and its
// license, in the machine-readable form of the REUSE specification: the GNU
// General Public License, version 3 or any later version
// (LICENSES/GPL-3.0-or-later.txt).

//! The kernels of KIRAL's networks.
//!
//! Every kernel has a portable form in plain Rust, written so that the compiler
//! turns it into vector code for the processor it compiles for: the loops run
//! over chunks of a fixed size, and a sum is kept in [`LANES`] lanes, each a sum
//! of its own, which the compiler may keep in vector registers, where a single
//! sum would be one long chain of additions it may not reorder. The portable
//! form is the reference. A faster form, in assembly, is added only with a
//! measurement that shows the portable one slower, and is tested against it at
//! every level of the processor ([`kiral-cpu`](../kiral_cpu/index.html)).
//!
//! The kernels here compute in single precision, as the networks do outside
//! their products: layer normalization, GELU, softmax, the residual stream and
//! attention over the cache of keys and values. The linear layers in single
//! precision ([`linear`]) are those of the reference that quantized models are
//! measured against.
//!
//! What a kernel promises is a precision, not an order of operations: each
//! value is within what the analysis of its roundings allows of the exact
//! result, which the tests check against double precision, and the order of the
//! additions is whatever computes it fastest. A result may so differ in its last
//! bits between builds, between processors, and between a row computed alone and
//! the same row computed among others.
#![forbid(unsafe_code)]

pub mod activation;
pub mod attention;
pub mod linear;
pub mod norm;
pub mod vector;

/// The lanes a sum is kept in: 16 values of single precision, 512 bits, one
/// vector of AVX-512, two of AVX2, four of SSE or NEON.
pub const LANES: usize = 16;

/// `a b + c`, rounded once where the target of the build has fused
/// multiply-adds (x86-64 with FMA, and every AArch64), and twice elsewhere, the
/// product and then the sum: a fused multiply-add the processor has not got
/// would be a call into the system's library, many times slower.
#[inline]
#[must_use]
pub fn multiply_add(a: f32, b: f32, c: f32) -> f32 {
    if cfg!(any(target_feature = "fma", target_arch = "aarch64")) {
        a.mul_add(b, c)
    } else {
        a * b + c
    }
}

/// The sum of the lanes: each with the one half the lanes after it, then the
/// halves of what is left, a tree four additions deep.
#[inline]
#[must_use]
pub(crate) fn sum_lanes(mut lanes: [f32; LANES]) -> f32 {
    let mut width = LANES;
    while width > 1 {
        width /= 2;
        let (low, high) = lanes[..2 * width].split_at_mut(width);
        for (low, &high) in low.iter_mut().zip(high.iter()) {
            *low += high;
        }
    }
    lanes[0]
}

/// The sum of the values: in [`LANES`] lanes, the lanes summed in a tree, then
/// the values the lanes leave over, one after another.
#[must_use]
pub fn sum(values: &[f32]) -> f32 {
    let (chunks, rest) = values.as_chunks::<LANES>();
    let mut lanes = [0.0f32; LANES];
    for chunk in chunks {
        for (lane, &value) in lanes.iter_mut().zip(chunk) {
            *lane += value;
        }
    }
    let mut total = sum_lanes(lanes);
    for &value in rest {
        total += value;
    }
    total
}

/// A length as a float: exact, every length the kernels take being below 2^24.
///
/// # Panics
///
/// Where the length is 2^24 or more, which a float would round.
#[must_use]
pub(crate) fn length(count: usize) -> f32 {
    assert!(
        count < 1 << 24,
        "a length of {count}, which single precision would round"
    );
    #[expect(clippy::as_conversions, reason = "a count below 2^24 converts exactly")]
    #[expect(clippy::cast_precision_loss, reason = "a count below 2^24 converts exactly")]
    let value = count as f32;
    value
}
