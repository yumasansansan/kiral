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
//! The forms ([`Form`]): the portable one, and the kernels of the products in
//! integers in assembly, which `build.rs` assembles with Clang: for AVX2
//! without VNNI (`asm/x86_64/qgemm_avx2.S`), AVX-VNNI
//! (`asm/x86_64/qgemm_avx_vnni.S`), AVX-512 with VNNI
//! (`asm/x86_64/qgemm_avx512_vnni.S`) and AArch64's dot products
//! (`asm/aarch64/qgemm_dotprod.S`). The form in effect ([`form`]) is the most
//! preferred that the processor runs and the environment allows (`KIRAL_ISA`,
//! `KIRAL_KERNELS=portable`), chosen once; each call into assembly is one
//! unsafe block, in the module of its architecture, whose comment says what
//! was checked, and the rest of the crate denies unsafe code.
//!
//! The kernels compute in single precision, as the networks do outside their
//! products: layer normalization, GELU, softmax, the residual stream and
//! attention over the cache of keys and values. The linear layers in single
//! precision ([`linear`]) are those of the reference that quantized models are
//! measured against; those in integers ([`quantized`]) take weights and inputs
//! quantized to bytes, their products summed exactly in 32 bits.
//!
//! What a kernel promises is a precision, not an order of operations: each
//! value is within what the analysis of its roundings allows of the exact
//! result, which the tests check against double precision, and the order of the
//! additions is whatever computes it fastest. A result may so differ in its last
//! bits between builds, between processors, and between a row computed alone and
//! the same row computed among others.
#![deny(unsafe_code)]

use std::sync::OnceLock;

use kiral_cpu::{Choice, Features, Level, Request};

#[cfg(target_arch = "aarch64")]
mod aarch64;
pub mod activation;
pub mod attention;
pub mod linear;
pub mod norm;
pub mod quantized;
pub mod vector;
#[cfg(target_arch = "x86_64")]
mod x86_64;

/// A form a kernel is computed in: the portable one, the reference, or one in
/// assembly for a level of the processor's.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Form {
    /// Plain Rust, compiled for the build's baseline.
    Portable,
    /// Assembly for AVX2 without VNNI: `vpmaddubsw` and `vpmaddwd` on 256 bits,
    /// the inputs of seven bits (x86-64).
    Avx2,
    /// Assembly for AVX-VNNI: `vpdpbusd` on 256 bits (x86-64).
    AvxVnni,
    /// Assembly for AVX-512 with VNNI: `vpdpbusd` on 512 bits (x86-64).
    Avx512Vnni,
    /// Assembly for AArch64's dot products: `sdot` (AArch64).
    Dotprod,
}

impl Form {
    /// Every form, the reference first, then the more preferred after the less.
    pub const ALL: [Self; 5] = [
        Self::Portable,
        Self::Avx2,
        Self::AvxVnni,
        Self::Avx512Vnni,
        Self::Dotprod,
    ];

    /// The form's name, as the tools take it.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Portable => "portable",
            Self::Avx2 => "avx2",
            Self::AvxVnni => "avx-vnni",
            Self::Avx512Vnni => "avx512-vnni",
            Self::Dotprod => "dotprod",
        }
    }

    /// The form of a name.
    #[must_use]
    pub fn from_name(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|form| form.name() == name)
    }

    /// The level the form needs; none for the portable form.
    #[must_use]
    pub const fn level(self) -> Option<Level> {
        match self {
            Self::Portable => None,
            Self::Avx2 => Some(Level::Avx2),
            Self::AvxVnni => Some(Level::AvxVnni),
            Self::Avx512Vnni => Some(Level::Avx512Vnni),
            Self::Dotprod => Some(Level::Dotprod),
        }
    }

    /// Whether this build has the form: the portable one always, one in
    /// assembly on its architecture.
    #[must_use]
    pub const fn is_built(self) -> bool {
        match self {
            Self::Portable => true,
            Self::Avx2 | Self::AvxVnni | Self::Avx512Vnni => cfg!(target_arch = "x86_64"),
            Self::Dotprod => cfg!(target_arch = "aarch64"),
        }
    }

    /// The greatest integer of an input that the form's products take: 255,
    /// a byte's, or 127 for AVX2, whose `vpmaddubsw` sums two products of an
    /// input by a weight in a word, which two of 255 by 127 would pass. The
    /// inputs are quantized to as many levels for it ([`quantized::Rows`]).
    #[must_use]
    pub const fn input_max(self) -> f32 {
        match self {
            Self::Avx2 => 127.0,
            Self::Portable | Self::AvxVnni | Self::Avx512Vnni | Self::Dotprod => 255.0,
        }
    }

    /// The forms that a processor of these extensions runs, the reference
    /// first: those the tests take every kernel in.
    #[must_use]
    pub fn available(features: Features) -> Vec<Self> {
        Self::ALL
            .into_iter()
            .filter(|form| form.is_built() && form.level().is_none_or(|level| features.has(level)))
            .collect()
    }

    /// The form a choice of the kernels takes: the most preferred form the
    /// processor runs whose level is within the choice's, or the portable form
    /// where the choice asks for it.
    #[must_use]
    pub fn of(choice: Choice, features: Features) -> Self {
        if choice.portable {
            return Self::Portable;
        }
        Self::available(features)
            .into_iter()
            .rfind(|form| {
                form.level()
                    .is_none_or(|level| choice.level.is_some_and(|chosen| level <= chosen))
            })
            .unwrap_or(Self::Portable)
    }
}

impl std::fmt::Display for Form {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.name())
    }
}

/// The form in effect, and why the environment's request was not taken, where
/// it was not.
static FORM: OnceLock<(Form, Option<kiral_cpu::Error>)> = OnceLock::new();

fn chosen() -> &'static (Form, Option<kiral_cpu::Error>) {
    FORM.get_or_init(|| {
        let features = Features::detect();
        match Request::from_environment() {
            Ok(request) => (Form::of(Choice::new(features, request), features), None),
            Err(error) => (Form::Portable, Some(error)),
        }
    })
}

/// The form of the kernels in effect: the most preferred that the processor
/// runs and the environment allows, chosen at the first call; the portable
/// form where the environment's request cannot be read ([`form_error`]).
#[must_use]
pub fn form() -> Form {
    chosen().0
}

/// Why the environment's request of the kernels was not taken, where it was
/// not.
#[must_use]
pub fn form_error() -> Option<&'static kiral_cpu::Error> {
    chosen().1.as_ref()
}

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
