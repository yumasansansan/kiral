// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later
//
// This file is part of KIRAL. The SPDX lines name its copyright holder and its
// license, in the machine-readable form of the REUSE specification: the GNU
// General Public License, version 3 or any later version
// (LICENSES/GPL-3.0-or-later.txt).

//! The files models come in, read by hand.
//!
//! - [`safetensors`]: the form zenz's weights are published in.
//! - [`json`]: the headers of safetensors files, and the tokenizers' files.
//! - [`half`]: bfloat16 and half precision, to single precision.
//!
//! Every parser here takes any bytes, and returns an error for those it cannot
//! read, never a panic.
#![forbid(unsafe_code)]

pub mod half;
pub mod json;
pub mod safetensors;
