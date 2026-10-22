// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later
//
// This file is part of KIRAL. The SPDX lines name its copyright holder and its
// license, in the machine-readable form of the REUSE specification: the GNU
// General Public License, version 3 or any later version
// (LICENSES/GPL-3.0-or-later.txt).

//! The operating systems' functions KIRAL calls, declared by hand: no crate
//! brings them.
//!
//! - [`affinity`]: the processors a thread may run on.
//! - [`map`]: files mapped into memory, read-only.
//!
//! Each call into a system is one unsafe block, whose `SAFETY` comment says
//! what makes it sound; the functions here are safe to call.

pub mod affinity;
pub mod map;
