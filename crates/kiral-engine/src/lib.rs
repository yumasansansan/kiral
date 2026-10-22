// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later
//
// This file is part of KIRAL. The SPDX lines name its copyright holder and its
// license, in the machine-readable form of the REUSE specification: the GNU
// General Public License, version 3 or any later version
// (LICENSES/GPL-3.0-or-later.txt).

//! KIRAL's engine: the dictionary's drafts, verified by the model.
//!
//! - [`refine`]: a draft -- the cheapest path of the reading's lattice --
//!   verified by zenz in one feeding, and searched again within what the model
//!   accepted and the token it wants, as Zenzai does, round after round.
//! - [`engine`]: the reading drafted on the input thread, which never waits for
//!   the model, and refined on a worker, a round a key, the results of a
//!   reading that has changed since dropped.
#![forbid(unsafe_code)]

pub mod engine;
pub mod refine;
