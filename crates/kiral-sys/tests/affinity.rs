// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later
//
// This file is part of KIRAL. The SPDX lines name its copyright holder and its
// license, in the machine-readable form of the REUSE specification: the GNU
// General Public License, version 3 or any later version
// (LICENSES/GPL-3.0-or-later.txt).

//! A thread pinned to a processor runs on it alone, and a processor the
//! machine does not have is refused. Each test runs on a thread of its own, and
//! pins only that thread.

use kiral_sys::affinity;

#[cfg(target_os = "linux")]
#[test]
fn a_pinned_thread_runs_on_its_processor_alone() {
    let allowed = affinity::current_thread_processors().expect("the thread's processors");
    let last = *allowed.last().expect("a processor the thread may run on");
    affinity::pin_current_thread(last).expect("pinned");
    assert_eq!(
        affinity::current_thread_processors().expect("the thread's processors"),
        [last]
    );
}

#[cfg(windows)]
#[test]
fn a_thread_is_pinned_to_its_first_processor() {
    affinity::pin_current_thread(0).expect("pinned");
}

#[test]
fn a_processor_out_of_reach_is_refused() {
    assert!(affinity::pin_current_thread(100_000).is_err());
}
