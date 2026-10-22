// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later
//
// This file is part of KIRAL. The SPDX lines name its copyright holder and its
// license, in the machine-readable form of the REUSE specification: the GNU
// General Public License, version 3 or any later version
// (LICENSES/GPL-3.0-or-later.txt).

//! The levels std's detection finds on the processor the tests run on. Under
//! Intel's SDE, ci/emulated.sh says in `KIRAL_EXPECT_LEVELS` which levels the
//! processor it emulates has, and they must be those found; elsewhere the run
//! prints them, for the log.

use kiral_cpu::{Choice, Features, LEVELS, Level, Request};

#[test]
fn the_levels_found_are_those_expected() {
    let found = Features::detect().levels();
    let names = found.iter().map(|level| level.name()).collect::<Vec<_>>();
    println!("the levels of this processor: {}", names.join(", "));
    let Ok(expected) = std::env::var("KIRAL_EXPECT_LEVELS") else {
        return;
    };
    let expected = expected
        .split(',')
        .map(str::trim)
        .filter(|name| !name.is_empty())
        .map(|name| Level::from_name(name).unwrap_or_else(|| panic!("KIRAL_EXPECT_LEVELS names {name}, no level")))
        .collect::<Vec<_>>();
    assert_eq!(
        found, expected,
        "the levels found are not those of the processor emulated"
    );
}

#[test]
fn the_baseline_is_always_there() {
    // The build is compiled for it: a processor without it would not have got here.
    if let Some(&baseline) = LEVELS.first() {
        assert!(
            Features::detect().has(baseline),
            "no {baseline} on the processor running the build"
        );
    }
}

#[test]
fn the_choice_keeps_to_the_request_of_the_environment() {
    let request = Request::from_environment().expect("KIRAL_ISA and KIRAL_KERNELS as the tests run with them");
    let features = Features::detect();
    let choice = Choice::new(features, request);
    println!("the kernels taken: {choice}");
    if let Some(level) = choice.level {
        assert!(features.has(level));
        assert!(request.limit.is_none_or(|limit| level <= limit));
    }
    assert_eq!(choice.portable, request.portable);
}
