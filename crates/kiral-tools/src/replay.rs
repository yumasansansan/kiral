// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later
//
// This file is part of KIRAL. The SPDX lines name its copyright holder and its
// license, in the machine-readable form of the REUSE specification: the GNU
// General Public License, version 3 or any later version
// (LICENSES/GPL-3.0-or-later.txt).

//! `kiral-tools replay`: the time an input method takes for each key, as a user
//! types AJIMEE-Bench's readings, and for each conversion with Space.
//!
//! Two are timed for now. Mozc, through `mozc_emacs_helper`, the readings typed
//! in romaji, the time of a key that of its answer -- the helper, Mozc's IPC and
//! its server included -- the first item typed once before the times are
//! taken, while the server loads its dictionary. And KIRAL's dictionary alone,
//! the readings typed a kana at a time, each prefix converted from its lattice
//! up: the time of a key that of building the lattice, finding its best path
//! and making its text, as an engine that kept nothing between keys would take.
//!
//! A run is made on each processor asked for, the thread that types pinned to
//! it before anything starts, so that Mozc's helper and server, which inherit
//! the pin, run there too; the clock of the processor is read every 20 ms while
//! the run goes on, and given with the times ([`crate::pinned`]).

use std::ffi::OsString;
use std::process::ExitCode;
use std::time::{Duration, Instant};

use kiral_cpu::topology::{self, Topology};
use kiral_dict::Dictionary;
use kiral_lattice::Lattice;

use crate::ajimee::{self, Item};
use crate::arguments;
use crate::eval::type_item;
use crate::mozc::{Mozc, Programs};
use crate::pinned::{place, print_times, processors, run_on};

const USAGE: &str = "\
usage: kiral-tools replay (--dict <file> | --mozc) [options]

The time an input method takes for each key of AJIMEE-Bench's readings, and for
each conversion (Space): the median, the 99th percentile and the longest, on
each processor asked for, with the clock it ran at.

  --dict <file>         a dictionary file: the readings typed a kana at a time,
                        each prefix converted by the dictionary alone
  --mozc                Mozc, through mozc_emacs_helper, the readings typed in
                        romaji, with a profile of the run's own that it removes
                        after (Linux)
  --mozc-helper <path>  the helper [mozc_emacs_helper]
  --mozc-server <path>  the server [/usr/lib/mozc/mozc_server]
  --cpu <processor>     a processor to pin the run to, again for another run:
                        p<n> the n-th performance core, e<n> the n-th
                        efficiency core, or the system's number [none: unpinned]
  --items <file>        the items [data/ajimee/evaluation_items.json]
  --limit <n>           the first n items only";

/// Runs the command.
pub fn main(arguments: impl IntoIterator<Item = OsString>) -> ExitCode {
    match run(arguments) {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => {
            eprintln!("error: {message}");
            ExitCode::FAILURE
        }
    }
}

/// The times of a run.
#[derive(Debug, Default)]
struct Times {
    keys: Vec<Duration>,
    conversions: Vec<Duration>,
}

fn run(arguments: impl IntoIterator<Item = OsString>) -> Result<(), String> {
    let options = arguments::parse(
        arguments,
        &["dict", "mozc-helper", "mozc-server", "items", "limit"],
        &["cpu"],
        &["mozc", "help"],
    )
    .map_err(|message| format!("{message}\n\n{USAGE}"))?;
    if options.flag("help") {
        println!("{USAGE}");
        return Ok(());
    }
    let dictionary_path = options.path("dict");
    if dictionary_path.is_some() == options.flag("mozc") {
        return Err(format!("one of --dict and --mozc\n\n{USAGE}"));
    }
    let items = ajimee::load(options.path("items"), options.parsed::<usize>("limit")?)?;
    let topology = Topology::detect();
    let processors = processors(&options, &topology)?;
    let name = topology::processor_name().unwrap_or_else(|| std::env::consts::ARCH.to_owned());
    let bytes = match &dictionary_path {
        Some(path) => std::fs::read(path).map_err(|error| format!("{}: {error}", path.display()))?,
        None => Vec::new(),
    };
    let dictionary = match &dictionary_path {
        Some(path) => Some(Dictionary::parse(&bytes).map_err(|error| format!("{}: {error}", path.display()))?),
        None => None,
    };
    let programs = Programs {
        helper: options
            .path("mozc-helper")
            .unwrap_or_else(|| Programs::default().helper),
        server: options
            .path("mozc-server")
            .unwrap_or_else(|| Programs::default().server),
    };
    if dictionary.is_some() {
        println!(
            "replay: AJIMEE-Bench JWTD_v2/v1, {} items typed a kana at a time, each converted by the dictionary \
             alone from its lattice up; {name}",
            items.len()
        );
    } else {
        println!(
            "replay: AJIMEE-Bench JWTD_v2/v1, {} items typed in romaji, into Mozc through mozc_emacs_helper; {name}",
            items.len()
        );
    }
    for processor in processors {
        let ((times, system), clock) = match &dictionary {
            Some(dictionary) => run_on(processor, || Ok(type_with_dictionary(dictionary, &items)))?,
            None => run_on(processor, || type_with_mozc(&programs, &items))?,
        };
        println!("  {system}, {}", place(&topology, processor, &clock));
        print_times(if dictionary.is_some() { "kana" } else { "keys" }, &times.keys);
        print_times("conversions (Space)", &times.conversions);
    }
    Ok(())
}

/// Types every item into Mozc; returns the times and Mozc's name.
fn type_with_mozc(programs: &Programs, items: &[Item]) -> Result<(Times, String), String> {
    let mut mozc = Mozc::start(programs)?;
    let mut warming = Vec::new();
    type_item(&mut mozc, &items[0], &mut warming)?;
    mozc.convert()?;
    mozc.cancel()?;
    let mut times = Times::default();
    for item in items {
        type_item(&mut mozc, item, &mut times.keys)?;
        let (_, time) = mozc
            .convert()
            .map_err(|error| format!("item {}: {error}", item.index))?;
        times.conversions.push(time);
        mozc.cancel().map_err(|error| format!("item {}: {error}", item.index))?;
    }
    Ok((times, format!("Mozc {}", mozc.version())))
}

/// The time of a conversion by the dictionary alone: the lattice built, its best
/// path found and its text made.
fn convert(dictionary: &Dictionary<'_>, reading: &str) -> Duration {
    let started = Instant::now();
    let lattice = Lattice::new(dictionary, reading);
    let best = lattice.best(dictionary);
    std::hint::black_box(lattice.text(&best));
    started.elapsed()
}

/// Types every item into the dictionary's conversion a kana at a time, each
/// prefix converted from its lattice up, as an engine that keeps nothing between
/// keys would, then converts the whole; returns the times and the
/// dictionary's name. The first item is converted once before, while the
/// dictionary's pages come into the caches.
fn type_with_dictionary(dictionary: &Dictionary<'_>, items: &[Item]) -> (Times, String) {
    std::hint::black_box(convert(dictionary, &items[0].reading));
    let mut times = Times::default();
    for item in items {
        let reading: Vec<char> = item.reading.chars().collect();
        for length in 1..=reading.len() {
            let prefix: String = reading[..length].iter().collect();
            times.keys.push(convert(dictionary, &prefix));
        }
        times.conversions.push(convert(dictionary, &item.reading));
    }
    (times, format!("the dictionary {}", dictionary.metadata().name))
}
