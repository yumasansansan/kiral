// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later
//
// This file is part of KIRAL. The SPDX lines name its copyright holder and its
// license, in the machine-readable form of the REUSE specification: the GNU
// General Public License, version 3 or any later version
// (LICENSES/GPL-3.0-or-later.txt).

//! `kiral-tools replay`: the time an input method takes for each key, as a user
//! types: AJIMEE-Bench's readings in romaji, a key at a time, then Space.
//!
//! For now the input method is Mozc, through `mozc_emacs_helper`, the time of a
//! key that of its answer -- the helper, Mozc's IPC and its server included. A
//! run is made on each processor asked for, the thread that types pinned to it
//! before Mozc is started, so that its helper and its server, which inherit the
//! pin, run there too; the clock of the processor is read every 20 ms while the
//! run goes on, and given with the times. The first item is typed once before
//! the times are taken, while the server loads its dictionary.

use std::ffi::OsString;
use std::process::ExitCode;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use kiral_cpu::topology::{self, Topology};

use crate::ajimee::{self, Item};
use crate::arguments;
use crate::eval::type_item;
use crate::metrics::percentile;
use crate::mozc::{Mozc, Programs};

const USAGE: &str = "\
usage: kiral-tools replay --mozc [options]

The time an input method takes for each key of AJIMEE-Bench's readings typed in
romaji, and for each conversion (Space): the median, the 99th percentile and the
longest, on each processor asked for, with the clock it ran at.

  --mozc                Mozc, through mozc_emacs_helper, with a profile of the
                        run's own that it removes after (Linux)
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
        &["mozc-helper", "mozc-server", "items", "limit"],
        &["cpu"],
        &["mozc", "help"],
    )
    .map_err(|message| format!("{message}\n\n{USAGE}"))?;
    if options.flag("help") {
        println!("{USAGE}");
        return Ok(());
    }
    if !options.flag("mozc") {
        return Err(format!(
            "no input method to type into: --mozc is the one there is for now\n\n{USAGE}"
        ));
    }
    let programs = Programs {
        helper: options
            .path("mozc-helper")
            .unwrap_or_else(|| Programs::default().helper),
        server: options
            .path("mozc-server")
            .unwrap_or_else(|| Programs::default().server),
    };
    let items = ajimee::load(options.path("items"), options.parsed::<usize>("limit")?)?;
    let topology = Topology::detect();
    let processors = options
        .values("cpu")
        .iter()
        .map(|name| {
            topology
                .processor(name)
                .map(Some)
                .ok_or_else(|| format!("--cpu {name}: no such processor in {topology:?}"))
        })
        .collect::<Result<Vec<_>, String>>()?;
    let processors = if processors.is_empty() { vec![None] } else { processors };
    let name = topology::processor_name().unwrap_or_else(|| std::env::consts::ARCH.to_owned());
    println!(
        "replay: AJIMEE-Bench JWTD_v2/v1, {} items typed in romaji, into Mozc through mozc_emacs_helper; {name}",
        items.len()
    );
    for processor in processors {
        let (times, clock, version) = replay_on(&programs, &items, processor)?;
        let place = match processor {
            Some(processor) => format!(
                "CPU {processor} ({}, pinned){}",
                topology.kind(processor).map_or_else(|| "?".to_owned(), String::from),
                clock
            ),
            None => "unpinned".to_owned(),
        };
        println!("  Mozc {version}, {place}");
        print_times("keys", &times.keys);
        print_times("conversions (Space)", &times.conversions);
    }
    Ok(())
}

/// Types the items on a processor, the clock read beside: the times, the clock
/// as text, and Mozc's version.
fn replay_on(programs: &Programs, items: &[Item], processor: Option<usize>) -> Result<(Times, String, String), String> {
    let done = AtomicBool::new(false);
    let (result, samples) = std::thread::scope(|scope| {
        let typist = scope.spawn(|| {
            let result = type_all(programs, items, processor);
            done.store(true, Ordering::Release);
            result
        });
        let mut samples = Vec::new();
        while !done.load(Ordering::Acquire) {
            if let Some(khz) = processor.and_then(topology::clock_khz) {
                samples.push(khz);
            }
            std::thread::sleep(Duration::from_millis(20));
        }
        (typist.join().expect("a typist that did not panic"), samples)
    });
    let (times, version) = result?;
    Ok((times, clock_of(samples), version))
}

/// The clock read while a run went on: its median and range, in GHz.
fn clock_of(mut samples: Vec<u64>) -> String {
    if samples.is_empty() {
        return String::new();
    }
    samples.sort_unstable();
    let ghz = |khz: u64| f64::from(u32::try_from(khz / 1000).unwrap_or(u32::MAX)) / 1000.0;
    format!(
        ", {:.2} GHz at the median of {} readings of its clock, {:.2} to {:.2}",
        ghz(samples[samples.len() / 2]),
        samples.len(),
        ghz(samples[0]),
        ghz(samples[samples.len() - 1])
    )
}

/// Types every item, the thread pinned first; returns the times and Mozc's
/// version.
fn type_all(programs: &Programs, items: &[Item], processor: Option<usize>) -> Result<(Times, String), String> {
    if let Some(processor) = processor {
        kiral_sys::affinity::pin_current_thread(processor)
            .map_err(|error| format!("pinning to processor {processor}: {error}"))?;
    }
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
    Ok((times, mozc.version().to_owned()))
}

fn print_times(what: &str, times: &[Duration]) {
    let mut sorted = times.to_vec();
    sorted.sort_unstable();
    let milliseconds = |time: Duration| time.as_secs_f64() * 1000.0;
    println!(
        "    {what}: {} of them, p50 {:.3} ms, p99 {:.3} ms, longest {:.3} ms",
        sorted.len(),
        milliseconds(percentile(&sorted, 50)),
        milliseconds(percentile(&sorted, 99)),
        milliseconds(sorted.last().copied().unwrap_or_default())
    );
}
