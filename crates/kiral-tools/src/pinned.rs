// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later
//
// This file is part of KIRAL. The SPDX lines name its copyright holder and its
// license, in the machine-readable form of the REUSE specification: the GNU
// General Public License, version 3 or any later version
// (LICENSES/GPL-3.0-or-later.txt).

//! Runs pinned to a processor, its clock read beside: what `replay` and
//! `bench` time on.
//!
//! The thread that does the work is pinned before it starts, so that processes
//! it starts inherit the pin; the processor's clock is read every 20 ms while
//! the work goes on, and given with the times, the median and the range of
//! the readings.

use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use kiral_cpu::topology::{self, Topology};

use crate::arguments;
use crate::metrics::percentile;

/// The processors the options' `--cpu`s name, in their order: none, for one
/// run unpinned, where there are none.
///
/// # Errors
///
/// Where a name is no processor of the system's.
pub fn processors(options: &arguments::Options, topology: &Topology) -> Result<Vec<Option<usize>>, String> {
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
    Ok(if processors.is_empty() { vec![None] } else { processors })
}

/// Does the work on a processor, the thread pinned to it first, the clock read
/// beside: what the work returns, and the clock as text.
///
/// # Errors
///
/// Where the thread cannot be pinned, or the work fails.
pub fn run_on<T, F>(processor: Option<usize>, work: F) -> Result<(T, String), String>
where
    T: Send,
    F: FnOnce() -> Result<T, String> + Send,
{
    let done = AtomicBool::new(false);
    let (result, samples) = std::thread::scope(|scope| {
        let worker = scope.spawn(|| {
            let result = match processor {
                Some(processor) => kiral_sys::affinity::pin_current_thread(processor)
                    .map_err(|error| format!("pinning to processor {processor}: {error}"))
                    .and_then(|()| work()),
                None => work(),
            };
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
        (worker.join().expect("a worker that did not panic"), samples)
    });
    Ok((result?, clock_of(samples)))
}

/// Where a run was made, for the log: the processor, its kind and its clock.
pub fn place(topology: &Topology, processor: Option<usize>, clock: &str) -> String {
    match processor {
        Some(processor) => format!(
            "CPU {processor} ({}, pinned){clock}",
            topology.kind(processor).map_or_else(|| "?".to_owned(), String::from)
        ),
        None => "unpinned".to_owned(),
    }
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

/// Prints times: how many, their median, their 99th percentile by the nearest
/// rank, and the longest.
pub fn print_times(what: &str, times: &[Duration]) {
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
