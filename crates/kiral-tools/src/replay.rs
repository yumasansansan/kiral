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
//! Three are timed. Mozc, through `mozc_emacs_helper`, the readings typed in
//! romaji, the time of a key that of its answer -- the helper, Mozc's IPC and
//! its server included -- the first item typed once before the times are
//! taken, while the server loads its dictionary. KIRAL's dictionary alone, the
//! readings typed a kana at a time, each prefix converted from its lattice up:
//! the time of a key that of building the lattice, finding its best path and
//! making its text, as an engine that kept nothing between keys would take. And
//! KIRAL's engine, the dictionary's drafts refined by zenz on a worker
//! ([`kiral_engine::engine`]): the readings typed a kana at a time, a kana every
//! interval as a typist types, each key's time on the input thread that of its
//! draft, then the time until its refinement comes, where it comes before the
//! next key; at the end of each item an explicit conversion, which waits for its
//! refinement up to the deadline, and what it shows checked against the
//! benchmark.
//!
//! A run is made on each processor asked for, the thread that types pinned to
//! it before anything starts, so that Mozc's helper and server, which inherit
//! the pin, run there too, and the engine's worker on a processor of its own
//! where one is given; the clocks of the processors are read every 20 ms while
//! the run goes on, and given with the times ([`crate::pinned`]).

use std::ffi::OsString;
use std::path::Path;
use std::process::ExitCode;
use std::time::{Duration, Instant};

use kiral_cpu::topology::{self, Topology};
use kiral_dict::Dictionary;
use kiral_engine::engine::{Engine, Settings};
use kiral_lattice::Lattice;

use crate::ajimee::{self, Item};
use crate::arguments;
use crate::eval::{kernels, precision_name, type_item};
use crate::metrics::count;
use crate::model::Source;
use crate::mozc::{Mozc, Programs};
use crate::pinned::{place, print_times, processors, run_on, run_on_beside};

const USAGE: &str = "\
usage: kiral-tools replay (--dict <file> [--model <path>] | --mozc) [options]

The time an input method takes for each key of AJIMEE-Bench's readings, and for
each conversion (Space): the median, the 99th percentile and the longest, on
each processor asked for, with the clock it ran at.

  --dict <file>         a dictionary file: the readings typed a kana at a time,
                        each prefix converted by the dictionary alone
  --model <path>        with --dict, KIRAL's engine: the dictionary's drafts
                        refined by zenz -- as Hugging Face publishes it, or a
                        model file -- on a worker, a kana typed every interval,
                        and the time until each key's refinement comes besides
  --interval <ms>       the time between two kana [100]
  --deadline <ms>       how long a conversion waits for its refinement [30]
  --mozc                Mozc, through mozc_emacs_helper, the readings typed in
                        romaji, with a profile of the run's own that it removes
                        after (Linux)
  --mozc-helper <path>  the helper [mozc_emacs_helper]
  --mozc-server <path>  the server [/usr/lib/mozc/mozc_server]
  --cpu <processor>     a processor to pin the run to, again for another run:
                        p<n> the n-th performance core, e<n> the n-th
                        efficiency core, or the system's number [none: unpinned];
                        <processor>:<processor> pins the engine's worker to the
                        second [unpinned]
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
        &[
            "dict",
            "model",
            "interval",
            "deadline",
            "mozc-helper",
            "mozc-server",
            "items",
            "limit",
        ],
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
    if let (Some(model), Some(dictionary_path)) = (options.path("model"), &dictionary_path) {
        return replay_engine(&options, &model, dictionary_path);
    }
    if options.value("model").is_some() || options.value("interval").is_some() || options.value("deadline").is_some() {
        return Err(format!(
            "--model, --interval and --deadline are for the engine, with --dict\n\n{USAGE}"
        ));
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

/// The times of a run of the engine, and what its conversions showed.
#[derive(Debug, Default)]
struct EngineTimes {
    /// Each key's time on the input thread: its draft.
    keys: Vec<Duration>,
    /// The time from a key to its refinement, where it came before the next.
    refinements: Vec<Duration>,
    /// The worker's time over a key's job, where its refinement came.
    jobs: Vec<Duration>,
    /// The keys whose refinement had not come by the next.
    passed: usize,
    /// Each conversion's wait.
    conversions: Vec<Duration>,
    /// The conversions refined by their deadline.
    refined: usize,
    /// The conversions the benchmark accepts.
    correct: usize,
}

/// `replay --model --dict`: types every item into KIRAL's engine on each
/// processor asked for, its worker on the processor paired with it.
fn replay_engine(options: &arguments::Options, model: &Path, dictionary_path: &Path) -> Result<(), String> {
    let items = ajimee::load(options.path("items"), options.parsed::<usize>("limit")?)?;
    let interval = Duration::from_millis(options.parsed::<u64>("interval")?.unwrap_or(100));
    let deadline = Duration::from_millis(options.parsed::<u64>("deadline")?.unwrap_or(30));
    let topology = Topology::detect();
    let runs = paired_processors(options, &topology)?;
    let source = Source::open(model)?;
    let zenz = source.zenz(None)?;
    let bytes = std::fs::read(dictionary_path).map_err(|error| format!("{}: {error}", dictionary_path.display()))?;
    let dictionary = Dictionary::parse(&bytes).map_err(|error| format!("{}: {error}", dictionary_path.display()))?;
    let name = topology::processor_name().unwrap_or_else(|| std::env::consts::ARCH.to_owned());
    let defaults = Settings::default();
    println!(
        "replay: AJIMEE-Bench JWTD_v2/v1, {} items typed a kana every {} ms into KIRAL's engine: the dictionary {}'s \
         drafts refined by {} ({}), a round a key, up to {} rounds for a conversion, which waits {} ms; {name}, \
         kernels in {}",
        items.len(),
        interval.as_millis(),
        dictionary.metadata().name,
        source.name(),
        precision_name(zenz.model.precision()),
        defaults.conversion_rounds,
        deadline.as_millis(),
        kernels()
    );
    for (processor, worker) in runs {
        let settings = Settings {
            deadline,
            worker_processor: worker,
            // Room for the longest prompt, and a draft of four tokens a
            // character, the most a character takes.
            capacity: zenz.model.config().positions.min(512),
            ..defaults.clone()
        };
        let (times, clock, worker_clock) = run_on_beside(processor, worker, || {
            Ok(type_into_engine(&zenz, &dictionary, &items, &settings, interval))
        })?;
        let worker_place = match worker {
            Some(_) => format!("the worker on {}", place(&topology, worker, &worker_clock)),
            None => "the worker unpinned".to_owned(),
        };
        println!("  typing on {}, {worker_place}", place(&topology, processor, &clock));
        print_times("kana (the input thread: the draft)", &times.keys);
        let keys = times.refinements.len() + times.passed;
        print_times(
            &format!(
                "refinements of a kana, from the key ({} of {keys} came before the next key)",
                times.refinements.len()
            ),
            &times.refinements,
        );
        print_times("the worker's jobs of a kana", &times.jobs);
        print_times(
            &format!(
                "conversions (Space), their wait ({} of {} refined by the deadline)",
                times.refined,
                times.conversions.len()
            ),
            &times.conversions,
        );
        println!(
            "    the conversions shown: Acc@1 {:.1} ({} of {})",
            100.0 * count(times.correct) / count(items.len()),
            times.correct,
            items.len()
        );
    }
    Ok(())
}

/// A run of the engine: the processor to type on, and the worker's.
type Run = (Option<usize>, Option<usize>);

/// The runs `--cpu` asks for: the processor to type on, and the worker's after
/// a colon; one run unpinned where there are none.
fn paired_processors(options: &arguments::Options, topology: &Topology) -> Result<Vec<Run>, String> {
    let named = |name: &str| {
        topology
            .processor(name)
            .ok_or_else(|| format!("--cpu {name}: no such processor in {topology:?}"))
    };
    let runs = options
        .values("cpu")
        .iter()
        .map(|value| match value.split_once(':') {
            Some((typing, worker)) => Ok((Some(named(typing)?), Some(named(worker)?))),
            None => Ok((Some(named(value)?), None)),
        })
        .collect::<Result<Vec<_>, String>>()?;
    Ok(if runs.is_empty() { vec![(None, None)] } else { runs })
}

/// Types every item into the engine, the first once before, while the model's
/// and the dictionary's pages come in.
fn type_into_engine(
    zenz: &kiral_nn::zenz::Zenz<'_>,
    dictionary: &Dictionary<'_>,
    items: &[Item],
    settings: &Settings,
    interval: Duration,
) -> EngineTimes {
    Engine::run(zenz, dictionary, settings, |engine| {
        type_item_into(engine, &items[0], interval, &mut EngineTimes::default());
        let mut times = EngineTimes::default();
        for item in items {
            type_item_into(engine, item, interval, &mut times);
        }
        times
    })
}

/// Types an item a kana every interval, waiting between keys for each key's
/// refinement, then converts it.
fn type_item_into(engine: &mut Engine<'_>, item: &Item, interval: Duration, times: &mut EngineTimes) {
    let kana: Vec<char> = item.reading.chars().collect();
    let mut prefix = String::new();
    for &character in &kana {
        prefix.push(character);
        let typed = Instant::now();
        std::hint::black_box(engine.compose(&prefix, &item.context));
        times.keys.push(typed.elapsed());
        let next = typed + interval;
        if engine.wait_until(next) {
            times.refinements.push(typed.elapsed());
            times.jobs.extend(engine.answer().map(|answer| answer.took));
            if let Some(left) = next.checked_duration_since(Instant::now()) {
                std::thread::sleep(left);
            }
        } else {
            times.passed += 1;
        }
    }
    let converting = Instant::now();
    let shown = engine.convert().to_owned();
    times.conversions.push(converting.elapsed());
    times.refined += usize::from(engine.answer().is_some());
    times.correct += usize::from(item.expected.contains(&shown));
}
