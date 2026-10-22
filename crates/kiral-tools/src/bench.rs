// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later
//
// This file is part of KIRAL. The SPDX lines name its copyright holder and its
// license, in the machine-readable form of the REUSE specification: the GNU
// General Public License, version 3 or any later version
// (LICENSES/GPL-3.0-or-later.txt).

//! `kiral-tools bench`: the time of KIRAL's computations, pinned.
//!
//! `bench model` converts AJIMEE-Bench's readings with zenz alone, greedily, as
//! `eval` does, one item after another on one thread pinned to each processor
//! asked for, and times each prompt -- its tokens fed at once, as a new
//! composition's are -- and each step of decoding after it, a token fed: the
//! median, the 99th percentile and the longest, with the clock
//! ([`crate::pinned`]). The first item is converted once before the times are
//! taken, while the weights come into the caches.

use std::ffi::OsString;
use std::process::ExitCode;
use std::time::{Duration, Instant};

use kiral_cpu::topology::{self, Topology};
use kiral_kernels::vector::argmax;
use kiral_nn::gpt2::{Precision, Session};
use kiral_nn::zenz::{Zenz, prompt, tag_in};

use crate::ajimee::{self, Item};
use crate::arguments;
use crate::eval::{limit, precision_name, scheme_of};
use crate::pinned::{place, print_times, processors, run_on};

const USAGE: &str = "\
usage: kiral-tools bench model --model <directory> [options]

The time zenz takes to convert AJIMEE-Bench's readings greedily, one item after
another on one thread: each prompt fed at once, and each step of decoding after
it; the median, the 99th percentile and the longest, on each processor asked
for, with the clock it ran at.

  --model <directory>   zenz as Hugging Face publishes it
  --int8                zenz's products in integers [single precision]
  --weight-bits <n>     the bits of a weight's integer, 2 to 8 [8]
  --rotate <order>      each block of this many inputs, a power of four,
                        rotated first by a regular Hadamard matrix [none]
  --fitted              each group of weights at the scale of least squared
                        error, not that of its greatest magnitude
  --cpu <processor>     a processor to pin the run to, again for another run:
                        p<n> the n-th performance core, e<n> the n-th
                        efficiency core, or the system's number [none: unpinned]
  --items <file>        the items [data/ajimee/evaluation_items.json]
  --limit <n>           the first n items only";

/// Runs the command.
pub fn main(mut arguments: impl Iterator<Item = OsString>) -> ExitCode {
    let what = arguments.next().map(|what| what.to_string_lossy().into_owned());
    let result = match what.as_deref() {
        Some("model") => model(arguments),
        Some("help" | "--help") => {
            println!("{USAGE}");
            Ok(())
        }
        _ => Err(format!("nothing to time\n\n{USAGE}")),
    };
    match result {
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
    prompts: Vec<Duration>,
    prompt_tokens: Vec<usize>,
    steps: Vec<Duration>,
}

fn model(arguments: impl Iterator<Item = OsString>) -> Result<(), String> {
    let options = arguments::parse(
        arguments,
        &["model", "items", "limit", "weight-bits", "rotate"],
        &["cpu"],
        &["int8", "fitted", "help"],
    )
    .map_err(|message| format!("{message}\n\n{USAGE}"))?;
    if options.flag("help") {
        println!("{USAGE}");
        return Ok(());
    }
    let model = options.path("model").ok_or_else(|| format!("no --model\n\n{USAGE}"))?;
    let items = ajimee::load(options.path("items"), options.parsed::<usize>("limit")?)?;
    let topology = Topology::detect();
    let processors = processors(&options, &topology)?;
    let precision = scheme_of(&options)?.map_or(Precision::Single, Precision::Integers);
    let directory = Zenz::directory_of(&model);
    let zenz = Zenz::load(&directory, precision).map_err(|error| format!("{}: {error}", directory.display()))?;
    let name = topology::processor_name().unwrap_or_else(|| std::env::consts::ARCH.to_owned());
    println!(
        "bench model: AJIMEE-Bench JWTD_v2/v1, {} items converted greedily by {}, {}; {name}",
        items.len(),
        directory
            .file_name()
            .map_or_else(String::new, |name| name.to_string_lossy().into_owned()),
        precision_name(precision)
    );
    for processor in processors {
        let (times, clock) = run_on(processor, || convert_all(&zenz, &items))?;
        println!("  {}", place(&topology, processor, &clock));
        let mut lengths = times.prompt_tokens.clone();
        lengths.sort_unstable();
        print_times(
            &format!(
                "prompts ({} tokens at the median)",
                lengths.get(lengths.len() / 2).copied().unwrap_or(0)
            ),
            &times.prompts,
        );
        print_times("steps of decoding (a token each)", &times.steps);
    }
    Ok(())
}

/// Converts every item, the first once more before, and times each prompt and
/// each step of decoding.
fn convert_all(zenz: &Zenz, items: &[Item]) -> Result<Times, String> {
    let mut tokens = Vec::new();
    let capacity = items
        .iter()
        .map(|item| {
            tokens.clear();
            zenz.tokenizer
                .encode(&prompt(&item.reading, &item.context), &mut tokens);
            tokens.len() + limit(item)
        })
        .max()
        .unwrap_or(1)
        .min(zenz.model.config().positions);
    let mut session = Session::new(&zenz.model, capacity);
    let mut logits = vec![0.0f32; zenz.model.config().vocabulary];
    let mut warming = Times::default();
    if let Some(item) = items.first() {
        convert(zenz, &mut session, &mut logits, item, &mut warming)?;
    }
    let mut times = Times::default();
    for item in items {
        convert(zenz, &mut session, &mut logits, item, &mut times)?;
    }
    Ok(times)
}

/// Converts an item greedily, as `eval` does, and times its prompt and each
/// step of decoding.
fn convert(
    zenz: &Zenz,
    session: &mut Session<'_>,
    logits: &mut [f32],
    item: &Item,
    times: &mut Times,
) -> Result<(), String> {
    let failed = |error: kiral_nn::Error| format!("item {}: {error}", item.index);
    let end = zenz.tokenizer.specials().end;
    let mut tokens = Vec::new();
    zenz.tokenizer
        .encode(&prompt(&item.reading, &item.context), &mut tokens);
    session.truncate(0);
    let started = Instant::now();
    session.feed(&tokens, logits).map_err(failed)?;
    times.prompts.push(started.elapsed());
    times.prompt_tokens.push(tokens.len());
    let mut written = Vec::new();
    for _ in 0..limit(item) {
        let Some(next) = argmax(logits)
            .and_then(|next| u32::try_from(next).ok())
            .filter(|&next| next != end)
        else {
            break;
        };
        written.extend_from_slice(zenz.tokenizer.piece(next));
        if tag_in(&written).is_some() {
            break;
        }
        let started = Instant::now();
        session.feed(&[next], logits).map_err(failed)?;
        times.steps.push(started.elapsed());
    }
    Ok(())
}
