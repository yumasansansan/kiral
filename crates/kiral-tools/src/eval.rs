// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later
//
// This file is part of KIRAL. The SPDX lines name its copyright holder and its
// license, in the machine-readable form of the REUSE specification: the GNU
// General Public License, version 3 or any later version
// (LICENSES/GPL-3.0-or-later.txt).

//! `kiral-tools eval`: the accuracy of a conversion against AJIMEE-Bench.
//!
//! The conversion measured is zenz's alone, as the evaluation of a bare model
//! takes it: zenz-v3's prompt, greedy decoding in single precision, no
//! dictionary. Each item is converted on its own, the items shared out among
//! threads. The run prints Acc@1 and the character error rate, of all the items
//! and of those with and without left context; compares the run, item by item,
//! with the last run of the same label (or the results given), with McNemar's
//! exact p; and keeps its results for the next run.

use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use kiral_cpu::{Choice, Features, Request};
use kiral_nn::zenz::{Conversion, Converter, Stop, Zenz, prompt};

use crate::ajimee::{self, Item};
use crate::metrics::{count, least_error_rate, mcnemar};
use crate::{arguments, results};

const USAGE: &str = "\
usage: kiral-tools eval --model <directory> [options]

The accuracy of zenz alone against AJIMEE-Bench (JWTD_v2/v1): zenz-v3's prompt,
greedy decoding in single precision, no dictionary.

  --model <directory>   zenz as Hugging Face publishes it: config.json,
                        model.safetensors, vocab.json, merges.txt and
                        special_tokens_map.json (a file in it names it too)
  --bench ajimee        the benchmark, the one there is for now
  --items <file>        its items [data/ajimee/evaluation_items.json]
  --limit <n>           the first n items only
  --threads <n>         the threads the items are shared among [all]
  --label <name>        the name the results are kept and compared under
                        [the model's directory's name]
  --against <file>      the results to compare with [the label's last]
  --results <directory> where results are kept [bench/results/ajimee]";

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

fn run(arguments: impl IntoIterator<Item = OsString>) -> Result<(), String> {
    let options = arguments::parse(
        arguments,
        &[
            "model", "bench", "items", "limit", "threads", "label", "against", "results",
        ],
        &["help"],
    )
    .map_err(|message| format!("{message}\n\n{USAGE}"))?;
    if options.flag("help") {
        println!("{USAGE}");
        return Ok(());
    }
    if let Some(bench) = options.text("bench").filter(|bench| bench != "ajimee") {
        return Err(format!("no benchmark {bench}: the one there is is ajimee"));
    }
    let model = options.path("model").ok_or_else(|| format!("no --model\n\n{USAGE}"))?;
    let directory = Zenz::directory_of(&model);
    let items_path = options.path("items").unwrap_or_else(|| PathBuf::from(ajimee::ITEMS));
    let bytes = std::fs::read(&items_path).map_err(|error| {
        format!(
            "{}: {error} (scripts/fetch_datasets.py fetch ajimee fetches it)",
            items_path.display()
        )
    })?;
    let mut items = ajimee::read(&bytes)?;
    if let Some(limit) = options.parsed::<usize>("limit")? {
        items.truncate(limit);
    }
    if items.is_empty() {
        return Err("no items to convert".to_owned());
    }
    let threads = match options.parsed::<usize>("threads")? {
        Some(threads) => threads.max(1),
        None => std::thread::available_parallelism().map_or(1, usize::from),
    }
    .min(items.len());
    let label = options.text("label").unwrap_or_else(|| {
        directory
            .file_name()
            .map_or_else(|| "zenz".to_owned(), |name| name.to_string_lossy().into_owned())
    });
    let kept = options
        .path("results")
        .unwrap_or_else(|| PathBuf::from("bench/results/ajimee"));
    let against = options.path("against").or_else(|| results::latest(&kept, &label));

    let zenz = Zenz::load(&directory).map_err(|error| format!("{}: {error}", directory.display()))?;
    println!(
        "AJIMEE-Bench JWTD_v2/v1, {} items: {label}, zenz alone (zenz-v3's prompt, greedy, single precision)",
        items.len()
    );
    println!("{}", machine(threads));
    let started = Instant::now();
    let conversions = convert_all(&zenz, &items, threads)?;
    let seconds = started.elapsed().as_secs_f64();
    let lines = items
        .iter()
        .zip(&conversions)
        .map(|(item, conversion)| line(item, conversion))
        .collect::<Vec<_>>();
    println!("converted in {seconds:.1} s");
    println!();
    let (accuracy, error_rate) = summarize(&lines);
    print_stops(&conversions);
    if let Some(against) = against {
        compare(&against, &items, &lines)?;
    }
    std::fs::create_dir_all(&kept).map_err(|error| format!("{}: {error}", kept.display()))?;
    let path = kept.join(format!(
        "{}-{}.tsv",
        results::timestamp(),
        results::label_for_files(&label)
    ));
    let title = format!(
        "AJIMEE-Bench JWTD_v2/v1, {} items: {label}, Acc@1 {accuracy:.1}, CER {error_rate:.2}",
        lines.len()
    );
    results::write(&path, &title, &lines).map_err(|error| format!("{}: {error}", path.display()))?;
    println!();
    println!("the results are in {}", path.display());
    Ok(())
}

/// The processor and the threads the run is on, for its log.
fn machine(threads: usize) -> String {
    let processor = std::fs::read_to_string("/proc/cpuinfo")
        .ok()
        .and_then(|info| {
            info.lines().find_map(|line| {
                line.strip_prefix("model name")
                    .map(|rest| rest.trim_start_matches([' ', '\t', ':']).to_owned())
            })
        })
        .unwrap_or_else(|| std::env::consts::ARCH.to_owned());
    let features = Features::detect();
    let levels = features
        .levels()
        .iter()
        .map(|level| level.name())
        .collect::<Vec<_>>()
        .join(", ");
    let choice = Request::from_environment().map(|request| Choice::new(features, request));
    let kernels = match choice {
        Ok(_) => "the portable form".to_owned(),
        Err(error) => format!("the portable form ({error})"),
    };
    format!("{processor}, levels {levels}; {threads} threads, kernels in {kernels}")
}

/// The most tokens a conversion of an item may take: three for each character
/// of its reading, as no character of a conversion takes more, and some over.
fn limit(item: &Item) -> usize {
    3 * item.reading.chars().count() + 8
}

/// The conversions of the items, in their order, the items shared out among
/// the threads.
fn convert_all(zenz: &Zenz, items: &[Item], threads: usize) -> Result<Vec<Conversion>, String> {
    // Room for the longest prompt and its conversion.
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
    let next = AtomicUsize::new(0);
    let done = AtomicUsize::new(0);
    let mut conversions: Vec<Option<Result<Conversion, String>>> = vec![None; items.len()];
    std::thread::scope(|scope| {
        let workers = (0..threads)
            .map(|_| {
                scope.spawn(|| {
                    let mut converter = Converter::new(zenz, capacity);
                    let mut results = Vec::new();
                    loop {
                        let index = next.fetch_add(1, Ordering::Relaxed);
                        let Some(item) = items.get(index) else {
                            break results;
                        };
                        let conversion = converter
                            .convert(&item.reading, &item.context, limit(item))
                            .map_err(|error| format!("item {}: {error}", item.index));
                        results.push((index, conversion));
                        let finished = done.fetch_add(1, Ordering::Relaxed) + 1;
                        if finished.is_multiple_of(20) || finished == items.len() {
                            eprint!("\r{finished} of {} items", items.len());
                        }
                    }
                })
            })
            .collect::<Vec<_>>();
        for worker in workers {
            for (index, conversion) in worker.join().expect("a worker that did not panic") {
                conversions[index] = Some(conversion);
            }
        }
    });
    eprintln!();
    conversions
        .into_iter()
        .map(|conversion| conversion.expect("every item converted"))
        .collect()
}

fn line(item: &Item, conversion: &Conversion) -> results::Line {
    results::Line {
        index: item.index.clone(),
        context: !item.context.is_empty(),
        correct: item.expected.contains(&conversion.text),
        error_rate: least_error_rate(&item.expected, &conversion.text),
        stop: match conversion.stop {
            Stop::End => "end",
            Stop::Tag => "tag",
            Stop::Limit => "limit",
        }
        .to_owned(),
        conversion: conversion.text.clone(),
    }
}

/// Acc@1 and the character error rate of lines, both in percent: the share of
/// conversions accepted, and the mean of the items' least error rates.
fn measures<'a>(lines: impl Iterator<Item = &'a results::Line>) -> (usize, f64, f64) {
    let (items, correct, error) = lines.fold((0, 0, 0.0), |(items, correct, error), line| {
        (items + 1, correct + usize::from(line.correct), error + line.error_rate)
    });
    if items == 0 {
        return (0, f64::NAN, f64::NAN);
    }
    (
        items,
        100.0 * count(correct) / count(items),
        100.0 * error / count(items),
    )
}

/// Prints the measures of all the items and of those with and without left
/// context, and returns those of all.
fn summarize(lines: &[results::Line]) -> (f64, f64) {
    println!("                   items   Acc@1     CER");
    let (items, accuracy, error_rate) = measures(lines.iter());
    println!("  all              {items:>5}  {accuracy:>6.1}  {error_rate:>6.2}");
    for (name, context) in [("with context", true), ("without context", false)] {
        let (items, accuracy, error_rate) = measures(lines.iter().filter(|line| line.context == context));
        if items > 0 {
            println!("  {name:<15}  {items:>5}  {accuracy:>6.1}  {error_rate:>6.2}");
        }
    }
    (accuracy, error_rate)
}

fn print_stops(conversions: &[Conversion]) {
    let stops = |stop: Stop| conversions.iter().filter(|conversion| conversion.stop == stop).count();
    let (ended, tagged, limited) = (stops(Stop::End), stops(Stop::Tag), stops(Stop::Limit));
    if tagged + limited > 0 {
        println!("  conversions ended by the end of text {ended}, by a tag {tagged}, by the limit {limited}");
    }
}

/// Prints the items that changed since the results of `path`, and McNemar's p
/// of the change.
fn compare(path: &Path, items: &[Item], lines: &[results::Line]) -> Result<(), String> {
    let previous = results::read(path)?;
    let mut better = Vec::new();
    let mut worse = Vec::new();
    let mut changed = 0;
    let mut paired = 0;
    for (item, line) in items.iter().zip(lines) {
        let Some(before) = previous.iter().find(|before| before.index == line.index) else {
            continue;
        };
        paired += 1;
        changed += usize::from(before.conversion != line.conversion);
        match (before.correct, line.correct) {
            (false, true) => better.push((item, before, line)),
            (true, false) => worse.push((item, before, line)),
            _ => {}
        }
    }
    println!();
    println!(
        "against {} ({paired} items paired): +{} / -{} items, McNemar p = {:.3}; {changed} conversions changed",
        path.display(),
        better.len(),
        worse.len(),
        mcnemar(better.len(), worse.len())
    );
    for (sign, list) in [('+', &better), ('-', &worse)] {
        for (item, before, line) in list {
            println!(
                "  {sign} {:>5}  {} -> {}  (accepted: {})",
                item.index,
                before.conversion,
                line.conversion,
                item.expected.join(" / ")
            );
        }
    }
    Ok(())
}
