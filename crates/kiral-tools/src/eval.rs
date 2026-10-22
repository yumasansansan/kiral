// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later
//
// This file is part of KIRAL. The SPDX lines name its copyright holder and its
// license, in the machine-readable form of the REUSE specification: the GNU
// General Public License, version 3 or any later version
// (LICENSES/GPL-3.0-or-later.txt).

//! `kiral-tools eval`: the accuracy of a conversion against AJIMEE-Bench.
//!
//! Two conversions are measured for now. zenz's alone, as the evaluation of a
//! bare model takes it: zenz-v3's prompt, greedy decoding in single precision,
//! no dictionary, each item on its own, the items shared out among threads. And
//! Mozc's, as a user gets it: the reading typed in romaji into a Mozc of the
//! run's own, one key at a time, and converted with Space, its first candidate of
//! each segment, with no left context (Mozc takes none); the preedit typed is
//! checked to be the reading, so that a key the table reads otherwise stops the
//! run rather than measure another reading.
//!
//! The run prints Acc@1 and the character error rate, of all the items and of
//! those with and without left context; compares the run, item by item, with the
//! last run of the same label (or the results given), with McNemar's exact p;
//! and keeps its results for the next run.

use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use kiral_cpu::Features;
use kiral_dict::Dictionary;
use kiral_kernels::quantized::{Rotation, Scheme};
use kiral_lattice::Lattice;
use kiral_nn::gpt2::Precision;
use kiral_nn::zenz::{Converter, Stop, Zenz, prompt};

use crate::ajimee::{self, Item};
use crate::metrics::{count, least_error_rate, mcnemar};
use crate::mozc::{Mozc, Programs};
use crate::{arguments, results, romaji};

const USAGE: &str = "\
usage: kiral-tools eval (--model <directory> | --dict <file> | --mozc) [options]

The accuracy of a conversion against AJIMEE-Bench (JWTD_v2/v1): zenz alone
(zenz-v3's prompt, greedy decoding, no dictionary), a dictionary alone (each
reading's best path by Viterbi, no left context), or Mozc (the reading typed in
romaji, converted with Space, no left context).

  --model <directory>   zenz as Hugging Face publishes it: config.json,
                        model.safetensors, vocab.json, merges.txt and
                        special_tokens_map.json (a file in it names it too)
  --int8                zenz's products in integers: its weights quantized to
                        bytes when it is read, a scale for each output's group
                        of 64 inputs [single precision]
  --weight-bits <n>     the bits of a weight's integer, 2 to 8 [8]
  --rotate <order>      each block of this many inputs, a power of four,
                        rotated first by a regular Hadamard matrix [none]
  --fitted              each group of weights at the scale of least squared
                        error, not that of its greatest magnitude
  --dict <file>         a dictionary file, which kiral-tools dict build makes;
                        the run says its oracle too
  --mozc                Mozc, through mozc_emacs_helper, with a profile of the
                        run's own that it removes after (Linux)
  --mozc-helper <path>  the helper [mozc_emacs_helper]
  --mozc-server <path>  the server [/usr/lib/mozc/mozc_server]
  --bench ajimee        the benchmark, the one there is for now
  --items <file>        its items [data/ajimee/evaluation_items.json]
  --limit <n>           the first n items only
  --threads <n>         the threads zenz's items are shared among [all]
  --label <name>        the name the results are kept and compared under
                        [the model's directory's name, with -int8 and the
                        scheme's other choices for --int8, dict-<name>, or
                        mozc-<version>]
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

/// A conversion of an item, and how it ended.
#[derive(Debug, Clone)]
struct Converted {
    text: String,
    stop: &'static str,
}

fn run(arguments: impl IntoIterator<Item = OsString>) -> Result<(), String> {
    let options = arguments::parse(
        arguments,
        &[
            "model",
            "dict",
            "mozc-helper",
            "mozc-server",
            "bench",
            "items",
            "limit",
            "threads",
            "label",
            "against",
            "results",
            "weight-bits",
            "rotate",
        ],
        &[],
        &["mozc", "int8", "fitted", "help"],
    )
    .map_err(|message| format!("{message}\n\n{USAGE}"))?;
    if options.flag("help") {
        println!("{USAGE}");
        return Ok(());
    }
    if let Some(bench) = options.text("bench").filter(|bench| bench != "ajimee") {
        return Err(format!("no benchmark {bench}: the one there is is ajimee"));
    }
    let model = options.path("model");
    let systems =
        usize::from(model.is_some()) + usize::from(options.flag("mozc")) + usize::from(options.value("dict").is_some());
    if systems != 1 {
        return Err(format!("one of --model, --dict and --mozc\n\n{USAGE}"));
    }
    if scheme_of(&options)?.is_some() && model.is_none() {
        return Err(format!("--int8 and its scheme are for --model\n\n{USAGE}"));
    }
    let items = ajimee::load(options.path("items"), options.parsed::<usize>("limit")?)?;
    let kept = options
        .path("results")
        .unwrap_or_else(|| PathBuf::from("bench/results/ajimee"));

    let started = Instant::now();
    let (label, conversions) = if let Some(model) = model {
        with_zenz(&options, &model, &items)?
    } else if let Some(path) = options.path("dict") {
        with_dictionary(&options, &path, &items)?
    } else {
        with_mozc(&options, &items)?
    };
    report(
        &options,
        &label,
        &items,
        &conversions,
        started.elapsed().as_secs_f64(),
        &kept,
    )
}

/// zenz's conversions, and the label of their results.
fn with_zenz(options: &arguments::Options, model: &Path, items: &[Item]) -> Result<(String, Vec<Converted>), String> {
    let directory = Zenz::directory_of(model);
    let threads = match options.parsed::<usize>("threads")? {
        Some(threads) => threads.max(1),
        None => std::thread::available_parallelism().map_or(1, usize::from),
    }
    .min(items.len());
    let precision = scheme_of(options)?.map_or(Precision::Single, Precision::Integers);
    let label = options.text("label").unwrap_or_else(|| {
        let name = directory
            .file_name()
            .map_or_else(|| "zenz".to_owned(), |name| name.to_string_lossy().into_owned());
        match precision {
            Precision::Single => name,
            Precision::Integers(scheme) => format!("{name}-{}", scheme_label(&scheme)),
        }
    });
    let zenz = Zenz::load(&directory, precision).map_err(|error| format!("{}: {error}", directory.display()))?;
    println!(
        "AJIMEE-Bench JWTD_v2/v1, {} items: {label}, zenz alone (zenz-v3's prompt, greedy, {})",
        items.len(),
        precision_name(precision)
    );
    println!("{}; {threads} threads, kernels in {}", machine(), kernels());
    Ok((label, convert_with_zenz(&zenz, items, threads)?))
}

/// Mozc's conversions, and the label of their results.
fn with_mozc(options: &arguments::Options, items: &[Item]) -> Result<(String, Vec<Converted>), String> {
    let programs = Programs {
        helper: options
            .path("mozc-helper")
            .unwrap_or_else(|| Programs::default().helper),
        server: options
            .path("mozc-server")
            .unwrap_or_else(|| Programs::default().server),
    };
    let mut mozc = Mozc::start(&programs)?;
    let label = options
        .text("label")
        .unwrap_or_else(|| format!("mozc-{}", mozc.version()));
    println!(
        "AJIMEE-Bench JWTD_v2/v1, {} items: {label}, Mozc {} (typed in romaji, Space, no context)",
        items.len(),
        mozc.version()
    );
    println!("{}; one session, nothing committed", machine());
    Ok((label, convert_with_mozc(&mut mozc, items)?))
}

/// The dictionary's conversions alone -- each reading's best path by Viterbi,
/// with no context -- and the label of their results; and the oracle: how many
/// items have an accepted conversion that is a path of their lattice, of the
/// dictionary's words alone, and with characters of no reading as well.
fn with_dictionary(
    options: &arguments::Options,
    path: &Path,
    items: &[Item],
) -> Result<(String, Vec<Converted>), String> {
    let bytes = std::fs::read(path)
        .map_err(|error| format!("{}: {error} (kiral-tools dict build makes one)", path.display()))?;
    let dictionary = Dictionary::parse(&bytes).map_err(|error| format!("{}: {error}", path.display()))?;
    let metadata = dictionary.metadata();
    let label = options
        .text("label")
        .unwrap_or_else(|| format!("dict-{}", metadata.name));
    println!(
        "AJIMEE-Bench JWTD_v2/v1, {} items: {label}, the dictionary alone (Viterbi, no context)",
        items.len()
    );
    println!(
        "{}; {} readings, {} entries, {}",
        machine(),
        dictionary.readings(),
        dictionary.entries(),
        metadata.license
    );
    let mut conversions = Vec::with_capacity(items.len());
    let (mut known, mut any) = (0, 0);
    let mut missed = Vec::new();
    for item in items {
        let lattice = Lattice::new(&dictionary, &item.reading);
        let best = lattice.best(&dictionary);
        conversions.push(Converted {
            text: lattice.text(&best),
            stop: "dict",
        });
        if item.expected.iter().any(|expected| lattice.reaches(expected, true)) {
            known += 1;
        } else {
            missed.push(item.index.clone());
        }
        any += usize::from(item.expected.iter().any(|expected| lattice.reaches(expected, false)));
    }
    println!(
        "oracle: an accepted conversion is a path of the lattice for {known} of {} items, of words of the \
         dictionary alone ({:.1}%), and for {any} ({:.1}%) with characters of no reading; not of words alone: {}",
        items.len(),
        100.0 * count(known) / count(items.len()),
        100.0 * count(any) / count(items.len()),
        if missed.is_empty() {
            "none".to_owned()
        } else {
            missed.join(", ")
        }
    );
    Ok((label, conversions))
}

/// Prints the measures of the run and the items it changed, and keeps its
/// results.
fn report(
    options: &arguments::Options,
    label: &str,
    items: &[Item],
    conversions: &[Converted],
    seconds: f64,
    kept: &Path,
) -> Result<(), String> {
    let lines = items
        .iter()
        .zip(conversions)
        .map(|(item, converted)| line(item, converted))
        .collect::<Vec<_>>();
    println!("converted in {seconds:.1} s");
    println!();
    let (accuracy, error_rate) = summarize(&lines);
    print_stops(conversions);
    if let Some(against) = options.path("against").or_else(|| results::latest(kept, label)) {
        compare(&against, items, &lines)?;
    }
    std::fs::create_dir_all(kept).map_err(|error| format!("{}: {error}", kept.display()))?;
    let path = kept.join(format!(
        "{}-{}.tsv",
        results::timestamp(),
        results::label_for_files(label)
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

/// The processor and the levels it has, for the log.
fn machine() -> String {
    let processor = kiral_cpu::topology::processor_name().unwrap_or_else(|| std::env::consts::ARCH.to_owned());
    let levels = Features::detect()
        .levels()
        .iter()
        .map(|level| level.name())
        .collect::<Vec<_>>()
        .join(", ");
    format!("{processor}, levels {levels}")
}

/// The scheme of quantization the options ask for, `--int8` and the choices
/// that go with it; none for single precision.
///
/// # Errors
///
/// Where a choice is out of its range.
pub fn scheme_of(options: &arguments::Options) -> Result<Option<Scheme>, String> {
    let bits = options.parsed::<u32>("weight-bits")?;
    let order = options.parsed::<usize>("rotate")?;
    let fitted = options.flag("fitted");
    if !options.flag("int8") && bits.is_none() && order.is_none() && !fitted {
        return Ok(None);
    }
    let weight_bits = bits.unwrap_or(8);
    if !(2..=8).contains(&weight_bits) {
        return Err(format!("--weight-bits {weight_bits}: from 2 to 8"));
    }
    let rotation = match order {
        None | Some(1) => Rotation::None,
        Some(order) if order.is_power_of_two() && order.trailing_zeros().is_multiple_of(2) => Rotation::Hadamard(order),
        Some(order) => return Err(format!("--rotate {order}: not a power of four")),
    };
    Ok(Some(Scheme {
        weight_bits,
        rotation,
        fitted,
    }))
}

/// A scheme's name in a label: `int8`, and what it chooses otherwise than
/// [`Scheme::INT8`].
pub fn scheme_label(scheme: &Scheme) -> String {
    let bits = (scheme.weight_bits != 8).then(|| format!("-w{}", scheme.weight_bits));
    let rotation = match scheme.rotation {
        Rotation::None => None,
        Rotation::Hadamard(order) => Some(format!("-h{order}")),
    };
    let fitted = scheme.fitted.then_some("-fit");
    format!(
        "int8{}{}{}",
        bits.unwrap_or_default(),
        rotation.unwrap_or_default(),
        fitted.unwrap_or_default()
    )
}

/// What a precision computes, for the log.
pub fn precision_name(precision: Precision) -> String {
    match precision {
        Precision::Single => "single precision".to_owned(),
        Precision::Integers(scheme) => format!(
            "products in integers: weights of {} bits{}{}, a scale for each output's group of 64 inputs{}, inputs \
             in bytes with a zero point for each token's group",
            scheme.weight_bits,
            if scheme.weight_bits == 8 {
                ""
            } else {
                " (the output layer's of 8)"
            },
            match scheme.rotation {
                Rotation::None => String::new(),
                Rotation::Hadamard(order) =>
                    format!(", both rotated in blocks of {order} by a regular Hadamard matrix"),
            },
            if scheme.fitted { " of least squared error" } else { "" }
        ),
    }
}

/// The form of the kernels in effect, for the log: the form of the products in
/// integers, the other kernels being portable alone for now.
pub fn kernels() -> String {
    match kiral_kernels::form_error() {
        None => format!("the {} form", kiral_kernels::form()),
        Some(error) => format!("the {} form ({error})", kiral_kernels::form()),
    }
}

/// The most tokens a conversion of an item may take: three for each character
/// of its reading, as no character of a conversion takes more, and some over.
pub fn limit(item: &Item) -> usize {
    3 * item.reading.chars().count() + 8
}

/// zenz's conversions of the items, in their order, the items shared out among
/// the threads.
fn convert_with_zenz(zenz: &Zenz, items: &[Item], threads: usize) -> Result<Vec<Converted>, String> {
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
    let mut conversions: Vec<Option<Result<Converted, String>>> = vec![None; items.len()];
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
                        let outcome = converter
                            .convert(&item.reading, &item.context, limit(item))
                            .map(|conversion| Converted {
                                text: conversion.text,
                                stop: match conversion.stop {
                                    Stop::End => "end",
                                    Stop::Tag => "tag",
                                    Stop::Limit => "limit",
                                },
                            })
                            .map_err(|error| format!("item {}: {error}", item.index));
                        results.push((index, outcome));
                        let finished = done.fetch_add(1, Ordering::Relaxed) + 1;
                        if finished.is_multiple_of(20) || finished == items.len() {
                            eprint!("\r{finished} of {} items", items.len());
                        }
                    }
                })
            })
            .collect::<Vec<_>>();
        for worker in workers {
            for (index, outcome) in worker.join().expect("a worker that did not panic") {
                conversions[index] = Some(outcome);
            }
        }
    });
    eprintln!();
    conversions
        .into_iter()
        .map(|outcome| outcome.expect("every item converted"))
        .collect()
}

/// Types an item into Mozc and checks that the preedit is its reading.
///
/// # Errors
///
/// Where the reading has a character the romaji table has no keys for, or the
/// preedit typed is not the reading.
pub fn type_item(mozc: &mut Mozc, item: &Item, times: &mut Vec<std::time::Duration>) -> Result<(), String> {
    let keys =
        romaji::keys(&item.reading).map_err(|character| format!("item {}: no keys type {character:?}", item.index))?;
    let preedit = mozc.type_keys(&keys, times)?;
    let expected = romaji::preedit(&item.reading);
    if preedit != expected {
        return Err(format!(
            "item {}: typing {keys} made the preedit {preedit}, not {expected}",
            item.index
        ));
    }
    Ok(())
}

/// Mozc's conversions of the items, in their order: each typed, converted and
/// cancelled.
fn convert_with_mozc(mozc: &mut Mozc, items: &[Item]) -> Result<Vec<Converted>, String> {
    let mut times = Vec::new();
    let mut conversions = Vec::with_capacity(items.len());
    for (number, item) in items.iter().enumerate() {
        type_item(mozc, item, &mut times)?;
        let (text, _) = mozc
            .convert()
            .map_err(|error| format!("item {}: {error}", item.index))?;
        mozc.cancel().map_err(|error| format!("item {}: {error}", item.index))?;
        conversions.push(Converted { text, stop: "mozc" });
        if (number + 1).is_multiple_of(20) || number + 1 == items.len() {
            eprint!("\r{} of {} items", number + 1, items.len());
        }
    }
    eprintln!();
    Ok(conversions)
}

fn line(item: &Item, converted: &Converted) -> results::Line {
    results::Line {
        index: item.index.clone(),
        context: !item.context.is_empty(),
        correct: item.expected.contains(&converted.text),
        error_rate: least_error_rate(&item.expected, &converted.text),
        stop: converted.stop.to_owned(),
        conversion: converted.text.clone(),
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

fn print_stops(conversions: &[Converted]) {
    let stops = |stop: &str| conversions.iter().filter(|converted| converted.stop == stop).count();
    let (ended, tagged, limited) = (stops("end"), stops("tag"), stops("limit"));
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
