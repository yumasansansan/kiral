// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later
//
// This file is part of KIRAL. The SPDX lines name its copyright holder and its
// license, in the machine-readable form of the REUSE specification: the GNU
// General Public License, version 3 or any later version
// (LICENSES/GPL-3.0-or-later.txt).

//! `kiral-tools agree`: how faithful zenz's products in integers are to single
//! precision, by what checking a draft depends on -- the most likely next token
//! at each position.
//!
//! Each item's prompt and single precision's greedy conversion of it are fed to
//! both forms of the model at once, and the most likely token after each
//! position compared: at the positions of the conversion -- the prompt's last
//! token, which predicts the conversion's first, to the conversion's last,
//! which predicts its end -- and at every position. A position where the two
//! differ is one where a draft that single precision accepts would be refused,
//! or the other way about.

use std::ffi::OsString;
use std::process::ExitCode;
use std::sync::atomic::{AtomicUsize, Ordering};

use kiral_kernels::quantized::Scheme;
use kiral_kernels::vector::argmax;
use kiral_nn::gpt2::{Precision, Session};
use kiral_nn::zenz::{Converter, Zenz, prompt};

use crate::ajimee::{self, Item};
use crate::arguments;
use crate::eval::{limit, precision_name, scheme_of};
use crate::metrics::count;
use crate::model::Source;

const USAGE: &str = "\
usage: kiral-tools agree --model <directory> [options]

The positions where zenz's products in integers and in single precision differ
in the most likely next token: each AJIMEE-Bench item's prompt and single
precision's greedy conversion fed to both, compared at the positions of the
conversion and at every position.

  --model <directory>   zenz as Hugging Face publishes it
  --quantized <file>    a model file, whose products in integers are compared
                        rather than those of the directory's weights quantized
                        when they are read
  --int8                the scheme of eval --int8, the one by default
  --weight-bits <n>     the bits of a weight's integer, 2 to 8 [8]
  --rotate <order>      each block of this many inputs, a power of four,
                        rotated first by a regular Hadamard matrix [none]
  --fitted              each group of weights at the scale of least squared
                        error, not that of its greatest magnitude
  --items <file>        the items [data/ajimee/evaluation_items.json]
  --limit <n>           the first n items only
  --threads <n>         the threads the items are shared among [all]";

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

/// What an item's positions show.
#[derive(Debug, Clone, Default)]
struct Compared {
    /// The positions of the conversion.
    conversion: usize,
    /// Those where the two differ.
    conversion_differ: usize,
    /// Every position.
    all: usize,
    /// Those where the two differ.
    all_differ: usize,
    /// Single precision's margin between its two most likely tokens, at each
    /// position of the conversion where the two differ.
    margins: Vec<f32>,
}

fn run(arguments: impl IntoIterator<Item = OsString>) -> Result<(), String> {
    let options = arguments::parse(
        arguments,
        &[
            "model",
            "quantized",
            "items",
            "limit",
            "threads",
            "weight-bits",
            "rotate",
        ],
        &[],
        &["fitted", "int8", "help"],
    )
    .map_err(|message| format!("{message}\n\n{USAGE}"))?;
    if options.flag("help") {
        println!("{USAGE}");
        return Ok(());
    }
    let model = options.path("model").ok_or_else(|| format!("no --model\n\n{USAGE}"))?;
    let items = ajimee::load(options.path("items"), options.parsed::<usize>("limit")?)?;
    let threads = match options.parsed::<usize>("threads")? {
        Some(threads) => threads.max(1),
        None => std::thread::available_parallelism().map_or(1, usize::from),
    }
    .min(items.len());
    let directory = Zenz::directory_of(&model);
    let single =
        Zenz::load(&directory, Precision::Single).map_err(|error| format!("{}: {error}", directory.display()))?;
    let quantized = options.path("quantized").map(|path| Source::open(&path)).transpose()?;
    let integers = match &quantized {
        Some(Source::File(..)) if scheme_of(&options)?.is_some() => {
            return Err("--quantized names a model file, whose scheme is its own".to_owned());
        }
        Some(source @ Source::File(..)) => source.zenz(None)?,
        Some(Source::Directory(path)) => return Err(format!("--quantized {}: not a model file", path.display())),
        None => Zenz::load(
            &directory,
            Precision::Integers(scheme_of(&options)?.unwrap_or(Scheme::INT8)),
        )
        .map_err(|error| format!("{}: {error}", directory.display()))?,
    };
    let precision = integers.model.precision();
    let from = quantized.as_ref().map_or_else(String::new, |source| {
        format!(", from the model file {}", source.path().display())
    });
    println!(
        "agree: AJIMEE-Bench JWTD_v2/v1, {} items, {}{from} against single precision; {threads} threads",
        items.len(),
        precision_name(precision)
    );
    let compared = compare_all(&single, &integers, &items, threads)?;
    let total = compared.iter().fold(Compared::default(), |mut total, item| {
        total.conversion += item.conversion;
        total.conversion_differ += item.conversion_differ;
        total.all += item.all;
        total.all_differ += item.all_differ;
        total.margins.extend_from_slice(&item.margins);
        total
    });
    let share = |differ: usize, of: usize| 100.0 * (1.0 - count(differ) / count(of.max(1)));
    println!(
        "  positions of the conversions: {} of {} agree ({:.2}%)",
        total.conversion - total.conversion_differ,
        total.conversion,
        share(total.conversion_differ, total.conversion)
    );
    println!(
        "  every position: {} of {} agree ({:.2}%)",
        total.all - total.all_differ,
        total.all,
        share(total.all_differ, total.all)
    );
    let differing: Vec<&str> = items
        .iter()
        .zip(&compared)
        .filter(|(_, compared)| compared.conversion_differ > 0)
        .map(|(item, _)| item.index.as_str())
        .collect();
    println!(
        "  items whose conversion has a position that differs: {} ({})",
        differing.len(),
        if differing.is_empty() {
            "none".to_owned()
        } else {
            differing.join(", ")
        }
    );
    let mut margins = total.margins;
    if !margins.is_empty() {
        margins.sort_by(f32::total_cmp);
        println!(
            "  single precision's margin between its two likeliest tokens where they differ: median {:.3}, \
             largest {:.3}",
            margins[margins.len() / 2],
            margins[margins.len() - 1]
        );
    }
    Ok(())
}

/// The comparison of every item, the items shared among threads.
fn compare_all(
    single: &Zenz<'_>,
    integers: &Zenz<'_>,
    items: &[Item],
    threads: usize,
) -> Result<Vec<Compared>, String> {
    let mut tokens = Vec::new();
    let capacity = items
        .iter()
        .map(|item| {
            tokens.clear();
            single
                .tokenizer
                .encode(&prompt(&item.reading, &item.context), &mut tokens);
            tokens.len() + limit(item)
        })
        .max()
        .unwrap_or(1)
        .min(single.model.config().positions);
    let next = AtomicUsize::new(0);
    let mut compared: Vec<Option<Result<Compared, String>>> = vec![None; items.len()];
    std::thread::scope(|scope| {
        let workers = (0..threads)
            .map(|_| {
                scope.spawn(|| {
                    let mut converter = Converter::new(single, capacity);
                    let mut results = Vec::new();
                    loop {
                        let index = next.fetch_add(1, Ordering::Relaxed);
                        let Some(item) = items.get(index) else {
                            break results;
                        };
                        let outcome = compare(&mut converter, single, integers, item, capacity)
                            .map_err(|error| format!("item {}: {error}", item.index));
                        results.push((index, outcome));
                    }
                })
            })
            .collect::<Vec<_>>();
        for worker in workers {
            for (index, outcome) in worker.join().expect("a worker that did not panic") {
                compared[index] = Some(outcome);
            }
        }
    });
    compared
        .into_iter()
        .map(|outcome| outcome.unwrap_or_else(|| Err("an item not compared".to_owned())))
        .collect()
}

/// An item compared: single precision's conversion, then its prompt and
/// conversion fed to both forms at once.
fn compare(
    converter: &mut Converter<'_>,
    single: &Zenz<'_>,
    integers: &Zenz<'_>,
    item: &Item,
    capacity: usize,
) -> Result<Compared, String> {
    let conversion = converter
        .convert(&item.reading, &item.context, limit(item))
        .map_err(|error| error.to_string())?;
    let sequence = &conversion.sequence;
    let vocabulary = single.model.config().vocabulary;
    let logits_of = |zenz: &Zenz<'_>| -> Result<Vec<f32>, String> {
        let mut logits = vec![0.0f32; sequence.len() * vocabulary];
        Session::new(&zenz.model, capacity)
            .feed_all(sequence, &mut logits)
            .map_err(|error| error.to_string())?;
        Ok(logits)
    };
    let exact = logits_of(single)?;
    let quantized = logits_of(integers)?;
    let mut compared = Compared::default();
    for (position, (exact, quantized)) in exact
        .chunks_exact(vocabulary)
        .zip(quantized.chunks_exact(vocabulary))
        .enumerate()
    {
        let differ = argmax(exact) != argmax(quantized);
        compared.all += 1;
        compared.all_differ += usize::from(differ);
        if position + 1 >= conversion.prompt_tokens {
            compared.conversion += 1;
            compared.conversion_differ += usize::from(differ);
            if differ {
                compared.margins.push(margin(exact));
            }
        }
    }
    Ok(compared)
}

/// The difference between the largest logit and the next.
fn margin(logits: &[f32]) -> f32 {
    let (mut first, mut second) = (f32::NEG_INFINITY, f32::NEG_INFINITY);
    for &logit in logits {
        if logit > first {
            second = first;
            first = logit;
        } else if logit > second {
            second = logit;
        }
    }
    first - second
}
