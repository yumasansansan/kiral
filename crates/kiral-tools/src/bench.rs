// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later
//
// This file is part of KIRAL. The SPDX lines name its copyright holder and its
// license, in the machine-readable form of the REUSE specification: the GNU
// General Public License, version 3 or any later version
// (LICENSES/GPL-3.0-or-later.txt).

//! `kiral-tools bench`: the time of KIRAL's computations, pinned.
//!
//! `bench kernels` times the products in integers of zenz's layers, in each
//! form of the kernels asked for, on random weights and inputs: for each
//! number of rows, the median of many calls, and the multiply-adds a second.
//! A layer's weights, at most 4.6 MB, stay in the caches between the calls,
//! so that the times are the kernel's own, not the memory's. Then the kernels
//! of single precision that a block of zenz takes besides, on as many rows:
//! GELU, the inputs quantized for the products, a layer normalization, and
//! attention over 72 positions, about a key's on AJIMEE-Bench.
//!
//! `bench model` converts AJIMEE-Bench's readings with zenz alone, greedily, as
//! `eval` does, one item after another on one thread pinned to each processor
//! asked for, and times each prompt -- its tokens fed at once, as a new
//! composition's are -- and each step of decoding after it, a token fed: the
//! median, the 99th percentile and the longest, with the clock
//! ([`crate::pinned`]). The first item is converted once before the times are
//! taken, while the weights come into the caches.
//!
//! `bench keys` times KIRAL's engine's refinement of each key, back to back on
//! one thread pinned to each processor asked for: AJIMEE-Bench's readings typed
//! a kana at a time, each prefix refined by a round from the constraint the
//! one before left, as the engine's worker refines a key
//! ([`kiral_engine::engine::refine_reading`]), and each reading then converted
//! by up to 8 rounds. Without the time between keys, the times are the
//! computation's own, its clock steady, where `replay` times it as typed.
//!
//! `bench load` loads zenz anew, as many times as asked, and feeds the first
//! item's prompt after each load and once more: a directory's weights read and
//! quantized, or a model file mapped and its tables checked, and then the
//! first prompt, which touches every weight -- a mapped file's pages come in
//! from the system's cache of the file as they are first touched -- against the
//! second, whose weights are all in place.

use std::ffi::OsString;
use std::path::Path;
use std::process::ExitCode;
use std::time::{Duration, Instant};

use kiral_cpu::Features;
use kiral_cpu::topology::{self, Topology};
use kiral_dict::Dictionary;
use kiral_engine::engine::{Settings, refine_reading};
use kiral_engine::refine::{ZenzJudge, draft};
use kiral_kernels::Form;
use kiral_kernels::activation::gelu_tanh;
use kiral_kernels::attention::attention;
use kiral_kernels::norm::layer_norm;
use kiral_kernels::quantized::{self, Packed, Rotation, Rows, Scheme};
use kiral_kernels::vector::argmax;
use kiral_lattice::{Constraint, Lattice};
use kiral_nn::gpt2::Session;
use kiral_nn::zenz::{Zenz, prompt, tag_in};

use crate::ajimee::{self, Item};
use crate::arguments;
use crate::eval::{limit, precision_name, scheme_of};
use crate::metrics::count;
use crate::model::Source;
use crate::pinned::{place, print_times, processors, run_on};

const USAGE: &str = "\
usage: kiral-tools bench kernels [options]
       kiral-tools bench model --model <path> [options]
       kiral-tools bench keys --model <path> --dict <file> [options]
       kiral-tools bench load --model <path> [options]

The time of the products in integers of zenz's layers -- 768 inputs to 2304, to
768, to 3072 and to 6000 outputs, and 3072 to 768 -- on random weights and
inputs, in each form of the kernels asked for: for each number of rows, the
median of repeated calls and the multiply-adds a second. Then, for each number
of rows, the kernels of single precision a block takes besides: GELU of 3072
numbers a row (a copy of them each time), the inputs of 768 and of 3072
quantized, a layer normalization of 768, and attention of twelve heads of 64 a
row over 72 positions.

  --forms <names>       the forms, separated by commas [every one this
                        processor runs]
  --rows <numbers>      the numbers of rows, separated by commas [1,4,16,40]
  --cpu <processor>     a processor to pin the run to, again for another run


The time zenz takes to convert AJIMEE-Bench's readings greedily, one item after
another on one thread: each prompt fed at once, and each step of decoding after
it; the median, the 99th percentile and the longest, on each processor asked
for, with the clock it ran at.

  --model <path>        zenz as Hugging Face publishes it, or a model file
  --int8                zenz's products in integers, for a directory [single
                        precision]
  --weight-bits <n>     the bits of a weight's integer, 2 to 8 [8]
  --rotate <order>      each block of this many inputs, a power of four,
                        rotated first by a regular Hadamard matrix [none]
  --fitted              each group of weights at the scale of least squared
                        error, not that of its greatest magnitude
  --cpu <processor>     a processor to pin the run to, again for another run:
                        p<n> the n-th performance core, e<n> the n-th
                        efficiency core, or the system's number [none: unpinned]
  --items <file>        the items [data/ajimee/evaluation_items.json]
  --limit <n>           the first n items only


The time of KIRAL's engine's refinements, back to back on one thread:
AJIMEE-Bench's readings typed a kana at a time, each prefix refined by a round
of verification from the constraint the one before left, as the engine's
worker refines a key, and each reading then converted by up to 8 rounds; the
median, the 99th percentile and the longest, on each processor asked for, with
the clock it ran at.

  --model <path>        zenz as Hugging Face publishes it, or a model file
  --dict <file>         a dictionary file
  --cpu <processor>     a processor to pin the run to, again for another run
  --items <file>        the items [data/ajimee/evaluation_items.json]
  --limit <n>           the first n items only


The time to load zenz -- a directory's weights read, and quantized for --int8,
or a model file mapped and its tables checked -- and then to feed the first
item's prompt, which touches every weight, and to feed it once more; on each
processor asked for, as many times as asked, each a load anew.

  --model <path>        zenz as Hugging Face publishes it, or a model file
  --int8, --weight-bits, --rotate, --fitted
                        the scheme of a directory's products in integers, as
                        for bench model
  --loads <n>           the loads [10]
  --cpu <processor>     a processor to pin the run to, again for another run
  --items <file>        the items, whose first gives the prompt
                        [data/ajimee/evaluation_items.json]";

/// Runs the command.
pub fn main(mut arguments: impl Iterator<Item = OsString>) -> ExitCode {
    let what = arguments.next().map(|what| what.to_string_lossy().into_owned());
    let result = match what.as_deref() {
        Some("kernels") => kernels(arguments),
        Some("model") => model(arguments),
        Some("keys") => keys(arguments),
        Some("load") => load(arguments),
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
    let source = Source::open(&model)?;
    let zenz = source.zenz(scheme_of(&options)?)?;
    let name = topology::processor_name().unwrap_or_else(|| std::env::consts::ARCH.to_owned());
    println!(
        "bench model: AJIMEE-Bench JWTD_v2/v1, {} items converted greedily by {}, {}; {name}",
        items.len(),
        source.name(),
        precision_name(zenz.model.precision())
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
fn convert_all(zenz: &Zenz<'_>, items: &[Item]) -> Result<Times, String> {
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
    zenz: &Zenz<'_>,
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

/// The times of the engine's refinements, and what its conversions gave.
#[derive(Debug, Default)]
struct KeyTimes {
    keys: Vec<Duration>,
    tokens: usize,
    conversions: Vec<Duration>,
    correct: usize,
}

fn keys(arguments: impl Iterator<Item = OsString>) -> Result<(), String> {
    let options = arguments::parse(arguments, &["model", "dict", "items", "limit"], &["cpu"], &["help"])
        .map_err(|message| format!("{message}\n\n{USAGE}"))?;
    if options.flag("help") {
        println!("{USAGE}");
        return Ok(());
    }
    let model = options.path("model").ok_or_else(|| format!("no --model\n\n{USAGE}"))?;
    let dictionary_path = options.path("dict").ok_or_else(|| format!("no --dict\n\n{USAGE}"))?;
    let items = ajimee::load(options.path("items"), options.parsed::<usize>("limit")?)?;
    let topology = Topology::detect();
    let processors = processors(&options, &topology)?;
    let source = Source::open(&model)?;
    let zenz = source.zenz(None)?;
    let bytes = std::fs::read(&dictionary_path).map_err(|error| format!("{}: {error}", dictionary_path.display()))?;
    let dictionary = Dictionary::parse(&bytes).map_err(|error| format!("{}: {error}", dictionary_path.display()))?;
    let name = topology::processor_name().unwrap_or_else(|| std::env::consts::ARCH.to_owned());
    let rounds = Settings::default().conversion_rounds;
    println!(
        "bench keys: AJIMEE-Bench JWTD_v2/v1, {} items typed a kana at a time, back to back, into the refinement of \
         KIRAL's engine: the dictionary {}'s drafts refined by {} ({}), a round a key, up to {rounds} rounds for a \
         conversion; {name}, kernels in {}",
        items.len(),
        dictionary.metadata().name,
        source.name(),
        precision_name(zenz.model.precision()),
        crate::eval::kernels()
    );
    for processor in processors {
        let (times, clock) = run_on(processor, || refine_all(&zenz, &dictionary, &items, rounds))?;
        println!("  {}", place(&topology, processor, &clock));
        print_times(
            &format!(
                "refinements of a key, a round ({:.1} tokens fed a key on average)",
                count(times.tokens) / count(times.keys.len())
            ),
            &times.keys,
        );
        print_times(&format!("conversions, up to {rounds} rounds"), &times.conversions);
        println!(
            "    the conversions: Acc@1 {:.1} ({} of {})",
            100.0 * count(times.correct) / count(items.len()),
            times.correct,
            items.len()
        );
    }
    Ok(())
}

/// Types every item into the engine's refinement, the first once before while
/// the weights come into the caches: each prefix of its reading refined by a
/// round from the constraint its draft is found within -- the one the last
/// refinement left, as the engine's input thread carries it -- then the whole
/// reading converted.
fn refine_all(zenz: &Zenz<'_>, dictionary: &Dictionary<'_>, items: &[Item], rounds: usize) -> Result<KeyTimes, String> {
    let mut judge = ZenzJudge::new(zenz, zenz.model.config().positions.min(512));
    let mut warming = KeyTimes::default();
    refine_item(zenz, dictionary, &mut judge, &items[0], rounds, &mut warming)?;
    let mut times = KeyTimes::default();
    for item in items {
        refine_item(zenz, dictionary, &mut judge, item, rounds, &mut times)?;
    }
    Ok(times)
}

/// Types an item into the engine's refinement, and times each key's and the
/// conversion's.
fn refine_item(
    zenz: &Zenz<'_>,
    dictionary: &Dictionary<'_>,
    judge: &mut ZenzJudge<'_>,
    item: &Item,
    rounds: usize,
    times: &mut KeyTimes,
) -> Result<(), String> {
    // A refinement, and the tokens it fed the model.
    let mut refine = |reading: &str, carried: &Constraint, rounds: usize| {
        let fed = judge.tokens_fed();
        refine_reading(
            zenz,
            dictionary,
            judge,
            reading,
            &item.context,
            carried,
            rounds,
            &mut || false,
        )
        .map_err(|error| format!("item {}: {error}", item.index))
        .map(|refined| {
            (
                refined.expect("a refinement that is never stopped"),
                judge.tokens_fed() - fed,
            )
        })
    };
    let mut carried = Constraint::default();
    let mut prefix = String::new();
    for character in item.reading.chars() {
        prefix.push(character);
        let lattice = Lattice::new(dictionary, &prefix);
        let (_, constraint) = draft(&lattice, dictionary, &carried);
        let started = Instant::now();
        let (refined, fed) = refine(&prefix, &constraint, 1)?;
        times.keys.push(started.elapsed());
        times.tokens += fed;
        carried = refined.constraint;
    }
    let started = Instant::now();
    let (refined, _) = refine(&item.reading, &carried, rounds)?;
    times.conversions.push(started.elapsed());
    times.correct += usize::from(item.expected.contains(&refined.text));
    Ok(())
}

/// The times of loads.
#[derive(Debug, Default)]
struct LoadTimes {
    loads: Vec<Duration>,
    first: Vec<Duration>,
    second: Vec<Duration>,
    prompt_tokens: usize,
}

fn load(arguments: impl Iterator<Item = OsString>) -> Result<(), String> {
    let options = arguments::parse(
        arguments,
        &["model", "items", "loads", "weight-bits", "rotate"],
        &["cpu"],
        &["int8", "fitted", "help"],
    )
    .map_err(|message| format!("{message}\n\n{USAGE}"))?;
    if options.flag("help") {
        println!("{USAGE}");
        return Ok(());
    }
    let model = options.path("model").ok_or_else(|| format!("no --model\n\n{USAGE}"))?;
    let items = ajimee::load(options.path("items"), Some(1))?;
    let item = items.first().ok_or("no items")?;
    let loads = options.parsed::<usize>("loads")?.unwrap_or(10).max(1);
    let scheme = scheme_of(&options)?;
    let topology = Topology::detect();
    let processors = processors(&options, &topology)?;
    let name = topology::processor_name().unwrap_or_else(|| std::env::consts::ARCH.to_owned());
    let what = match Source::open(&model)? {
        Source::File(path, _) => format!("the model file {}, mapped", path.display()),
        Source::Directory(path) if scheme.is_some() => format!("{}, read and quantized", path.display()),
        Source::Directory(path) => format!("{}, read", path.display()),
    };
    println!("bench load: {what}, {loads} times; {name}");
    for processor in processors {
        let (times, clock) = run_on(processor, || load_all(&model, scheme, item, loads))?;
        println!("  {}", place(&topology, processor, &clock));
        print_times("loads", &times.loads);
        print_times(
            &format!("first prompts after a load ({} tokens)", times.prompt_tokens),
            &times.first,
        );
        print_times("second prompts", &times.second);
    }
    Ok(())
}

/// Loads the model anew `loads` times, and times each load and the item's
/// prompt fed after it, twice.
fn load_all(model: &Path, scheme: Option<Scheme>, item: &Item, loads: usize) -> Result<LoadTimes, String> {
    let mut times = LoadTimes::default();
    let mut tokens = Vec::new();
    for _ in 0..loads {
        let started = Instant::now();
        let source = Source::open(model)?;
        let zenz = source.zenz(scheme)?;
        times.loads.push(started.elapsed());
        tokens.clear();
        zenz.tokenizer
            .encode(&prompt(&item.reading, &item.context), &mut tokens);
        times.prompt_tokens = tokens.len();
        let mut session = Session::new(&zenz.model, tokens.len());
        let mut logits = vec![0.0f32; zenz.model.config().vocabulary];
        for pass in [&mut times.first, &mut times.second] {
            session.truncate(0);
            let begun = Instant::now();
            session
                .feed(&tokens, &mut logits)
                .map_err(|error| format!("item {}: {error}", item.index))?;
            pass.push(begun.elapsed());
        }
    }
    Ok(times)
}

/// zenz's layers: inputs and outputs.
const LAYERS: [(usize, usize); 5] = [(768, 2304), (768, 768), (768, 3072), (3072, 768), (768, 6000)];

fn kernels(arguments: impl Iterator<Item = OsString>) -> Result<(), String> {
    let options = arguments::parse(arguments, &["forms", "rows"], &["cpu"], &["help"])
        .map_err(|message| format!("{message}\n\n{USAGE}"))?;
    if options.flag("help") {
        println!("{USAGE}");
        return Ok(());
    }
    let available = Form::available(Features::detect());
    let forms = match options.text("forms") {
        None => available.clone(),
        Some(names) => names
            .split(',')
            .map(|name| {
                let form = Form::from_name(name.trim()).ok_or_else(|| format!("--forms: no form {name}"))?;
                if available.contains(&form) {
                    Ok(form)
                } else {
                    Err(format!("--forms: this processor does not run {form}"))
                }
            })
            .collect::<Result<Vec<_>, String>>()?,
    };
    let rows = match options.text("rows") {
        None => vec![1, 4, 16, 40],
        Some(numbers) => numbers
            .split(',')
            .map(|number| {
                number
                    .trim()
                    .parse::<usize>()
                    .ok()
                    .filter(|&rows| rows > 0)
                    .ok_or_else(|| format!("--rows: {number} is not a number of rows"))
            })
            .collect::<Result<Vec<_>, String>>()?,
    };
    let topology = Topology::detect();
    let name = topology::processor_name().unwrap_or_else(|| std::env::consts::ARCH.to_owned());
    println!(
        "bench kernels: the products in integers of zenz's layers, weights of 8 bits; {name}; forms {}",
        forms.iter().map(|form| form.name()).collect::<Vec<_>>().join(", ")
    );
    for processor in processors(&options, &topology)? {
        let (lines, clock) = run_on(processor, || {
            let mut numbers = Numbers(0x853c_49e6_748f_ea9b);
            let mut lines = time_kernels(&mut numbers, &forms, &rows);
            lines.extend(time_single(&mut numbers, &rows));
            Ok(lines)
        })?;
        println!("  {}", place(&topology, processor, &clock));
        for line in lines {
            println!("    {line}");
        }
    }
    Ok(())
}

/// A fixed stream of numbers in [-1, 1): a linear congruential generator's.
struct Numbers(u64);

impl Numbers {
    /// The next `count` numbers.
    fn values(&mut self, count: usize) -> Vec<f32> {
        (0..count)
            .map(|_| {
                self.0 = self
                    .0
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                f32::from(u16::try_from(self.0 >> 48).unwrap_or(0)) / 32_768.0 - 1.0
            })
            .collect()
    }
}

/// Times each layer at each number of rows in each form; returns a line for
/// each.
fn time_kernels(numbers: &mut Numbers, forms: &[Form], rows: &[usize]) -> Vec<String> {
    let mut lines = Vec::new();
    for (depth, width) in LAYERS {
        let weights = numbers.values(depth * width);
        let packed = Packed::quantize(depth, width, Scheme::INT8, |k, j| weights[k * width + j]);
        let bias = numbers.values(width);
        for &count in rows {
            let mut input = Rows::new();
            input.quantize(&numbers.values(depth * count), depth, Rotation::None);
            let mut output = vec![0.0f32; count * width];
            let mut times = Vec::new();
            for &form in forms {
                times.push((
                    form,
                    median_call(|| quantized::linear_in(form, &input, &packed, &bias, &mut output)),
                ));
            }
            let macs = f64::from(u32::try_from(depth * width * count).unwrap_or(u32::MAX));
            let text = times
                .iter()
                .map(|(form, time)| {
                    let seconds = time.as_secs_f64();
                    format!("{form} {:.3} ms ({:.1} GMAC/s)", seconds * 1e3, macs / seconds / 1e9)
                })
                .collect::<Vec<_>>()
                .join(", ");
            lines.push(format!("{depth} to {width}, {count} rows: {text}"));
        }
    }
    lines
}

/// The positions attention is timed over: about a key's on AJIMEE-Bench, whose
/// key's refinement feeds 20 rows on average, each attending to 71 positions.
const POSITIONS: usize = 72;

/// Times the kernels of single precision a block takes besides its products,
/// at each number of rows; returns a line for each.
fn time_single(numbers: &mut Numbers, rows: &[usize]) -> Vec<String> {
    let (width, heads, head_width) = (768, 12, 64);
    let gain = numbers.values(width);
    let bias = numbers.values(width);
    let keys = numbers.values(heads * POSITIONS * head_width);
    let values = numbers.values(heads * POSITIONS * head_width);
    let milliseconds = |time: Duration| time.as_secs_f64() * 1e3;
    let mut lines = Vec::new();
    for &count in rows {
        // GELU of a fresh copy each time: again and again in place, it takes the
        // values below the least normal float, where they are slow.
        let expansion = numbers.values(count * 4 * width);
        let mut expanded = expansion.clone();
        let gelu = median_call(|| {
            expanded.copy_from_slice(&expansion);
            gelu_tanh(&mut expanded);
        });
        let mut quantized = Rows::new();
        let narrow = numbers.values(count * width);
        let wide = numbers.values(count * 4 * width);
        let quantize_narrow = median_call(|| quantized.quantize(&narrow, width, Rotation::None));
        let quantize_wide = median_call(|| quantized.quantize(&wide, 4 * width, Rotation::None));
        let mut normalized = vec![0.0f32; count * width];
        let norm = median_call(|| layer_norm(&narrow, &gain, &bias, 1e-5, &mut normalized));
        let queries = numbers.values(count * width);
        let mut scores = vec![0.0f32; POSITIONS];
        let mut found = vec![0.0f32; count * width];
        let attend = median_call(|| {
            for (query, found) in queries.chunks_exact(width).zip(found.chunks_exact_mut(width)) {
                for head in 0..heads {
                    let cached = &keys[head * POSITIONS * head_width..][..POSITIONS * head_width];
                    let values = &values[head * POSITIONS * head_width..][..POSITIONS * head_width];
                    attention(
                        &query[head * head_width..][..head_width],
                        cached,
                        values,
                        0.125,
                        &mut scores,
                        &mut found[head * head_width..][..head_width],
                    );
                }
            }
        });
        lines.push(format!(
            "single precision, {count} rows: GELU of {} {:.4} ms; quantized, {width} {:.4} ms and {} {:.4} ms; \
             normalized, {width} {:.4} ms; attention of {heads} heads over {POSITIONS} positions {:.4} ms",
            4 * width,
            milliseconds(gelu),
            milliseconds(quantize_narrow),
            4 * width,
            milliseconds(quantize_wide),
            milliseconds(norm),
            milliseconds(attend)
        ));
    }
    lines
}

/// The median time of a call, of as many as fill 0.2 s and at least 11, after
/// one that warms the caches.
fn median_call(mut call: impl FnMut()) -> Duration {
    call();
    let started = Instant::now();
    let mut times = Vec::new();
    while times.len() < 11 || started.elapsed() < Duration::from_millis(200) {
        let begun = Instant::now();
        call();
        times.push(begun.elapsed());
    }
    times.sort_unstable();
    times[times.len() / 2]
}
