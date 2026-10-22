// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later
//
// This file is part of KIRAL. The SPDX lines name its copyright holder and its
// license, in the machine-readable form of the REUSE specification: the GNU
// General Public License, version 3 or any later version
// (LICENSES/GPL-3.0-or-later.txt).

//! `kiral-tools model`: model files.
//!
//! `model convert` makes one from zenz as Hugging Face publishes it: its
//! weights quantized by a scheme as `eval --int8` quantizes them when it reads
//! them, and written with its configuration and its tokenizer into a file that
//! is loaded by mapping it (docs/MODEL-FORMAT.md). The file is not KIRAL's: it
//! goes under the license of the weights, which it names, with the notices that
//! go with them -- their source, their license, and what was changed. `model
//! check` reads a file whole against the digest it carries, loads it, and says
//! what it holds.
//!
//! The other commands take a model by its [`Source`]: a directory as Hugging
//! Face publishes zenz, or a model file, mapped.

use std::ffi::OsString;
use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

use kiral_kernels::quantized::{Rotation, Scheme};
use kiral_model::file::{self, Kind, Metadata, ModelFile};
use kiral_model::sha256;
use kiral_nn::gpt2::{Config, Precision};
use kiral_nn::zenz::Zenz;
use kiral_sys::map::Map;

use crate::arguments;
use crate::eval::{precision_name, scheme_of};

const USAGE: &str = "\
usage: kiral-tools model convert --model <directory> --output <file> --source <text> [options]
       kiral-tools model check --model <file>

convert makes a model file of zenz in integers from zenz as Hugging Face
publishes it, its weights quantized as eval --int8 quantizes them when it reads
them; the file is loaded by mapping it. It goes under the license of the
weights, which it names, with their notices.

  --model <directory>   zenz as Hugging Face publishes it: config.json,
                        model.safetensors, vocab.json, merges.txt,
                        special_tokens_map.json, and README.md, whose front
                        matter gives the license
  --output <file>       the model file to write
  --source <text>       where the weights come from: their repository and its
                        revision
  --name <name>         the model's name [the directory's name]
  --license <SPDX>      the weights' license [README.md's]
  --notice <text>       the notices that go with any copy [the weights' source
                        and license, and what was changed]
  --weight-bits <n>     the bits of a weight's integer, 2 to 8 [8]
  --rotate <order>      each block of this many inputs, a power of four,
                        rotated first by a regular Hadamard matrix [none]
  --fitted              each group of weights at the scale of least squared
                        error, not that of its greatest magnitude

check reads a model file whole against the digest it carries, loads it as
zenz, and says what it holds.

  --model <file>        the model file";

/// The SPDX identifiers a model card's license is read as, in SPDX's case:
/// Hugging Face writes them in lowercase.
const LICENSES: [&str; 7] = [
    "Apache-2.0",
    "CC-BY-4.0",
    "CC-BY-NC-4.0",
    "CC-BY-NC-SA-4.0",
    "CC-BY-SA-4.0",
    "CC0-1.0",
    "MIT",
];

/// Runs the command.
pub fn main(mut arguments: impl Iterator<Item = OsString>) -> ExitCode {
    let action = arguments.next().map(|action| action.to_string_lossy().into_owned());
    let result = match action.as_deref() {
        Some("convert") => convert(arguments),
        Some("check") => check(arguments),
        Some("help" | "--help") => {
            println!("{USAGE}");
            Ok(())
        }
        _ => Err(format!("no action\n\n{USAGE}")),
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => {
            eprintln!("error: {message}");
            ExitCode::FAILURE
        }
    }
}

/// Where a model comes from.
#[derive(Debug)]
pub enum Source {
    /// A directory as Hugging Face publishes zenz.
    Directory(PathBuf),
    /// A model file, mapped.
    File(PathBuf, Map),
}

impl Source {
    /// The source a path names: a model file where it is a file that begins
    /// with the format's magic, and otherwise the directory of zenz it is or is
    /// in.
    ///
    /// # Errors
    ///
    /// Where a model file cannot be mapped.
    pub fn open(path: &Path) -> Result<Self, String> {
        let mut magic = [0u8; 8];
        let is_model_file = File::open(path)
            .and_then(|mut file| file.read_exact(&mut magic))
            .is_ok_and(|()| &magic == file::MAGIC);
        if is_model_file {
            let map = Map::open(path).map_err(|error| format!("{}: {error}", path.display()))?;
            Ok(Self::File(path.to_path_buf(), map))
        } else {
            Ok(Self::Directory(Zenz::directory_of(path)))
        }
    }

    /// zenz: from a directory, its products in integers by the scheme given or
    /// in single precision where none is; from a model file, in integers by the
    /// file's scheme, its weights read where they lie.
    ///
    /// # Errors
    ///
    /// Where the model cannot be read, or a scheme is given for a model file,
    /// which has its own.
    pub fn zenz(&self, scheme: Option<Scheme>) -> Result<Zenz<'_>, String> {
        match self {
            Self::Directory(directory) => Zenz::load(directory, scheme.map_or(Precision::Single, Precision::Integers))
                .map_err(|error| format!("{}: {error}", directory.display())),
            Self::File(path, map) => {
                if scheme.is_some() {
                    return Err(format!(
                        "{}: a model file, whose scheme is its own: --int8 and its choices are for a directory",
                        path.display()
                    ));
                }
                let failed = |error: &dyn std::fmt::Display| format!("{}: {error}", path.display());
                let model = ModelFile::parse(map.bytes()).map_err(|error| failed(&error))?;
                Zenz::from_file(&model).map_err(|error| failed(&error))
            }
        }
    }

    /// The model's name: the directory's, or the one the model file gives.
    pub fn name(&self) -> String {
        match self {
            Self::Directory(directory) => directory
                .file_name()
                .map_or_else(|| "zenz".to_owned(), |name| name.to_string_lossy().into_owned()),
            Self::File(path, map) => ModelFile::parse(map.bytes())
                .map_or_else(|_| path.display().to_string(), |model| model.metadata().name.to_owned()),
        }
    }

    /// Where it is, for the log.
    pub fn path(&self) -> &Path {
        match self {
            Self::Directory(path) | Self::File(path, _) => path,
        }
    }
}

/// Megabytes, of a number of bytes.
fn megabytes(bytes: usize) -> f64 {
    f64::from(u32::try_from(bytes / 1000).unwrap_or(u32::MAX)) / 1000.0
}

fn convert(arguments: impl Iterator<Item = OsString>) -> Result<(), String> {
    let options = arguments::parse(
        arguments,
        &[
            "model",
            "output",
            "source",
            "name",
            "license",
            "notice",
            "weight-bits",
            "rotate",
        ],
        &[],
        &["int8", "fitted", "help"],
    )
    .map_err(|message| format!("{message}\n\n{USAGE}"))?;
    if options.flag("help") {
        println!("{USAGE}");
        return Ok(());
    }
    let model = options.path("model").ok_or_else(|| format!("no --model\n\n{USAGE}"))?;
    let output = options
        .path("output")
        .ok_or_else(|| format!("no --output\n\n{USAGE}"))?;
    let source = options
        .text("source")
        .ok_or_else(|| format!("no --source: where the weights come from\n\n{USAGE}"))?;
    let scheme = scheme_of(&options)?.unwrap_or(Scheme::INT8);
    let Source::Directory(directory) = Source::open(&model)? else {
        return Err(format!("--model {}: a model file, not a directory", model.display()));
    };
    let name = options
        .text("name")
        .unwrap_or_else(|| Source::Directory(directory.clone()).name());
    let license = match options.text("license") {
        Some(license) => license,
        None => card_license(&directory).ok_or_else(|| {
            format!(
                "{}: no license of {} in README.md's front matter; give --license",
                directory.display(),
                LICENSES.join(", ")
            )
        })?,
    };
    let notice = options
        .text("notice")
        .unwrap_or_else(|| default_notice(&name, &source, &license, &scheme));
    let started = Instant::now();
    let zenz = Zenz::load(&directory, Precision::Integers(scheme))
        .map_err(|error| format!("{}: {error}", directory.display()))?;
    let loaded = started.elapsed();
    let metadata = Metadata {
        name: &name,
        license: &license,
        source: &source,
        notice: &notice,
    };
    let bytes = zenz.to_file(&metadata).map_err(|error| error.to_string())?;
    drop(zenz);
    if let Some(parent) = output.parent().filter(|parent| !parent.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent).map_err(|error| format!("{}: {error}", parent.display()))?;
    }
    std::fs::write(&output, &bytes).map_err(|error| format!("{}: {error}", output.display()))?;
    drop(bytes);
    let written = started.elapsed();
    println!(
        "{}: {name} in integers ({}): read and quantized in {:.2} s, its file made, digest and all, and written in \
         {:.2} s",
        output.display(),
        precision_name(Precision::Integers(scheme)),
        loaded.as_secs_f64(),
        written.saturating_sub(loaded).as_secs_f64()
    );
    let source = Source::open(&output)?;
    let Source::File(_, map) = &source else {
        return Err(format!("{}: not read back as a model file", output.display()));
    };
    source.zenz(None)?;
    let model = ModelFile::parse(map.bytes()).map_err(|error| error.to_string())?;
    for line in describe(&model) {
        println!("  {line}");
    }
    Ok(())
}

/// The license of a model card's front matter, as SPDX writes it: one of
/// [`LICENSES`].
fn card_license(directory: &Path) -> Option<String> {
    let card = std::fs::read_to_string(directory.join("README.md")).ok()?;
    let front = card.strip_prefix("---")?.split("\n---").next()?;
    let license = front
        .lines()
        .find_map(|line| line.trim().strip_prefix("license:"))?
        .trim();
    LICENSES
        .iter()
        .find(|known| known.eq_ignore_ascii_case(license))
        .map(|&known| known.to_owned())
}

/// The notices of a converted model: where its weights come from, under what
/// license -- with the address of its text where the license is one
/// identifier -- and what was changed, as the licenses of Creative Commons ask
/// of adapted material.
fn default_notice(name: &str, source: &str, license: &str, scheme: &Scheme) -> String {
    let link = if license.contains(' ') {
        String::new()
    } else {
        format!(" (https://spdx.org/licenses/{license}.html)")
    };
    let rotation = match scheme.rotation {
        Rotation::None => String::new(),
        Rotation::Hadamard(order) => {
            format!(", after a rotation of each block of {order} inputs by a regular Hadamard matrix")
        }
    };
    let scales = if scheme.fitted {
        "the scale of least squared error"
    } else {
        "the scale of its greatest magnitude"
    };
    let quantized = if scheme.weight_bits == 8 {
        format!(
            "the weights of the linear layers and of the output layer, the token embedding's, quantized to \
             integers of 8 bits{rotation}"
        )
    } else {
        format!(
            "the weights of the linear layers quantized to integers of {} bits{rotation}, and those of the output \
             layer, the token embedding's, to integers of 8",
            scheme.weight_bits
        )
    };
    format!(
        "{name}: weights from {source}, under the license {license}{link}.\n\
         Changed by kiral-tools model convert, of KIRAL: {quantized}, each output's group of 64 inputs at \
         {scales}, and packed in panels of 16 outputs; the other tensors as they were. This file is under the same \
         license."
    )
}

/// What a model file says of itself and holds, a line for each.
fn describe(model: &ModelFile<'_>) -> Vec<String> {
    let metadata = model.metadata();
    let vocabulary = model.vocabulary();
    let mut lines = vec![
        format!(
            "{:.1} MB, sha256 {}",
            megabytes(model.bytes().len()),
            sha256::hex(&model.digest())
        ),
        format!(
            "architecture {}, tokenizer {} of {} words and {} merges",
            model.architecture(),
            model.tokenizer(),
            vocabulary.len(),
            vocabulary.merge_count()
        ),
        format!("name {}; license {}", metadata.name, metadata.license),
        format!("source {}", metadata.source),
    ];
    if let Ok(config) = Config::from_bytes(model.configuration()) {
        lines.push(format!(
            "{} blocks of {} heads, width {}, {} positions, {} words, epsilon {:e}",
            config.layers, config.heads, config.width, config.positions, config.vocabulary, config.epsilon
        ));
    }
    let (mut panels, mut panel_bytes, mut floats, mut float_bytes) = (0, 0, 0, 0);
    let mut kinds = Vec::new();
    for tensor in model.tensors() {
        let kind = match tensor.kind {
            Kind::F32 => "single precision".to_owned(),
            Kind::Bf16 => "bfloat16".to_owned(),
            Kind::Panels(scheme) => format!(
                "panels of {} columns, groups of {}, {} bits{}{}",
                scheme.columns,
                scheme.group,
                scheme.bits,
                if scheme.rotation == 0 {
                    String::new()
                } else {
                    format!(", rotated in blocks of {}", scheme.rotation)
                },
                if scheme.fitted { ", fitted scales" } else { "" }
            ),
        };
        if matches!(tensor.kind, Kind::Panels(_)) {
            panels += 1;
            panel_bytes += tensor.bytes.len();
        } else {
            floats += 1;
            float_bytes += tensor.bytes.len();
        }
        if !kinds.contains(&kind) {
            kinds.push(kind);
        }
    }
    lines.push(format!(
        "{} tensors: {panels} of weights in panels, {:.1} MB, and {floats} of floats, {:.1} MB ({})",
        model.tensors().len(),
        megabytes(panel_bytes),
        megabytes(float_bytes),
        kinds.join("; ")
    ));
    lines.push("notice:".to_owned());
    lines.extend(metadata.notice.lines().map(|line| format!("  {line}")));
    lines
}

fn check(arguments: impl Iterator<Item = OsString>) -> Result<(), String> {
    let options =
        arguments::parse(arguments, &["model"], &[], &["help"]).map_err(|message| format!("{message}\n\n{USAGE}"))?;
    if options.flag("help") {
        println!("{USAGE}");
        return Ok(());
    }
    let path = options.path("model").ok_or_else(|| format!("no --model\n\n{USAGE}"))?;
    let source = Source::open(&path)?;
    let Source::File(_, map) = &source else {
        return Err(format!("{}: not a model file", path.display()));
    };
    let model = ModelFile::parse(map.bytes()).map_err(|error| format!("{}: {error}", path.display()))?;
    println!("model check: {}", path.display());
    for line in describe(&model) {
        println!("  {line}");
    }
    let started = Instant::now();
    let verified = model.verify();
    let digested = started.elapsed();
    if !verified {
        return Err(format!(
            "{}: its bytes are not those of the digest it carries",
            path.display()
        ));
    }
    let zenz = source.zenz(None)?;
    println!(
        "  the digest checked, every byte read, in {:.3} s; loaded as zenz, {}",
        digested.as_secs_f64(),
        precision_name(zenz.model.precision())
    );
    Ok(())
}
