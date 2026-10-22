// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later
//
// This file is part of KIRAL. The SPDX lines name its copyright holder and its
// license, in the machine-readable form of the REUSE specification: the GNU
// General Public License, version 3 or any later version
// (LICENSES/GPL-3.0-or-later.txt).

//! `kiral-tools dict`: dictionary files.
//!
//! `dict build --mozc <directory>` builds one from Mozc's open-source
//! dictionary, as `scripts/fetch_datasets.py fetch mozc-dictionary` lays it
//! out: its words and the costs of their connections, with Mozc's notices --
//! IPAdic's license, the Okinawa dictionary's, Mozc's BSD-3-Clause -- in the
//! file, which goes under their terms, not KIRAL's.

use std::ffi::OsString;
use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Instant;

use kiral_dict::file::{self, Unknown};
use kiral_dict::{Dictionary, Metadata, mozc};

use crate::arguments;

const USAGE: &str = "\
usage: kiral-tools dict build --mozc <directory> --output <file> [options]

Builds a dictionary file from Mozc's open-source dictionary: dictionary00.txt to
dictionary09.txt, connection_single_column.txt, README.txt and LICENSE, as
scripts/fetch_datasets.py fetch mozc-dictionary lays them out in
data/mozc-dictionary/. The file holds their notices and goes under their terms.

  --mozc <directory>    Mozc's files
  --output <file>       the dictionary file to write
  --name <name>         the dictionary's name [mozc-oss]
  --source <text>       where its words come from [Mozc's open-source dictionary]";

/// The license of Mozc's open-source dictionary: Google's additions under
/// Mozc's BSD-3-Clause, the rest under IPAdic's terms (the Okinawa dictionary's
/// words are in the public domain).
const MOZC_LICENSE: &str = "BSD-3-Clause AND LicenseRef-IPAdic";

/// The part of speech a character of no reading is taken as: a common noun.
const UNKNOWN_PART_OF_SPEECH: &str = "名詞,一般,*,*,*,*,*";

/// The cost of a character of no reading: above that of the costliest word of
/// Mozc's dictionary (18,318), so that a character is taken alone only where no
/// word covers it.
const UNKNOWN_COST: u16 = 20_000;

/// Runs the command.
pub fn main(mut arguments: impl Iterator<Item = OsString>) -> ExitCode {
    let action = arguments.next().map(|action| action.to_string_lossy().into_owned());
    let result = match action.as_deref() {
        Some("build") => build(arguments),
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

fn build(arguments: impl Iterator<Item = OsString>) -> Result<(), String> {
    let options = arguments::parse(arguments, &["mozc", "output", "name", "source"], &[], &["help"])
        .map_err(|message| format!("{message}\n\n{USAGE}"))?;
    if options.flag("help") {
        println!("{USAGE}");
        return Ok(());
    }
    let directory: PathBuf = options.path("mozc").ok_or_else(|| format!("no --mozc\n\n{USAGE}"))?;
    let output = options
        .path("output")
        .ok_or_else(|| format!("no --output\n\n{USAGE}"))?;
    let name = options.text("name").unwrap_or_else(|| "mozc-oss".to_owned());
    let source = options
        .text("source")
        .unwrap_or_else(|| "Mozc's open-source dictionary (src/data/dictionary_oss)".to_owned());
    let started = Instant::now();
    let mozc = mozc::read(&directory).map_err(|error| error.to_string())?;
    let read = started.elapsed();
    let metadata = Metadata {
        name: &name,
        license: MOZC_LICENSE,
        source: &source,
        notice: &mozc.notice,
    };
    let unknown = Unknown {
        id: mozc
            .id_of(UNKNOWN_PART_OF_SPEECH)
            .ok_or_else(|| format!("id.def names no {UNKNOWN_PART_OF_SPEECH}"))?,
        cost: UNKNOWN_COST,
    };
    let bytes = file::build(&mozc.words, &mozc.matrix, unknown, &metadata).map_err(|error| error.to_string())?;
    let built = started.elapsed();
    std::fs::write(&output, &bytes).map_err(|error| format!("{}: {error}", output.display()))?;
    let dictionary = Dictionary::parse(&bytes).map_err(|error| error.to_string())?;
    println!(
        "{}: {} words read in {:.1} s, {} readings and {} entries built in {:.1} s, {:.1} MB",
        output.display(),
        mozc.words.len(),
        read.as_secs_f64(),
        dictionary.readings(),
        dictionary.entries(),
        built.saturating_sub(read).as_secs_f64(),
        f64::from(u32::try_from(bytes.len() / 1000).unwrap_or(u32::MAX)) / 1000.0
    );
    println!("{}", dictionary.sizes());
    Ok(())
}
