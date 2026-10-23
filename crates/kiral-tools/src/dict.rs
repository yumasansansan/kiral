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
//! file, which goes under their terms, not KIRAL's. The digits read in kana
//! and the suffixes of nouns are marked, for a path's text to write the digits
//! as Mozc shows them ([`kiral_dict::mozc::mark_digits`]). With `--sudachi
//! <directory>`, the nouns of SudachiDict's lists that Mozc's dictionary lacks
//! are added in Mozc's classes ([`kiral_dict::sudachi`]), and SudachiDict's
//! notices -- its Apache License, UniDic's BSD-3-Clause, NEologd's sources --
//! go in the file too. Then the second words of compounds read voiced, which
//! two compounds of the words or more attest, are added as words of their own
//! ([`kiral_dict::rendaku`]).

use std::ffi::OsString;
use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Instant;

use kiral_dict::file::{self, Unknown};
use kiral_dict::{Dictionary, Metadata, mozc, rendaku, sudachi};

use crate::arguments;

const USAGE: &str = "\
usage: kiral-tools dict build --mozc <directory> --output <file> [options]

Builds a dictionary file from Mozc's open-source dictionary: dictionary00.txt to
dictionary09.txt, connection_single_column.txt, id.def, README.txt and LICENSE,
as scripts/fetch_datasets.py fetch mozc-dictionary lays them out in
data/mozc-dictionary/; with --sudachi, the nouns of SudachiDict's lists that
Mozc's dictionary lacks are added, in Mozc's classes, from the files
scripts/fetch_datasets.py fetch sudachi-dictionary lays out in
data/sudachi-dictionary/. The second words of compounds read voiced -- 駆け,
かけ, read がけ in 朝駆け -- are added where two compounds of the words or more
attest it. The file holds their notices and goes under their terms.

  --mozc <directory>     Mozc's files
  --sudachi <directory>  SudachiDict's files
  --lists <names>        SudachiDict's lists to read, separated by commas
                         [small_lex.csv,core_lex.csv]
  --output <file>        the dictionary file to write
  --name <name>          the dictionary's name [mozc-oss, or mozc-sudachi]
  --source <text>        where its words come from";

/// The license of Mozc's open-source dictionary: Google's additions under
/// Mozc's BSD-3-Clause, the rest under IPAdic's terms (the Okinawa dictionary's
/// words are in the public domain).
const MOZC_LICENSE: &str = "BSD-3-Clause AND LicenseRef-IPAdic";

/// The license of Mozc's dictionary with SudachiDict's words: SudachiDict is
/// under the Apache License 2.0, and what it holds of UniDic under UniDic's
/// BSD-3-Clause.
const MOZC_SUDACHI_LICENSE: &str = "BSD-3-Clause AND LicenseRef-IPAdic AND Apache-2.0";

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
    let options = arguments::parse(
        arguments,
        &["mozc", "sudachi", "lists", "output", "name", "source"],
        &[],
        &["help"],
    )
    .map_err(|message| format!("{message}\n\n{USAGE}"))?;
    if options.flag("help") {
        println!("{USAGE}");
        return Ok(());
    }
    let directory: PathBuf = options.path("mozc").ok_or_else(|| format!("no --mozc\n\n{USAGE}"))?;
    let output = options
        .path("output")
        .ok_or_else(|| format!("no --output\n\n{USAGE}"))?;
    let sudachi_directory = options.path("sudachi");
    let lists: Vec<String> = options.text("lists").map_or_else(
        || sudachi::CORE.iter().map(|&list| list.to_owned()).collect(),
        |lists| lists.split(',').map(str::to_owned).collect(),
    );
    let started = Instant::now();
    let mut mozc = mozc::read(&directory).map_err(|error| error.to_string())?;
    let (digits, suffixes) = mozc::mark_digits(&mut mozc);
    let mozc_words = mozc.words.len();
    let (name, source, license, notice) = if let Some(sudachi_directory) = &sudachi_directory {
        let lists: Vec<&str> = lists.iter().map(String::as_str).collect();
        let nouns = sudachi::read(sudachi_directory, &lists, &mozc).map_err(|error| error.to_string())?;
        print_nouns(&nouns, &mozc, &lists);
        mozc.words.extend(nouns.words);
        (
            "mozc-sudachi",
            format!(
                "Mozc's open-source dictionary (src/data/dictionary_oss), with the nouns of SudachiDict's {} it lacks \
                 and the voiced second words their compounds attest",
                lists.join(", ")
            ),
            MOZC_SUDACHI_LICENSE,
            format!("{}\n\n{}", mozc.notice, nouns.notice),
        )
    } else {
        (
            "mozc-oss",
            "Mozc's open-source dictionary (src/data/dictionary_oss), with the voiced second words its compounds attest"
                .to_owned(),
            MOZC_LICENSE,
            mozc.notice.clone(),
        )
    };
    let suffix = mozc
        .id_of(rendaku::SUFFIX_CLASS)
        .ok_or_else(|| format!("id.def names no {}", rendaku::SUFFIX_CLASS))?;
    let voiced = rendaku::voiced(&mozc.words, &mozc.parts_of_speech, suffix);
    println!(
        "rendaku: {} voiced second words attested by {} compounds or more, {} of them added",
        voiced.attested,
        rendaku::COMPOUNDS,
        voiced.words.len()
    );
    mozc.words.extend(voiced.words);
    let read = started.elapsed();
    let name = options.text("name").unwrap_or_else(|| name.to_owned());
    let source = options.text("source").unwrap_or(source);
    let metadata = Metadata {
        name: &name,
        license,
        source: &source,
        notice: &notice,
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
        "{}: {} words ({mozc_words} of Mozc's, {digits} digits read in kana and {suffixes} suffixes marked) read in \
         {:.1} s, {} readings and {} entries built in {:.1} s, {:.1} MB",
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

/// Prints what was taken of SudachiDict: its words, those Mozc's dictionary
/// has, and for each class the words taken and the cost they are given.
fn print_nouns(nouns: &sudachi::Nouns, mozc: &mozc::Source, lists: &[&str]) {
    println!(
        "SudachiDict ({}): {} words, {} nouns taken, {} nouns Mozc's dictionary or a line before has",
        lists.join(", "),
        nouns.lines,
        nouns.words.len(),
        nouns.repeated
    );
    for (&(sudachi, _), class) in sudachi::CLASSES.iter().zip(&nouns.classes) {
        let name = mozc
            .parts_of_speech
            .get(usize::from(class.id))
            .map_or("", String::as_str);
        println!(
            "  {sudachi:<24} {:>8} words as {} {name}, cost {}",
            class.words, class.id, class.cost
        );
    }
}
