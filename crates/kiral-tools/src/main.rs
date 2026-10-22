// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later
//
// This file is part of KIRAL. The SPDX lines name its copyright holder and its
// license, in the machine-readable form of the REUSE specification: the GNU
// General Public License, version 3 or any later version
// (LICENSES/GPL-3.0-or-later.txt).

//! `kiral-tools`, KIRAL's command line.
//!
//! - `agree`: how faithful zenz's products in integers are to single
//!   precision, by the most likely next token at each position.
//! - `bench`: the time of KIRAL's computations, pinned: `bench kernels`, the
//!   products in integers in each form; `bench model`, zenz's prompts and
//!   steps of decoding.
//! - `dict`: dictionary files, built from Mozc's open-source dictionary.
//! - `eval`: the accuracy of a conversion against AJIMEE-Bench, and the items
//!   it changed since the run before.
//! - `replay`: the time an input method takes for each key, typed as a user
//!   types, on the processors asked for.
#![forbid(unsafe_code)]

mod agree;
mod ajimee;
mod arguments;
mod bench;
mod dict;
mod eval;
mod metrics;
mod mozc;
mod pinned;
mod replay;
mod results;
mod romaji;
mod sexp;

use std::process::ExitCode;

const USAGE: &str = "\
usage: kiral-tools <command> [options]

commands:
  agree   zenz's products in integers against single precision (kiral-tools agree --help)
  bench   the time of a computation, pinned: bench kernels, bench model (kiral-tools bench help)
  dict    dictionary files: dict build (kiral-tools dict help)
  eval    the accuracy of a conversion against AJIMEE-Bench (kiral-tools eval --help)
  replay  the time of each key typed into an input method (kiral-tools replay --help)";

fn main() -> ExitCode {
    let mut arguments = std::env::args_os().skip(1);
    let command = arguments.next().map(|command| command.to_string_lossy().into_owned());
    match command.as_deref() {
        Some("agree") => agree::main(arguments),
        Some("bench") => bench::main(arguments),
        Some("dict") => dict::main(arguments),
        Some("eval") => eval::main(arguments),
        Some("replay") => replay::main(arguments),
        Some("help" | "--help" | "-h") => {
            println!("{USAGE}");
            ExitCode::SUCCESS
        }
        Some(command) => {
            eprintln!("error: no command {command}\n\n{USAGE}");
            ExitCode::from(2)
        }
        None => {
            eprintln!("{USAGE}");
            ExitCode::from(2)
        }
    }
}
