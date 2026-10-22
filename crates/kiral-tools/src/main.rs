// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later
//
// This file is part of KIRAL. The SPDX lines name its copyright holder and its
// license, in the machine-readable form of the REUSE specification: the GNU
// General Public License, version 3 or any later version
// (LICENSES/GPL-3.0-or-later.txt).

//! `kiral-tools`, KIRAL's command line.
//!
//! - `eval`: the accuracy of a conversion against AJIMEE-Bench, and the items
//!   it changed since the run before.
//! - `replay`: the time an input method takes for each key, typed as a user
//!   types, on the processors asked for.
#![forbid(unsafe_code)]

mod ajimee;
mod arguments;
mod eval;
mod metrics;
mod mozc;
mod replay;
mod results;
mod romaji;
mod sexp;

use std::process::ExitCode;

const USAGE: &str = "\
usage: kiral-tools <command> [options]

commands:
  eval    the accuracy of a conversion against AJIMEE-Bench (kiral-tools eval --help)
  replay  the time of each key typed into an input method (kiral-tools replay --help)";

fn main() -> ExitCode {
    let mut arguments = std::env::args_os().skip(1);
    let command = arguments.next().map(|command| command.to_string_lossy().into_owned());
    match command.as_deref() {
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
