// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later
//
// This file is part of KIRAL. The SPDX lines name its copyright holder and its
// license, in the machine-readable form of the REUSE specification: the GNU
// General Public License, version 3 or any later version
// (LICENSES/GPL-3.0-or-later.txt).

//! Writes the files the frontends' tests open, which tests cannot fetch: a
//! dictionary of three words whose conversions are known -- かな is 仮名, its
//! one word costing less than か 可 and な 名 together -- and, where a second
//! path is given, the model file of a small zenz of random weights.
//!
//! ```text
//! cargo run --release --locked -p kiral-capi --example fixtures -- <dictionary> [<model>]
//! ```

#[path = "../tests/common/mod.rs"]
mod common;

use std::path::Path;
use std::process::ExitCode;

use kiral_dict::file::{Unknown, build};
use kiral_dict::mozc::Matrix;
use kiral_dict::{Metadata, Word};

/// Writes bytes to a path, its directories made first.
fn write(path: &Path, bytes: &[u8]) -> Result<(), String> {
    if let Some(directory) = path.parent() {
        std::fs::create_dir_all(directory).map_err(|error| format!("{}: {error}", directory.display()))?;
    }
    std::fs::write(path, bytes).map_err(|error| format!("{}: {error}", path.display()))
}

fn run() -> Result<(), String> {
    let mut paths = std::env::args_os().skip(1);
    let dictionary = paths.next().ok_or("usage: fixtures <dictionary> [<model>]")?;
    let words = [("かな", "仮名", 100), ("か", "可", 900), ("な", "名", 900)].map(|(reading, surface, cost)| Word {
        reading: reading.to_owned(),
        surface: surface.to_owned(),
        lid: 1,
        rid: 1,
        cost,
        flags: 0,
    });
    let matrix = Matrix {
        ids: 2,
        costs: vec![0; 4],
    };
    let metadata = Metadata {
        name: "fixture",
        license: "GPL-3.0-or-later",
        source: "made by kiral-capi's example fixtures, for the frontends' tests",
        notice: "",
    };
    let bytes = build(&words, &matrix, Unknown { id: 1, cost: 8000 }, &metadata).map_err(|error| error.to_string())?;
    write(Path::new(&dictionary), &bytes)?;
    if let Some(model) = paths.next() {
        write(Path::new(&model), &common::model_file(&mut common::Random(1)))?;
    }
    Ok(())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => {
            eprintln!("error: {message}");
            ExitCode::FAILURE
        }
    }
}
