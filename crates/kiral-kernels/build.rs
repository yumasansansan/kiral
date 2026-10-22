// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later
//
// This file is part of KIRAL. The SPDX lines name its copyright holder and its
// license, in the machine-readable form of the REUSE specification: the GNU
// General Public License, version 3 or any later version
// (LICENSES/GPL-3.0-or-later.txt).

//! Assembles the kernels written in assembly for the target's architecture
//! into a static library that the crate links: each `.S` of `asm/<arch>/` by
//! Clang, through its preprocessor and `asm/common.h`, for the target's triple,
//! then the objects archived by `llvm-ar`. The LLVM the project builds with has
//! both: `clang` and `llvm-ar` on `PATH`, or the programs `KIRAL_CLANG` and
//! `KIRAL_AR` name. An architecture with no kernels in assembly gets no library.

use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    println!("cargo::rerun-if-changed=asm");
    println!("cargo::rerun-if-env-changed=KIRAL_CLANG");
    println!("cargo::rerun-if-env-changed=KIRAL_AR");
    let architecture = std::env::var("CARGO_CFG_TARGET_ARCH").expect("Cargo names the target's architecture");
    let Ok(entries) = std::fs::read_dir(Path::new("asm").join(&architecture)) else {
        return;
    };
    let mut sources: Vec<PathBuf> = entries
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| path.extension().is_some_and(|extension| extension == "S"))
        .collect();
    if sources.is_empty() {
        return;
    }
    sources.sort();
    let target = std::env::var("TARGET").expect("Cargo names the target");
    let out = PathBuf::from(std::env::var_os("OUT_DIR").expect("Cargo names the output directory"));
    let clang = std::env::var("KIRAL_CLANG").unwrap_or_else(|_| "clang".to_owned());
    let archiver = std::env::var("KIRAL_AR").unwrap_or_else(|_| "llvm-ar".to_owned());
    let mut objects = Vec::with_capacity(sources.len());
    for source in &sources {
        println!("cargo::rerun-if-changed={}", source.display());
        let stem = source.file_stem().expect("a source's name");
        let object = out.join(stem).with_extension("o");
        run(Command::new(&clang)
            .arg(format!("--target={target}"))
            .args(["-c", "-Iasm", "-Werror"])
            .arg(source)
            .arg("-o")
            .arg(&object));
        objects.push(object);
    }
    // The name MSVC's linker looks for, and the one of every other.
    let library = out.join(if target.ends_with("-msvc") {
        "kiral_asm.lib"
    } else {
        "libkiral_asm.a"
    });
    if library.exists() {
        std::fs::remove_file(&library).expect("the library of an earlier build removed");
    }
    run(Command::new(&archiver).arg("crs").arg(&library).args(&objects));
    println!("cargo::rustc-link-search=native={}", out.display());
    println!("cargo::rustc-link-lib=static=kiral_asm");
}

/// Runs a program of the toolchain, and stops the build where it fails.
fn run(command: &mut Command) {
    let status = command
        .status()
        .unwrap_or_else(|error| panic!("{command:?} could not run: {error}"));
    assert!(status.success(), "{command:?} failed: {status}");
}
