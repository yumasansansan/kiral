// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later
//
// This file is part of KIRAL. The SPDX lines name its copyright holder and its
// license, in the machine-readable form of the REUSE specification: the GNU
// General Public License, version 3 or any later version
// (LICENSES/GPL-3.0-or-later.txt).

//! A file mapped is its bytes, read from any thread; an empty file is no bytes,
//! and a file that is not there an error. Where files are replaced by renaming
//! a new one over them, a mapped file keeps its bytes.

use std::path::PathBuf;

use kiral_sys::map::Map;

/// A file of the temporary directory, of a name of this process's and this
/// test's, holding `bytes`.
fn temporary(name: &str, bytes: &[u8]) -> PathBuf {
    let path = std::env::temp_dir().join(format!("kiral-map-{}-{name}", std::process::id()));
    std::fs::write(&path, bytes).expect("a temporary file");
    path
}

/// The bytes i mod 251, of a length that ends within a page.
fn bytes() -> Vec<u8> {
    (0..100_003usize)
        .map(|i| u8::try_from(i % 251).expect("a byte"))
        .collect()
}

#[test]
fn a_mapped_file_is_its_bytes_from_any_thread() {
    let bytes = bytes();
    let path = temporary("bytes", &bytes);
    let map = Map::open(&path).expect("mapped");
    assert_eq!(map.bytes(), bytes);
    std::thread::scope(|scope| {
        let halves = map.bytes().split_at(map.bytes().len() / 2);
        let sums =
            [halves.0, halves.1].map(|half| scope.spawn(move || half.iter().map(|&byte| u64::from(byte)).sum::<u64>()));
        let sum: u64 = sums.into_iter().map(|sum| sum.join().expect("a sum")).sum();
        assert_eq!(sum, bytes.iter().map(|&byte| u64::from(byte)).sum::<u64>());
    });
    drop(map);
    std::fs::remove_file(&path).expect("removed");
}

#[test]
fn an_empty_file_is_no_bytes() {
    let path = temporary("empty", &[]);
    let map = Map::open(&path).expect("mapped");
    assert_eq!(map.bytes(), []);
    drop(map);
    std::fs::remove_file(&path).expect("removed");
}

#[test]
fn a_file_not_there_is_an_error() {
    let path = std::env::temp_dir().join(format!("kiral-map-{}-missing", std::process::id()));
    let error = Map::open(&path).expect_err("no file");
    assert_eq!(error.kind(), std::io::ErrorKind::NotFound);
}

#[cfg(unix)]
#[test]
fn a_file_replaced_by_renaming_stays_as_it_was_mapped() {
    let bytes = bytes();
    let path = temporary("replaced", &bytes);
    let map = Map::open(&path).expect("mapped");
    let new = temporary("replacement", b"another file");
    std::fs::rename(&new, &path).expect("renamed over");
    assert_eq!(map.bytes(), bytes);
    std::fs::remove_file(&path).expect("removed");
    assert_eq!(map.bytes(), bytes);
}
