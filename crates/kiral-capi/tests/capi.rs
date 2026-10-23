// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later
//
// This file is part of KIRAL. The SPDX lines name its copyright holder and its
// license, in the machine-readable form of the REUSE specification: the GNU
// General Public License, version 3 or any later version
// (LICENSES/GPL-3.0-or-later.txt).

//! The C ABI: its header compiles as C23 and as C++26 with every warning Clang
//! has, and declares the functions the library defines and the codes its keys
//! take; an engine opened from files types, converts and commits through it,
//! its wake-up called on the worker's thread; an engine that does not open
//! says why, within the buffer it is given; and NULL is nothing done.
#![allow(
    unsafe_code,
    reason = "calls of the C ABI, each in a block of its own that says why it is sound"
)]

mod common;

use std::collections::BTreeSet;
use std::ffi::{CString, c_char, c_void};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::mpsc::{self, Sender};
use std::time::Duration;

use common::{Random, dictionary, model_file};
use kiral_capi::{
    EngineHandle, KIRAL_KEY_F1, KIRAL_KEY_SPACE, SessionHandle, key_of, kiral_engine_close, kiral_engine_open,
    kiral_session_candidate, kiral_session_candidate_selected, kiral_session_candidates,
    kiral_session_candidates_shown, kiral_session_commit, kiral_session_free, kiral_session_key, kiral_session_new,
    kiral_session_poll, kiral_session_preedit, kiral_session_preedit_cursor, kiral_session_preedit_highlight,
    kiral_session_reset, kiral_session_set_context, kiral_session_set_notify,
};
use kiral_engine::session::{Key, Modifiers};

const HEADER: &str = include_str!("../include/kiral.h");
const LIBRARY: &str = include_str!("../src/lib.rs");

/// The warnings the header is compiled with: every one Clang has, but those
/// about what is not the code -- that it would not compile as an older C or
/// C++, and where a structure has padding -- as stable-diffusion.assembly
/// builds its sources.
const WARNINGS: [&str; 13] = [
    "-Weverything",
    "-Wno-c++98-compat-pedantic",
    "-Wno-pre-c++14-compat-pedantic",
    "-Wno-pre-c++17-compat-pedantic",
    "-Wno-pre-c++20-compat-pedantic",
    "-Wno-pre-c++23-compat-pedantic",
    "-Wno-pre-c++26-compat-pedantic",
    "-Wno-pre-c11-compat",
    "-Wno-pre-c23-compat",
    "-Wno-pre-c2y-compat",
    "-Wno-padded",
    "-Wno-switch-default",
    "-Werror",
];

#[test]
fn the_header_compiles_as_c23_and_cxx26_with_every_warning() {
    let clang = std::env::var_os("KIRAL_CLANG").unwrap_or_else(|| "clang".into());
    let header = Path::new(env!("CARGO_MANIFEST_DIR")).join("include/kiral.h");
    for (language, standard) in [("c", "-std=c23"), ("c++", "-std=c++26")] {
        // A unit of nothing but the header, as a frontend includes it.
        let output = Command::new(&clang)
            .args([standard, "-fsyntax-only"])
            .args(WARNINGS)
            .arg("-include")
            .arg(&header)
            .args(["-x", language, "-"])
            .stdin(Stdio::null())
            .output()
            .expect("Clang, which builds the kernels too (KIRAL_CLANG names another)");
        assert!(
            output.status.success(),
            "{language}: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
}

/// The names of the functions a text declares or defines: each `kiral_` name
/// before a parenthesis, after `fn ` where `rust` is given.
fn functions(text: &str, rust: bool) -> BTreeSet<String> {
    let mut names = BTreeSet::new();
    for (at, _) in text.match_indices("kiral_") {
        if rust && !text[..at].ends_with("extern \"C\" fn ") {
            continue;
        }
        let name: String = text[at..]
            .chars()
            .take_while(|character| character.is_ascii_alphanumeric() || *character == '_')
            .collect();
        if text[at + name.len()..].starts_with('(') {
            names.insert(name);
        }
    }
    names
}

#[test]
fn the_header_declares_what_the_library_defines() {
    let declared = functions(HEADER, false);
    assert_eq!(declared.len(), 17, "{declared:?}");
    assert_eq!(declared, functions(LIBRARY, true));
    // The keys' codes, in the header's order from KIRAL_KEY_SPACE on.
    let names: Vec<&str> = HEADER
        .lines()
        .map(str::trim)
        .filter(|line| line.starts_with("KIRAL_KEY_") && !line.starts_with("KIRAL_KEY_F1"))
        .map(|line| line.trim_end_matches(',').split(' ').next().unwrap_or(line))
        .collect();
    let keys = [
        Key::Space,
        Key::Enter,
        Key::Backspace,
        Key::Delete,
        Key::Escape,
        Key::Tab,
        Key::Left,
        Key::Right,
        Key::Up,
        Key::Down,
        Key::Home,
        Key::End,
        Key::PageUp,
        Key::PageDown,
    ];
    assert_eq!(names.len(), keys.len(), "{names:?}");
    assert!(HEADER.contains(&format!("KIRAL_KEY_SPACE = {KIRAL_KEY_SPACE:#x},")));
    assert!(HEADER.contains(&format!("KIRAL_KEY_F1 = {KIRAL_KEY_F1:#x},")));
    for (code, (name, key)) in (KIRAL_KEY_SPACE..).zip(names.iter().zip(keys)) {
        assert_eq!(key_of(code), Some(key), "{name}");
    }
    assert_eq!(key_of(KIRAL_KEY_F1), Some(Key::Function(1)));
    assert_eq!(key_of(KIRAL_KEY_F1 + 23), Some(Key::Function(24)));
    assert_eq!(key_of(KIRAL_KEY_F1 + 24), None);
    assert_eq!(key_of(KIRAL_KEY_SPACE + 14), None);
    assert_eq!(key_of(0xd800), None, "a surrogate");
    assert_eq!(key_of(u32::from('か')), Some(Key::Character('か')));
    for (name, modifiers) in [
        ("KIRAL_SHIFT = 1u << 0,", Modifiers::SHIFT),
        ("KIRAL_CONTROL = 1u << 1,", Modifiers::CONTROL),
        ("KIRAL_ALT = 1u << 2,", Modifiers::ALT),
        ("KIRAL_SUPER = 1u << 3,", Modifiers::SUPER),
    ] {
        assert!(HEADER.contains(name), "{name}");
        let bit: u32 = name
            .rsplit("<< ")
            .next()
            .and_then(|bit| bit.trim_end_matches(',').parse().ok())
            .expect("a bit");
        assert_eq!(modifiers, Modifiers(1 << bit), "{name}");
    }
}

/// A file of bytes in the temporary directory, removed when it is dropped.
struct Temporary(PathBuf);

impl Temporary {
    fn new(name: &str, bytes: &[u8]) -> Self {
        let path = std::env::temp_dir().join(format!("kiral-capi-{}-{name}", std::process::id()));
        std::fs::write(&path, bytes).expect("a temporary file");
        Self(path)
    }

    fn c_path(&self) -> CString {
        CString::new(self.0.to_str().expect("a UTF-8 path")).expect("no NUL")
    }
}

impl Drop for Temporary {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

/// Opens an engine, its error written to a buffer of `size` bytes.
fn open(dictionary: Option<&CString>, model: Option<&CString>, size: usize) -> (*mut EngineHandle, String) {
    let mut error: Vec<c_char> = vec![0x55; size.max(1)];
    let pointer = |path: Option<&CString>| path.map_or(std::ptr::null(), |path| path.as_ptr());
    // SAFETY: the paths are NULL or NUL-terminated, and the buffer has `size`
    // bytes, all alive for the call.
    let engine = unsafe { kiral_engine_open(pointer(dictionary), pointer(model), error.as_mut_ptr(), size) };
    let written: Vec<u8> = error
        .iter()
        .take_while(|&&byte| byte != 0)
        .map(|&byte| byte.to_ne_bytes()[0])
        .collect();
    (engine, String::from_utf8(written).expect("UTF-8"))
}

/// A string the C ABI returned, and its length, as Rust's.
fn string_of(pointer: *const c_char, length: usize) -> String {
    assert!(!pointer.is_null());
    // SAFETY: the C ABI returns `length` bytes and a NUL, alive until the next
    // call that changes the session.
    let bytes = unsafe { std::slice::from_raw_parts(pointer.cast::<u8>(), length + 1) };
    assert_eq!(bytes[length], 0, "NUL-terminated");
    String::from_utf8(bytes[..length].to_vec()).expect("UTF-8")
}

fn preedit(session: *const SessionHandle) -> String {
    let mut length = 0;
    // SAFETY: a live session, and a writable length.
    let pointer = unsafe { kiral_session_preedit(session, &raw mut length) };
    string_of(pointer, length)
}

fn commit(session: *const SessionHandle) -> String {
    let mut length = 0;
    // SAFETY: a live session, and a writable length.
    let pointer = unsafe { kiral_session_commit(session, &raw mut length) };
    string_of(pointer, length)
}

fn candidates(session: *const SessionHandle) -> Vec<String> {
    // SAFETY: a live session.
    let count = unsafe { kiral_session_candidates(session) };
    (0..count)
        .map(|index| {
            let mut length = 0;
            // SAFETY: a live session, an index below the count, and a writable
            // length.
            let pointer = unsafe { kiral_session_candidate(session, index, &raw mut length) };
            string_of(pointer, length)
        })
        .collect()
}

fn typed(session: *mut SessionHandle, keys: &str) {
    for key in keys.chars() {
        // SAFETY: a live session of this thread's.
        assert!(unsafe { kiral_session_key(session, u32::from(key), 0) }, "{key}");
    }
}

fn key(session: *mut SessionHandle, code: u32) -> bool {
    // SAFETY: a live session of this thread's.
    unsafe { kiral_session_key(session, code, 0) }
}

#[test]
fn an_engine_without_a_model_types_converts_and_commits() {
    let mut random = Random(91);
    let file = Temporary::new("dictionary-alone.kdic", &dictionary(&mut random));
    let (engine, error) = open(Some(&file.c_path()), None, 256);
    assert!(!engine.is_null(), "{error}");
    // SAFETY: an engine just opened.
    let session = unsafe { kiral_session_new(engine) };
    assert!(!session.is_null());
    let context = "左の文";
    // SAFETY: a live session, and the context's bytes.
    unsafe { kiral_session_set_context(session, context.as_ptr().cast(), context.len()) };
    typed(session, "kan");
    let shown = preedit(session);
    assert!(shown.ends_with('ｎ'), "{shown}");
    // SAFETY: a live session.
    assert_eq!(unsafe { kiral_session_preedit_cursor(session) }, shown.len());
    let (mut start, mut end) = (0, 0);
    // SAFETY: a live session, and writable bounds.
    assert!(!unsafe { kiral_session_preedit_highlight(session, &raw mut start, &raw mut end) });
    typed(session, "a");
    assert!(key(session, KIRAL_KEY_SPACE));
    let texts = candidates(session);
    assert!(texts.len() >= 3, "{texts:?}");
    assert!(texts.contains(&"かな".to_owned()) && texts.contains(&"カナ".to_owned()));
    assert_eq!(preedit(session), texts[0]);
    // SAFETY: a live session, and writable bounds.
    assert!(unsafe { kiral_session_preedit_highlight(session, &raw mut start, &raw mut end) });
    assert_eq!((start, end), (0, texts[0].len()));
    // SAFETY: a live session.
    assert!(!unsafe { kiral_session_candidates_shown(session) });
    assert!(key(session, KIRAL_KEY_SPACE));
    // SAFETY: a live session.
    assert!(unsafe { kiral_session_candidates_shown(session) });
    // SAFETY: a live session.
    assert_eq!(unsafe { kiral_session_candidate_selected(session) }, 1);
    let mut length = 0;
    // SAFETY: a live session, an index past the last, and a writable length.
    assert!(unsafe { kiral_session_candidate(session, texts.len(), &raw mut length) }.is_null());
    assert!(key(session, KIRAL_KEY_SPACE + 1), "Enter");
    assert_eq!(commit(session), texts[1]);
    assert_eq!(preedit(session), "");
    // SAFETY: a live session.
    assert_eq!(unsafe { kiral_session_candidates(session) }, 0);
    // Keys the session does not take: Space with nothing typed, Control.
    assert!(!key(session, KIRAL_KEY_SPACE));
    // SAFETY: a live session of this thread's.
    assert!(!unsafe { kiral_session_key(session, u32::from('a'), 1 << 1) });
    assert!(!key(session, 0xd800), "no key");
    typed(session, "ka");
    // SAFETY: a live session of this thread's.
    unsafe { kiral_session_reset(session) };
    assert_eq!(preedit(session), "");
    // SAFETY: a live session of this thread's; nothing to take without a model.
    assert!(!unsafe { kiral_session_poll(session) });
    // SAFETY: the session, then the engine, each freed once.
    unsafe { kiral_session_free(session) };
    // SAFETY: as above.
    unsafe { kiral_engine_close(engine) };
}

/// The wake-up the tests give: it sends the name of its thread on the channel
/// its data points to.
unsafe extern "C" fn woken(data: *mut c_void) {
    // SAFETY: the test gives a sender that outlives the session.
    let sender = unsafe { &*data.cast::<Sender<Option<String>>>() };
    let _ = sender.send(std::thread::current().name().map(str::to_owned));
}

#[test]
fn an_engine_with_a_model_wakes_its_frontend() {
    let mut random = Random(92);
    let dictionary = Temporary::new("dictionary.kdic", &dictionary(&mut random));
    let model = Temporary::new("model.kmdl", &model_file(&mut random));
    let (engine, error) = open(Some(&dictionary.c_path()), Some(&model.c_path()), 256);
    assert!(!engine.is_null(), "{error}");
    // SAFETY: an engine just opened.
    let session = unsafe { kiral_session_new(engine) };
    let (sender, wake_ups) = mpsc::channel::<Option<String>>();
    let sender = Box::new(sender);
    // SAFETY: a live session; the wake-up may be called on another thread with
    // the sender, which outlives the session.
    unsafe { kiral_session_set_notify(session, Some(woken), (&raw const *sender).cast_mut().cast()) };
    typed(session, "ka");
    let thread = wake_ups.recv_timeout(Duration::from_secs(5)).expect("woken");
    assert_eq!(thread.as_deref(), Some("kiral-worker"));
    // SAFETY: a live session of this thread's.
    assert!(unsafe { kiral_session_poll(session) });
    assert_ne!(preedit(session), "");
    // SAFETY: a live session of this thread's; no wake-up from here on.
    unsafe { kiral_session_set_notify(session, None, std::ptr::null_mut()) };
    typed(session, "na");
    assert!(
        wake_ups.recv_timeout(Duration::from_millis(300)).is_err(),
        "no wake-up once it is taken away"
    );
    // SAFETY: a live session of this thread's.
    let refined = unsafe { kiral_session_poll(session) };
    assert!(refined, "the refinement taken all the same");
    // SAFETY: the session, then the engine, each freed once.
    unsafe { kiral_session_free(session) };
    // SAFETY: as above.
    unsafe { kiral_engine_close(engine) };
    drop(sender);
}

#[test]
fn an_engine_that_does_not_open_says_why_and_null_is_nothing() {
    let missing = CString::new("/nonexistent/kiral.kdic").expect("no NUL");
    let (engine, error) = open(Some(&missing), None, 256);
    assert!(engine.is_null());
    assert!(error.starts_with("/nonexistent/kiral.kdic: "), "{error}");
    let (engine, error) = open(Some(&missing), None, 5);
    assert!(engine.is_null());
    assert_eq!(error, "/non", "cut within the buffer, a NUL at its end");
    let (engine, error) = open(None, None, 64);
    assert!(engine.is_null());
    assert_eq!(error, "no dictionary");
    let mut random = Random(93);
    let file = Temporary::new("not-a-model.kdic", &dictionary(&mut random));
    let (engine, error) = open(Some(&file.c_path()), Some(&file.c_path()), 256);
    assert!(engine.is_null());
    assert!(error.starts_with("the model: "), "{error}");
    let (engine, _) = open(Some(&file.c_path()), Some(&file.c_path()), 0);
    assert!(engine.is_null(), "no buffer, no message");
    // NULL is nothing done.
    // SAFETY: NULL, which every function takes.
    assert!(unsafe { kiral_session_new(std::ptr::null_mut()) }.is_null());
    assert!(!key(std::ptr::null_mut(), u32::from('a')));
    assert_eq!(preedit(std::ptr::null()), "");
    assert_eq!(commit(std::ptr::null()), "");
    assert_eq!(candidates(std::ptr::null()).len(), 0);
    // SAFETY: NULL, which every function takes.
    unsafe {
        kiral_session_free(std::ptr::null_mut());
    }
    // SAFETY: as above.
    unsafe { kiral_engine_close(std::ptr::null_mut()) };
}
