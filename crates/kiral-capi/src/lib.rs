// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later
//
// This file is part of KIRAL. The SPDX lines name its copyright holder and its
// license, in the machine-readable form of the REUSE specification: the GNU
// General Public License, version 3 or any later version
// (LICENSES/GPL-3.0-or-later.txt).

//! KIRAL's C ABI, which `include/kiral.h` declares: an engine of a dictionary
//! file and a model file, both mapped ([`kiral_engine::session::Worker`]), and
//! its sessions ([`kiral_engine::session::Session`]), for the frontends whose
//! framework is reached from C or C++ -- Fcitx 5's addon. The crate builds a
//! static library that holds them.
//!
//! Each function takes what C gives it -- pointers it checks for NULL, strings
//! with their lengths -- and makes the one unsafe operation of each block on
//! what the header asks of the caller, which the block's comment names. A panic
//! does not unwind into C: a session's call that panics resets the session and
//! returns as for nothing done. The strings a session returns are kept with it,
//! NUL-terminated, until the next call that changes it.
#![allow(
    unsafe_code,
    reason = "the C ABI: pointers from C, each used in a block of its own that says why it is sound"
)]

use std::ffi::{CStr, c_char, c_void};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::PathBuf;
use std::sync::{Arc, Mutex, PoisonError};

use kiral_engine::engine::{Notify, Settings};
use kiral_engine::session::{Bytes, Key, Modifiers, Session, Worker};
use kiral_sys::map::Map;

/// `KIRAL_KEY_SPACE`, the first of the keys that type no character.
pub const KIRAL_KEY_SPACE: u32 = 0x11_0000;
/// `KIRAL_KEY_F1`; F1 to F24 are `KIRAL_KEY_F1 + n - 1`.
pub const KIRAL_KEY_F1: u32 = 0x11_0100;

/// The keys that type no character, from `KIRAL_KEY_SPACE` on, in the
/// header's order.
const KEYS: [Key; 14] = [
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

/// The key of a code the header gives: a key that types no character, a
/// function key, or a character's scalar value; none for any other code.
#[must_use]
pub fn key_of(code: u32) -> Option<Key> {
    if let Some(at) = code.checked_sub(KIRAL_KEY_SPACE) {
        if let Some(key) = usize::try_from(at).ok().and_then(|at| KEYS.get(at)) {
            return Some(*key);
        }
        return code
            .checked_sub(KIRAL_KEY_F1)
            .filter(|&function| function < 24)
            .and_then(|function| u8::try_from(function + 1).ok())
            .map(Key::Function);
    }
    char::from_u32(code).map(Key::Character)
}

/// A string as C takes it: its bytes and a NUL.
fn nul_terminated(text: &str) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(text.len() + 1);
    bytes.extend_from_slice(text.as_bytes());
    bytes.push(0);
    bytes
}

/// The string C is given for none: empty, NUL-terminated.
static EMPTY: [u8; 1] = [0];

/// An engine, as C holds it: `kiral_engine`.
#[derive(Debug)]
pub struct EngineHandle {
    worker: Worker,
}

/// A session, as C holds it -- `kiral_session` -- the strings it last
/// returned to C, NUL-terminated, and the frontend's wake-up.
#[derive(Debug)]
pub struct SessionHandle {
    session: Session,
    preedit: Vec<u8>,
    commit: Vec<u8>,
    candidates: Vec<Vec<u8>>,
    wake_up: Arc<WakeUp>,
}

impl SessionHandle {
    fn new(worker: &Worker) -> Self {
        let mut session = Session::new(worker);
        let wake_up = Arc::new(WakeUp::default());
        let called = Arc::clone(&wake_up);
        session.set_notify(Some(Notify(Arc::new(move || called.call()))));
        Self {
            session,
            preedit: EMPTY.to_vec(),
            commit: EMPTY.to_vec(),
            candidates: Vec::new(),
            wake_up,
        }
    }

    /// Keeps what the session shows as C reads it.
    fn keep(&mut self) {
        self.preedit = nul_terminated(&self.session.preedit().text);
        self.commit = nul_terminated(self.session.commit());
        self.candidates = self
            .session
            .candidates()
            .map(|candidates| candidates.texts.iter().map(|text| nul_terminated(text)).collect())
            .unwrap_or_default();
    }
}

/// A path given by C: NULL for none, or a NUL-terminated string, its bytes on
/// Unix and UTF-8 elsewhere.
///
/// # Safety
///
/// `path` is NULL, or a NUL-terminated string that lives for the call.
unsafe fn path_of(path: *const c_char) -> Option<PathBuf> {
    if path.is_null() {
        return None;
    }
    // SAFETY: not NULL, checked above, and NUL-terminated and alive for the
    // call, as the caller says.
    let bytes = unsafe { CStr::from_ptr(path) }.to_bytes();
    #[cfg(unix)]
    {
        use std::os::unix::ffi::OsStrExt;
        Some(PathBuf::from(std::ffi::OsStr::from_bytes(bytes)))
    }
    #[cfg(not(unix))]
    {
        std::str::from_utf8(bytes).ok().map(PathBuf::from)
    }
}

/// Writes a message to a buffer of `size` bytes, NUL-terminated, cut at a
/// character's boundary where it does not fit; nothing to NULL or no bytes.
///
/// # Safety
///
/// `buffer` is NULL, or `size` writable bytes that live for the call.
unsafe fn write_message(message: &str, buffer: *mut c_char, size: usize) {
    if buffer.is_null() || size == 0 {
        return;
    }
    let mut length = message.len().min(size - 1);
    while !message.is_char_boundary(length) {
        length -= 1;
    }
    // SAFETY: `buffer` is not NULL, checked above, and has `size` writable bytes,
    // as the caller says; `length + 1` is at most `size`.
    let written = unsafe { std::slice::from_raw_parts_mut(buffer.cast::<u8>(), length + 1) };
    written[..length].copy_from_slice(&message.as_bytes()[..length]);
    written[length] = 0;
}

/// Gives C a string and its length, but the NUL.
///
/// # Safety
///
/// `length` is NULL, or a writable `size_t`.
unsafe fn give(text: &[u8], length: *mut usize) -> *const c_char {
    if !length.is_null() {
        // SAFETY: not NULL, checked above, and writable, as the caller says.
        unsafe { length.write(text.len().saturating_sub(1)) };
    }
    text.as_ptr().cast()
}

/// Runs a call on a session that C gives: as for nothing done for a NULL one,
/// and for one whose call panics, which is reset.
///
/// # Safety
///
/// `session` is NULL, or a session [`kiral_session_new`] returned, not freed,
/// and used from no other thread.
unsafe fn with_session<R>(session: *mut SessionHandle, nothing: R, call: impl FnOnce(&mut SessionHandle) -> R) -> R {
    // SAFETY: NULL or a live session, used from this thread alone, as the
    // caller says.
    let Some(session) = (unsafe { session.as_mut() }) else {
        return nothing;
    };
    if let Ok(result) = catch_unwind(AssertUnwindSafe(|| call(session))) {
        result
    } else {
        let _ = catch_unwind(AssertUnwindSafe(|| {
            session.session.reset();
            session.keep();
        }));
        nothing
    }
}

/// A session C gives, to read.
///
/// # Safety
///
/// `session` is NULL, or a session [`kiral_session_new`] returned and not freed.
unsafe fn session_of<'s>(session: *const SessionHandle) -> Option<&'s SessionHandle> {
    // SAFETY: NULL or a live session, as the caller says.
    unsafe { session.as_ref() }
}

/// A frontend's wake-up: the function, and the data it is called with.
type WakeUpFunction = unsafe extern "C" fn(data: *mut c_void);

/// The wake-up a frontend gave a session, which the worker calls through, and
/// which the frontend's thread replaces or takes away -- setting another, or
/// freeing the session -- under the lock: once that returns, the wake-up it
/// replaced is not called. The lock is taken on the frontend's thread only
/// then, never on a key, and waits at most for a call of the wake-up, which
/// does no more than schedule the frontend's own.
#[derive(Debug, Default)]
struct WakeUp(Mutex<Option<(WakeUpFunction, WakeUpData)>>);

impl WakeUp {
    fn set(&self, wake_up: Option<(WakeUpFunction, WakeUpData)>) {
        *self.0.lock().unwrap_or_else(PoisonError::into_inner) = wake_up;
    }

    /// Calls the wake-up, on the worker's thread, where there is one.
    fn call(&self) {
        let wake_up = self.0.lock().unwrap_or_else(PoisonError::into_inner);
        if let Some((function, data)) = &*wake_up {
            // SAFETY: the frontend gave the function to be called with its data
            // on the worker's thread until it replaces it or frees the session,
            // which it cannot do while the lock is held, as the header asks.
            unsafe { function(data.0) };
        }
    }
}

/// The data a frontend's wake-up is called with.
#[derive(Debug)]
struct WakeUpData(*mut c_void);

// SAFETY: the header asks the frontend for data that its wake-up may be called
// with on the worker's thread.
unsafe impl Send for WakeUpData {}

impl Drop for SessionHandle {
    /// Takes the wake-up away: none is called once the session is freed.
    fn drop(&mut self) {
        self.wake_up.set(None);
    }
}

/// Opens an engine of a dictionary file and, where its path is not NULL, of a
/// model file, both mapped where they lie, with the default settings; NULL
/// where it does not open, the reason written to `error`.
///
/// # Safety
///
/// The paths are NULL or NUL-terminated strings, and `error` NULL or
/// `error_size` writable bytes, alive for the call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kiral_engine_open(
    dictionary_path: *const c_char,
    model_path: *const c_char,
    error: *mut c_char,
    error_size: usize,
) -> *mut EngineHandle {
    // SAFETY: NULL or a NUL-terminated string alive for the call, as the header
    // asks.
    let dictionary = unsafe { path_of(dictionary_path) };
    // SAFETY: as for the dictionary's path.
    let model = unsafe { path_of(model_path) };
    let opened = catch_unwind(|| open(dictionary, model.as_ref())).unwrap_or_else(|_| Err("a panic".to_owned()));
    match opened {
        Ok(engine) => Box::into_raw(Box::new(engine)),
        Err(message) => {
            // SAFETY: NULL or `error_size` writable bytes, as the header asks.
            unsafe { write_message(&message, error, error_size) };
            std::ptr::null_mut()
        }
    }
}

/// An engine of the files at the paths.
fn open(dictionary: Option<PathBuf>, model: Option<&PathBuf>) -> Result<EngineHandle, String> {
    let map = |path: &PathBuf| {
        Map::open(path)
            .map(Bytes::new)
            .map_err(|error| format!("{}: {error}", path.display()))
    };
    let dictionary = dictionary.ok_or("no dictionary")?;
    let model = model.map(map).transpose()?;
    let worker = Worker::start(map(&dictionary)?, model, Settings::default()).map_err(|error| error.to_string())?;
    Ok(EngineHandle { worker })
}

/// Closes an engine; NULL does nothing.
///
/// # Safety
///
/// `engine` is NULL, or an engine [`kiral_engine_open`] returned and not
/// closed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kiral_engine_close(engine: *mut EngineHandle) {
    if engine.is_null() {
        return;
    }
    // SAFETY: an engine `kiral_engine_open` boxed and C has not closed, as the
    // header asks; the box is dropped once, here.
    let engine = unsafe { Box::from_raw(engine) };
    let _ = catch_unwind(AssertUnwindSafe(|| drop(engine)));
}

/// A new session of an engine; NULL for a NULL engine.
///
/// # Safety
///
/// `engine` is NULL, or an engine [`kiral_engine_open`] returned and not
/// closed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kiral_session_new(engine: *mut EngineHandle) -> *mut SessionHandle {
    // SAFETY: NULL or a live engine, as the header asks.
    let Some(engine) = (unsafe { engine.as_ref() }) else {
        return std::ptr::null_mut();
    };
    catch_unwind(AssertUnwindSafe(|| SessionHandle::new(&engine.worker)))
        .map_or(std::ptr::null_mut(), |session| Box::into_raw(Box::new(session)))
}

/// Frees a session; NULL does nothing.
///
/// # Safety
///
/// `session` is NULL, or a session [`kiral_session_new`] returned and not
/// freed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kiral_session_free(session: *mut SessionHandle) {
    if session.is_null() {
        return;
    }
    // SAFETY: a session `kiral_session_new` boxed and C has not freed, as the
    // header asks; the box is dropped once, here.
    let session = unsafe { Box::from_raw(session) };
    let _ = catch_unwind(AssertUnwindSafe(|| drop(session)));
}

/// The wake-up the engine's worker calls with `data`, on its own thread, once
/// it has given back a refinement of the session's; NULL for none. Once this
/// returns, the wake-up it replaced is not called.
///
/// # Safety
///
/// `session` as for [`kiral_session_key`]; `notify` NULL, or a function that
/// may be called with `data` on another thread until it is replaced or the
/// session freed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kiral_session_set_notify(
    session: *mut SessionHandle,
    notify: Option<unsafe extern "C" fn(data: *mut c_void)>,
    data: *mut c_void,
) {
    let wake_up = notify.map(|function| (function, WakeUpData(data)));
    let call = |session: &mut SessionHandle| session.wake_up.set(wake_up);
    // SAFETY: NULL or a live session of this thread's, as the header asks.
    unsafe { with_session(session, (), call) };
}

/// The text before the cursor: the left context of the next conversion.
///
/// # Safety
///
/// `session` as for [`kiral_session_key`]; `text` NULL, or `length` readable
/// bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kiral_session_set_context(session: *mut SessionHandle, text: *const c_char, length: usize) {
    let context = if text.is_null() || length == 0 {
        String::new()
    } else {
        // SAFETY: not NULL, checked above, and `length` readable bytes, as the
        // header asks.
        let bytes = unsafe { std::slice::from_raw_parts(text.cast::<u8>(), length) };
        String::from_utf8_lossy(bytes).into_owned()
    };
    let call = |session: &mut SessionHandle| session.session.set_context(&context);
    // SAFETY: NULL or a live session of this thread's, as the header asks.
    unsafe { with_session(session, (), call) };
}

/// Types a key, with the modifiers held: whether the session took it.
///
/// # Safety
///
/// `session` is NULL, or a session [`kiral_session_new`] returned, not freed,
/// and used from no other thread.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kiral_session_key(session: *mut SessionHandle, key: u32, modifiers: u32) -> bool {
    let Some(key) = key_of(key) else {
        return false;
    };
    let call = |session: &mut SessionHandle| {
        let taken = session.session.key(key, Modifiers(modifiers));
        session.keep();
        taken
    };
    // SAFETY: NULL or a live session of this thread's, as the header asks.
    unsafe { with_session(session, false, call) }
}

/// Takes the refinements that have come, without waiting: whether what the
/// session shows changed.
///
/// # Safety
///
/// As for [`kiral_session_key`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kiral_session_poll(session: *mut SessionHandle) -> bool {
    let call = |session: &mut SessionHandle| {
        let changed = session.session.poll();
        if changed {
            session.keep();
        }
        changed
    };
    // SAFETY: NULL or a live session of this thread's, as the header asks.
    unsafe { with_session(session, false, call) }
}

/// Deletes what is typed and the left context.
///
/// # Safety
///
/// As for [`kiral_session_key`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kiral_session_reset(session: *mut SessionHandle) {
    let call = |session: &mut SessionHandle| {
        session.session.reset();
        session.keep();
    };
    // SAFETY: NULL or a live session of this thread's, as the header asks.
    unsafe { with_session(session, (), call) };
}

/// What the session shows where the text is typed, and its length.
///
/// # Safety
///
/// `session` is NULL, or a session [`kiral_session_new`] returned and not
/// freed; `length` NULL, or a writable `size_t`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kiral_session_preedit(session: *const SessionHandle, length: *mut usize) -> *const c_char {
    // SAFETY: NULL or a live session, as the header asks.
    let text = unsafe { session_of(session) }.map_or(&EMPTY[..], |session| &session.preedit[..]);
    // SAFETY: NULL or a writable `size_t`, as the header asks.
    unsafe { give(text, length) }
}

/// Where the cursor is in the preedit, in bytes.
///
/// # Safety
///
/// `session` is NULL, or a session [`kiral_session_new`] returned and not
/// freed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kiral_session_preedit_cursor(session: *const SessionHandle) -> usize {
    // SAFETY: NULL or a live session, as the header asks.
    unsafe { session_of(session) }.map_or(0, |session| session.session.preedit().cursor)
}

/// The part of the preedit to highlight, from `start` to `end` in bytes:
/// whether there is one.
///
/// # Safety
///
/// `session` as for [`kiral_session_preedit_cursor`]; `start` and `end` NULL,
/// or writable `size_t`s.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kiral_session_preedit_highlight(
    session: *const SessionHandle,
    start: *mut usize,
    end: *mut usize,
) -> bool {
    // SAFETY: NULL or a live session, as the header asks.
    let Some(range) = (unsafe { session_of(session) }).and_then(|session| session.session.preedit().highlight.clone())
    else {
        return false;
    };
    if !start.is_null() {
        // SAFETY: not NULL, checked above, and writable, as the header asks.
        unsafe { start.write(range.start) };
    }
    if !end.is_null() {
        // SAFETY: not NULL, checked above, and writable, as the header asks.
        unsafe { end.write(range.end) };
    }
    true
}

/// The text the last key committed, and its length.
///
/// # Safety
///
/// As for [`kiral_session_preedit`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kiral_session_commit(session: *const SessionHandle, length: *mut usize) -> *const c_char {
    // SAFETY: NULL or a live session, as the header asks.
    let text = unsafe { session_of(session) }.map_or(&EMPTY[..], |session| &session.commit[..]);
    // SAFETY: NULL or a writable `size_t`, as the header asks.
    unsafe { give(text, length) }
}

/// The number of the conversion's candidates.
///
/// # Safety
///
/// As for [`kiral_session_preedit_cursor`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kiral_session_candidates(session: *const SessionHandle) -> usize {
    // SAFETY: NULL or a live session, as the header asks.
    unsafe { session_of(session) }.map_or(0, |session| session.candidates.len())
}

/// A candidate, and its length; NULL past the last.
///
/// # Safety
///
/// As for [`kiral_session_preedit`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kiral_session_candidate(
    session: *const SessionHandle,
    index: usize,
    length: *mut usize,
) -> *const c_char {
    // SAFETY: NULL or a live session, as the header asks.
    let Some(text) = (unsafe { session_of(session) }).and_then(|session| session.candidates.get(index)) else {
        return std::ptr::null();
    };
    // SAFETY: NULL or a writable `size_t`, as the header asks.
    unsafe { give(text, length) }
}

/// The candidate the preedit shows.
///
/// # Safety
///
/// As for [`kiral_session_preedit_cursor`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kiral_session_candidate_selected(session: *const SessionHandle) -> usize {
    // SAFETY: NULL or a live session, as the header asks.
    unsafe { session_of(session) }
        .and_then(|session| session.session.candidates())
        .map_or(0, |candidates| candidates.selected)
}

/// Whether the list of candidates is to be shown.
///
/// # Safety
///
/// As for [`kiral_session_preedit_cursor`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kiral_session_candidates_shown(session: *const SessionHandle) -> bool {
    // SAFETY: NULL or a live session, as the header asks.
    unsafe { session_of(session) }
        .and_then(|session| session.session.candidates())
        .is_some_and(|candidates| candidates.shown)
}
