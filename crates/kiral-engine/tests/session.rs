// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later
//
// This file is part of KIRAL. The SPDX lines name its copyright holder and its
// license, in the machine-readable form of the REUSE specification: the GNU
// General Public License, version 3 or any later version
// (LICENSES/GPL-3.0-or-later.txt).

//! Scripted sessions, keys in and what is shown, committed and offered out,
//! with a small zenz of random weights from its model file: romaji typed shows
//! the draft of its kana at once, a path of their lattice, and the refinement
//! of the latest reading when it comes, never of one the keys have passed;
//! Space converts, its candidates step, show and commit; Backspace, Escape, F6
//! and F7 do what they do while typing and converting; the keys a session does
//! not take pass; the wake-up is called and the refinement taken; and an
//! engine without a model, or with its worker gone, shows its drafts at once.

mod common;

use std::sync::{Arc, mpsc};
use std::time::{Duration, Instant};

use common::{Random, dictionary, texts, zenz};
use kiral_dict::Dictionary;
use kiral_engine::engine::{Notify, Settings};
use kiral_engine::session::{Bytes, Key, Modifiers, Session, StartError, Worker};
use kiral_lattice::Lattice;
use kiral_model::file::Metadata;

/// Long enough for any refinement of the small model to come.
const LONG: Duration = Duration::from_secs(5);

fn settings() -> Settings {
    Settings {
        capacity: 128,
        deadline: LONG,
        ..Settings::default()
    }
}

/// The model file of a small zenz of random weights.
fn model_file(random: &mut Random) -> Vec<u8> {
    let metadata = Metadata {
        name: "test",
        license: "CC0-1.0",
        source: "made by the test",
        notice: "",
    };
    zenz(random).to_file(&metadata).expect("a model file")
}

/// Whether a text is the text of a path of a reading's lattice.
fn is_path(dictionary: &Dictionary<'_>, reading: &str, text: &str) -> bool {
    texts(&Lattice::new(dictionary, reading))
        .iter()
        .any(|path| path == text)
}

/// Keys as a session shows them while they wait: full-width.
fn full_width(keys: &str) -> String {
    keys.chars()
        .map(|key| char::from_u32(u32::from(key) + 0xfee0).expect("a full-width form"))
        .collect()
}

/// Types keys that the session takes.
fn typed(session: &mut Session, keys: &str) {
    for key in keys.chars() {
        assert!(session.key(Key::Character(key), Modifiers::NONE), "{key}");
    }
}

fn key(session: &mut Session, key: Key) -> bool {
    session.key(key, Modifiers::NONE)
}

/// An engine of a random dictionary and, where asked, a random model, and
/// the dictionary's bytes.
fn engine(seed: u64, with_model: bool) -> (Worker, Vec<u8>) {
    let mut random = Random(seed);
    let bytes = dictionary(&mut random);
    let model = with_model.then(|| Bytes::new(model_file(&mut random)));
    let worker = Worker::start(Bytes::new(bytes.clone()), model, settings()).expect("an engine");
    (worker, bytes)
}

#[test]
fn romaji_shows_the_draft_at_once_and_its_refinement_when_it_comes() {
    let (worker, bytes) = engine(71, true);
    let dictionary = Dictionary::parse(&bytes).expect("a dictionary");
    let mut session = Session::new(&worker);
    let mut reading = String::new();
    for (keys, kana) in [("ka", "か"), ("na", "な"), ("nn", "ん"), ("ko", "こ")] {
        let (first, second) = keys.split_at(1);
        let before = (session.preedit().text.clone(), session.generation());
        typed(&mut session, first);
        // A key that waits for a rule changes no reading: what is shown, and
        // the full-width key after it.
        assert_eq!(session.generation(), before.1);
        assert_eq!(session.preedit().text, before.0 + &full_width(first));
        typed(&mut session, second);
        reading.push_str(kana);
        let shown = session.preedit().text.clone();
        assert!(is_path(&dictionary, &reading, &shown), "{reading}: the draft {shown}");
        assert!(session.preedit().highlight.is_none());
        assert_eq!(session.preedit().cursor, shown.len());
        assert!(session.wait_until(Instant::now() + LONG), "{reading}: refined");
        let refined = session.refinement().expect("a refinement");
        assert_eq!(refined.generation, session.generation());
        let text = &refined.refined.as_ref().expect("refined").text;
        assert_eq!(text, &session.preedit().text);
        assert!(is_path(&dictionary, &reading, text), "{reading}: {text}");
    }
}

#[test]
fn space_converts_and_its_candidates_step_show_and_commit() {
    let (worker, bytes) = engine(72, true);
    let dictionary = Dictionary::parse(&bytes).expect("a dictionary");
    let mut session = Session::new(&worker);
    typed(&mut session, "kanannko");
    assert!(key(&mut session, Key::Space));
    let candidates = session.candidates().expect("candidates").clone();
    assert!(is_path(&dictionary, "かなんこ", &candidates.texts[0]));
    assert!(candidates.texts.iter().any(|text| text == "かなんこ"));
    assert!(candidates.texts.iter().any(|text| text == "カナンコ"));
    assert_eq!((candidates.selected, candidates.shown), (0, false));
    assert_eq!(session.preedit().text, candidates.texts[0]);
    assert_eq!(session.preedit().highlight, Some(0..candidates.texts[0].len()));
    let refined = session
        .refinement()
        .expect("a conversion refined by its deadline of 5 s");
    assert_eq!(refined.generation, session.generation());
    assert!(key(&mut session, Key::Space));
    let shown = session.candidates().expect("candidates");
    assert_eq!((shown.selected, shown.shown), (1, true));
    assert!(key(&mut session, Key::Up));
    assert_eq!(session.candidates().expect("candidates").selected, 0);
    assert!(key(&mut session, Key::Down));
    assert!(key(&mut session, Key::Enter));
    assert_eq!(session.commit(), candidates.texts[1]);
    assert_eq!(session.preedit().text, "");
    assert!(session.candidates().is_none());
    // The next key commits nothing more.
    typed(&mut session, "k");
    assert_eq!(session.commit(), "");
    assert_eq!(session.preedit().text, "ｋ");
}

#[test]
fn backspace_escape_and_the_function_keys() {
    let (worker, bytes) = engine(73, true);
    let dictionary = Dictionary::parse(&bytes).expect("a dictionary");
    let mut session = Session::new(&worker);
    typed(&mut session, "kana");
    assert!(key(&mut session, Key::Backspace));
    assert!(is_path(&dictionary, "か", &session.preedit().text));
    assert!(key(&mut session, Key::Backspace));
    assert_eq!(session.preedit().text, "");
    assert!(!key(&mut session, Key::Backspace), "nothing typed: the application's");
    typed(&mut session, "kana");
    assert!(key(&mut session, Key::Escape));
    assert_eq!(session.preedit().text, "");
    typed(&mut session, "kana");
    assert!(key(&mut session, Key::Function(7)));
    assert_eq!(session.preedit().text, "カナ");
    assert!(key(&mut session, Key::Escape));
    assert!(session.candidates().is_none());
    assert!(is_path(&dictionary, "かな", &session.preedit().text));
    assert!(key(&mut session, Key::Function(6)));
    assert!(key(&mut session, Key::Enter));
    assert_eq!(session.commit(), "かな");
    // Keys that wait are taken as they are by Enter: ｋ, a k to convert.
    typed(&mut session, "kak");
    assert!(session.preedit().text.ends_with('ｋ'));
    assert!(key(&mut session, Key::Enter));
    assert!(is_path(&dictionary, "かk", session.commit()), "{}", session.commit());
    // A conversion's Backspace goes back to what is typed.
    typed(&mut session, "kana");
    assert!(key(&mut session, Key::Space));
    assert!(key(&mut session, Key::Backspace));
    assert!(session.candidates().is_none());
    assert!(is_path(&dictionary, "かな", &session.preedit().text));
}

#[test]
fn a_character_during_a_conversion_commits_it_and_a_digit_chooses() {
    let (worker, _) = engine(74, true);
    let mut session = Session::new(&worker);
    typed(&mut session, "kana");
    assert!(key(&mut session, Key::Space));
    let conversion = session.preedit().text.clone();
    typed(&mut session, "k");
    assert_eq!(session.commit(), conversion);
    assert_eq!(session.preedit().text, "ｋ");
    typed(&mut session, "o");
    assert!(key(&mut session, Key::Space));
    // A digit before the list is shown types.
    let conversion = session.preedit().text.clone();
    typed(&mut session, "1");
    assert_eq!(session.commit(), conversion);
    assert!(key(&mut session, Key::Escape));
    typed(&mut session, "kana");
    assert!(key(&mut session, Key::Space));
    assert!(key(&mut session, Key::Space));
    let texts = session.candidates().expect("candidates").texts.clone();
    typed(&mut session, "3");
    assert_eq!(session.commit(), texts[2]);
    assert!(session.candidates().is_none() && session.preedit().text.is_empty());
    // A candidate chosen otherwise than by a key -- clicked -- commits too.
    assert!(!session.choose(0), "no conversion");
    typed(&mut session, "kana");
    assert!(key(&mut session, Key::Space));
    let texts = session.candidates().expect("candidates").texts.clone();
    assert!(!session.choose(texts.len()));
    assert!(session.choose(texts.len() - 1));
    assert_eq!(session.commit(), texts[texts.len() - 1]);
    assert!(session.candidates().is_none() && session.preedit().text.is_empty());
}

#[test]
fn keys_the_session_does_not_take_pass() {
    let (worker, _) = engine(75, false);
    let mut session = Session::new(&worker);
    for unused in [
        Key::Space,
        Key::Enter,
        Key::Backspace,
        Key::Left,
        Key::Escape,
        Key::Function(6),
    ] {
        assert!(!key(&mut session, unused), "{unused:?}");
    }
    assert!(!session.key(Key::Character('a'), Modifiers::CONTROL));
    assert!(!session.key(Key::Character('a'), Modifiers::ALT | Modifiers::SHIFT));
    assert_eq!(session.preedit().text, "");
    typed(&mut session, "ka");
    let shown = session.preedit().text.clone();
    assert!(!session.key(Key::Character('c'), Modifiers::CONTROL));
    assert!(!session.key(Key::Character('s'), Modifiers::SUPER));
    assert_eq!(session.preedit().text, shown);
    // An uppercase letter is taken as Shift makes it, and Space after it types
    // a space.
    assert!(session.key(Key::Character('P'), Modifiers::SHIFT));
    assert!(key(&mut session, Key::Space));
    assert!(session.candidates().is_none());
    assert!(key(&mut session, Key::Enter));
    assert!(session.commit().ends_with("P "), "{}", session.commit());
    // The cursor's keys are taken while something is typed, and do nothing yet.
    typed(&mut session, "ka");
    let shown = session.preedit().clone();
    assert!(key(&mut session, Key::Left));
    assert_eq!(session.preedit(), &shown);
}

#[test]
fn refinements_of_readings_the_keys_have_passed_never_show() {
    let (worker, bytes) = engine(76, true);
    let dictionary = Dictionary::parse(&bytes).expect("a dictionary");
    let mut session = Session::new(&worker);
    for _ in 0..10 {
        typed(&mut session, "kanannko");
        assert!(session.wait_until(Instant::now() + LONG), "refined");
        assert_eq!(
            session.refinement().map(|outcome| outcome.generation),
            Some(session.generation())
        );
        assert!(is_path(&dictionary, "かなんこ", &session.preedit().text));
        std::thread::sleep(Duration::from_millis(5));
        session.poll();
        assert_eq!(
            session.refinement().map(|outcome| outcome.generation),
            Some(session.generation())
        );
        assert!(is_path(&dictionary, "かなんこ", &session.preedit().text));
        session.reset();
        assert_eq!(session.preedit().text, "");
    }
}

#[test]
fn the_wake_up_is_called_and_poll_takes_the_refinement() {
    let (worker, _) = engine(77, true);
    let mut session = Session::new(&worker);
    let (sender, woken) = mpsc::channel();
    session.set_notify(Some(Notify(Arc::new(move || {
        let _ = sender.send(std::thread::current().name().map(str::to_owned));
    }))));
    typed(&mut session, "ka");
    assert!(session.refinement().is_none());
    let thread = woken.recv_timeout(LONG).expect("woken");
    assert_eq!(thread.as_deref(), Some("kiral-worker"));
    assert!(session.poll());
    let refined = session.refinement().expect("taken").refined.as_ref().expect("refined");
    assert_eq!(refined.text, session.preedit().text);
    assert!(!session.poll(), "taken once");
}

#[test]
fn without_a_model_or_with_its_worker_gone_the_drafts_show_at_once() {
    let (worker, bytes) = engine(78, false);
    let dictionary = Dictionary::parse(&bytes).expect("a dictionary");
    assert!(!worker.has_model());
    let mut session = Session::new(&worker);
    typed(&mut session, "kana");
    let draft = session.preedit().text.clone();
    assert!(is_path(&dictionary, "かな", &draft));
    let started = Instant::now();
    assert!(key(&mut session, Key::Space));
    assert!(started.elapsed() < LONG / 5, "no wait for a model there is not");
    assert_eq!(session.candidates().expect("candidates").texts[0], draft);
    assert!(session.refinement().is_none());
    let (worker, bytes) = engine(79, true);
    let dictionary = Dictionary::parse(&bytes).expect("a dictionary");
    let mut session = Session::new(&worker);
    drop(worker);
    typed(&mut session, "kana");
    let started = Instant::now();
    assert!(key(&mut session, Key::Space));
    assert!(started.elapsed() < LONG / 5, "no wait for a worker that has gone");
    assert!(is_path(&dictionary, "かな", &session.preedit().text));
}

#[test]
fn an_engine_of_what_are_not_its_files_does_not_start() {
    let mut random = Random(80);
    let bytes = dictionary(&mut random);
    assert!(matches!(
        Worker::start(Bytes::new(vec![0u8; 10]), None, settings()),
        Err(StartError::Dictionary(_))
    ));
    assert!(matches!(
        Worker::start(Bytes::new(bytes), Some(Bytes::new(vec![1u8, 2, 3])), settings()),
        Err(StartError::Model(_))
    ));
}
