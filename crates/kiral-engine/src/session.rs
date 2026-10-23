// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later
//
// This file is part of KIRAL. The SPDX lines name its copyright holder and its
// license, in the machine-readable form of the REUSE specification: the GNU
// General Public License, version 3 or any later version
// (LICENSES/GPL-3.0-or-later.txt).

//! Sessions: what a frontend's input context types into, a key at a time, and
//! the engine they share, which owns its files and its worker.
//!
//! A session composes romaji into kana ([`crate::romaji`]) and shows the
//! conversion as it is typed: the dictionary's draft of the kana at once, the
//! model's refinement in its place when it comes, and the keys that wait for a
//! rule after it, full-width. Space converts: it waits for the refinement of
//! the whole reading up to the deadline, shows the conversion highlighted, and
//! keeps the candidates behind it -- the conversion, the dictionary's own, the
//! kana and the katakana, each once -- which Space again, Down and Up show and
//! step through, and a digit chooses while they are shown. Enter commits what
//! is shown; a character typed during a conversion commits it and begins the
//! next. Backspace deletes a character, or goes back from a conversion to what
//! is typed, as Escape does, which deletes what is typed otherwise; F6 and F7
//! take the kana and the katakana. Keys held with Control, Alt or Super are
//! the application's, and so are keys while nothing is typed but characters.
//!
//! The text committed is the left context of the next conversion, unless the
//! frontend gives the text before the cursor; the model reads its last 40
//! characters. The engine's worker gives each session's refinements back to
//! it, and calls its wake-up, on the worker's thread: the frontend has its own
//! thread take them ([`Session::poll`]), and the input thread never waits for
//! the model but on Space.

use std::fmt;
use std::ops::Range;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{self, Sender};
use std::thread::JoinHandle;
use std::time::Instant;

use kiral_dict::Checked;
use kiral_lattice::Lattice;
use kiral_model::file::ModelFile;
use kiral_nn::zenz::{self, Zenz};

use crate::engine::{Drafts, Message, Notify, Outcome, Settings, work};
use crate::refine::katakana;
use crate::romaji::Composer;

/// Bytes that their owner keeps for as long as anything reads them: a mapped
/// file, or a vector.
#[derive(Clone)]
pub struct Bytes(Arc<dyn AsRef<[u8]> + Send + Sync>);

impl Bytes {
    /// The bytes of an owner.
    pub fn new(owner: impl AsRef<[u8]> + Send + Sync + 'static) -> Self {
        Self(Arc::new(owner))
    }
}

impl AsRef<[u8]> for Bytes {
    fn as_ref(&self) -> &[u8] {
        (*self.0).as_ref()
    }
}

impl fmt::Debug for Bytes {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "Bytes({})", self.as_ref().len())
    }
}

/// Why an engine does not start.
#[derive(Debug)]
pub enum StartError {
    /// The dictionary's bytes are not a dictionary file.
    Dictionary(kiral_dict::Error),
    /// The model's bytes are not a model file of zenz.
    Model(String),
    /// The system does not start the worker's thread.
    Thread(std::io::Error),
}

impl fmt::Display for StartError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Dictionary(error) => write!(formatter, "the dictionary: {error}"),
            Self::Model(error) => write!(formatter, "the model: {error}"),
            Self::Thread(error) => write!(formatter, "the worker's thread: {error}"),
        }
    }
}

impl std::error::Error for StartError {}

/// What the sessions of an engine share.
#[derive(Debug)]
struct Shared {
    dictionary: Checked<Bytes>,
    settings: Settings,
    generation: Arc<AtomicU64>,
    /// The worker's jobs; none without a model.
    jobs: Option<Sender<Message>>,
}

/// An engine that owns its files: a dictionary, checked once, and the worker
/// that refines with a model, where there is one; its sessions share it.
#[derive(Debug)]
pub struct Worker {
    shared: Arc<Shared>,
    thread: Option<JoinHandle<()>>,
}

impl Worker {
    /// Starts an engine of a dictionary file's bytes and, where they are given,
    /// a model file's -- zenz in integers, which the worker refines with on a
    /// thread of its own. Without a model, the dictionary's drafts are all its
    /// sessions show.
    ///
    /// # Errors
    ///
    /// Where the dictionary's bytes are not a dictionary file, the model's not
    /// a model file of zenz, or the worker's thread does not start.
    pub fn start(dictionary: Bytes, model: Option<Bytes>, settings: Settings) -> Result<Self, StartError> {
        let dictionary = Checked::new(dictionary).map_err(StartError::Dictionary)?;
        let generation = Arc::new(AtomicU64::new(0));
        let (jobs, thread) = match model {
            None => (None, None),
            Some(model) => {
                // Read here, where what is wrong can be said; read again on the
                // worker, which keeps it.
                let file = ModelFile::parse(model.as_ref()).map_err(|error| StartError::Model(error.to_string()))?;
                Zenz::from_file(&file).map_err(|error| StartError::Model(error.to_string()))?;
                let (jobs, receiver) = mpsc::channel();
                let worker_dictionary = dictionary.clone();
                let worker_settings = settings.clone();
                let worker_generation = Arc::clone(&generation);
                let thread = std::thread::Builder::new()
                    .name("kiral-worker".to_owned())
                    .spawn(move || {
                        let dictionary = worker_dictionary.dictionary();
                        let Ok(file) = ModelFile::parse(model.as_ref()) else {
                            return;
                        };
                        let Ok(zenz) = Zenz::from_file(&file) else {
                            return;
                        };
                        work(&zenz, &dictionary, &worker_settings, &worker_generation, &receiver);
                    })
                    .map_err(StartError::Thread)?;
                (Some(jobs), Some(thread))
            }
        };
        Ok(Self {
            shared: Arc::new(Shared {
                dictionary,
                settings,
                generation,
                jobs,
            }),
            thread,
        })
    }

    /// Whether a model refines the drafts.
    #[must_use]
    pub fn has_model(&self) -> bool {
        self.shared.jobs.is_some()
    }
}

impl Drop for Worker {
    /// Stops the worker -- a job under way at its next block -- and waits for
    /// it to leave; sessions left show the drafts alone.
    fn drop(&mut self) {
        self.shared.generation.fetch_add(1, Ordering::AcqRel);
        if let Some(jobs) = &self.shared.jobs {
            let _ = jobs.send(Message::Stop);
        }
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

/// A key, as a frontend gives it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Key {
    /// A key that types a character: the character the keyboard's layout and
    /// Shift make of it.
    Character(char),
    /// Space.
    Space,
    /// Enter.
    Enter,
    /// Backspace.
    Backspace,
    /// Delete.
    Delete,
    /// Escape.
    Escape,
    /// Tab.
    Tab,
    /// The arrow to the left.
    Left,
    /// The arrow to the right.
    Right,
    /// The arrow up.
    Up,
    /// The arrow down.
    Down,
    /// Home.
    Home,
    /// End.
    End,
    /// Page Up.
    PageUp,
    /// Page Down.
    PageDown,
    /// A function key, F1 to F24.
    Function(u8),
}

/// The modifiers held with a key: a set of [`Modifiers::SHIFT`],
/// [`Modifiers::CONTROL`], [`Modifiers::ALT`] and [`Modifiers::SUPER`], each a
/// bit, as the C ABI gives them.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Modifiers(pub u32);

impl Modifiers {
    /// None held.
    pub const NONE: Self = Self(0);
    /// Shift.
    pub const SHIFT: Self = Self(1);
    /// Control.
    pub const CONTROL: Self = Self(1 << 1);
    /// Alt.
    pub const ALT: Self = Self(1 << 2);
    /// Super, the logo key.
    pub const SUPER: Self = Self(1 << 3);

    /// Whether all of the modifiers given are held.
    #[must_use]
    pub fn contains(self, modifiers: Self) -> bool {
        self.0 & modifiers.0 == modifiers.0
    }

    /// Whether a key held with these is the application's shortcut: Control,
    /// Alt or Super held.
    fn shortcut(self) -> bool {
        self.0 & (Self::CONTROL | Self::ALT | Self::SUPER).0 != 0
    }
}

impl std::ops::BitOr for Modifiers {
    type Output = Self;

    fn bitor(self, modifiers: Self) -> Self {
        Self(self.0 | modifiers.0)
    }
}

/// What a session shows where the text is typed.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Preedit {
    /// The text.
    pub text: String,
    /// Where the cursor is, in bytes of the text.
    pub cursor: usize,
    /// The part converted and selected, in bytes of the text, shown
    /// highlighted; none while the text is typed.
    pub highlight: Option<Range<usize>>,
}

/// The candidates of a conversion.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Candidates {
    /// The candidates, the conversion first, each once.
    pub texts: Vec<String>,
    /// The candidate shown.
    pub selected: usize,
    /// Whether the list is shown: from the second Space, Down or Up on.
    pub shown: bool,
}

/// A character a key types into a composition: anything but a control
/// character or a space.
fn types(character: char) -> bool {
    !character.is_control() && !character.is_whitespace()
}

/// The last characters of a text that the model reads as its left context.
fn keep_context(context: &mut String) {
    let count = context.chars().count();
    if let Some((at, _)) = context.char_indices().nth(count.saturating_sub(zenz::CONTEXT_LENGTH)) {
        context.drain(..at);
    }
}

/// A frontend's input context: what is typed into it, what it shows, and what
/// it commits.
#[derive(Debug)]
pub struct Session {
    shared: Arc<Shared>,
    drafts: Drafts,
    composer: Composer,
    candidates: Option<Candidates>,
    preedit: Preedit,
    commit: String,
    context: String,
}

impl Session {
    /// A session of an engine, nothing typed.
    #[must_use]
    pub fn new(worker: &Worker) -> Self {
        let shared = Arc::clone(&worker.shared);
        let drafts = Drafts::new(
            shared.settings.clone(),
            Arc::clone(&shared.generation),
            shared.jobs.clone(),
        );
        Self {
            shared,
            drafts,
            composer: Composer::new(),
            candidates: None,
            preedit: Preedit::default(),
            commit: String::new(),
            context: String::new(),
        }
    }

    /// The wake-up the worker calls, on its own thread, once it has given back
    /// a refinement of this session's: the frontend's, to have its own thread
    /// call [`Session::poll`].
    pub fn set_notify(&mut self, notify: Option<Notify>) {
        self.drafts.set_notify(notify);
    }

    /// The text before the cursor, where the frontend knows it: the left
    /// context of the next conversion.
    pub fn set_context(&mut self, context: &str) {
        self.context.clear();
        self.context.push_str(context);
        keep_context(&mut self.context);
    }

    /// Takes a key: whether the session took it, and the frontend is not to
    /// pass it on. What it shows after it is [`Session::preedit`],
    /// [`Session::commit`] and [`Session::candidates`].
    pub fn key(&mut self, key: Key, modifiers: Modifiers) -> bool {
        self.commit.clear();
        if modifiers.shortcut() {
            return false;
        }
        let taken = if self.candidates.is_some() {
            self.key_converting(key);
            true
        } else if self.composer.is_empty() {
            match key {
                Key::Character(character) if types(character) => {
                    self.type_character(character);
                    true
                }
                _ => false,
            }
        } else {
            self.key_composing(key);
            true
        };
        self.show();
        taken
    }

    /// Takes the refinements that have come, without waiting: whether what the
    /// session shows changed. The frontend's thread calls it once the wake-up
    /// has been called.
    pub fn poll(&mut self) -> bool {
        self.drafts.poll() && self.refined()
    }

    /// Waits for the refinement of what is typed up to an instant: whether it
    /// has come. For tools and tests; a frontend waits on its wake-up.
    pub fn wait_until(&mut self, until: Instant) -> bool {
        let before = self.drafts.answer().is_some();
        let came = self.drafts.wait_until(until);
        if came && !before {
            self.refined();
        }
        came
    }

    /// Deletes what is typed and the left context: the input context lost its
    /// focus, or its text changed under it.
    pub fn reset(&mut self) {
        self.cancel();
        self.context.clear();
        self.commit.clear();
        self.show();
    }

    /// What the session shows where the text is typed.
    #[must_use]
    pub fn preedit(&self) -> &Preedit {
        &self.preedit
    }

    /// The text the last key committed: empty where it committed none.
    #[must_use]
    pub fn commit(&self) -> &str {
        &self.commit
    }

    /// The candidates of the conversion, while there is one.
    #[must_use]
    pub fn candidates(&self) -> Option<&Candidates> {
        self.candidates.as_ref()
    }

    /// The refinement of the latest job, once it has come.
    #[must_use]
    pub fn refinement(&self) -> Option<&Outcome> {
        self.drafts.answer()
    }

    /// The generation of the latest job: raised by each key that changes the
    /// reading, and by each conversion.
    #[must_use]
    pub fn generation(&self) -> u64 {
        self.drafts.posted()
    }

    fn key_composing(&mut self, key: Key) {
        match key {
            Key::Character(character) if types(character) => self.type_character(character),
            Key::Space if self.composer.is_typed_as_is() => self.type_character(' '),
            Key::Space => self.convert(),
            Key::Enter => {
                self.flush();
                let text = self.drafts.text().to_owned();
                self.finish(text);
            }
            Key::Backspace => {
                self.composer.backspace();
                if self.composer.is_empty() {
                    self.cancel();
                } else {
                    self.composed();
                }
            }
            Key::Escape => self.cancel(),
            Key::Function(6) => self.take_kana(false),
            Key::Function(7) => self.take_kana(true),
            // The cursor's keys wait for segments.
            _ => {}
        }
    }

    fn key_converting(&mut self, key: Key) {
        let Some((count, shown)) = self
            .candidates
            .as_ref()
            .map(|candidates| (candidates.texts.len(), candidates.shown))
        else {
            return;
        };
        match key {
            Key::Space | Key::Down | Key::Tab => self.step(1),
            Key::Up => self.step(count.saturating_sub(1)),
            Key::Enter => self.commit_selected(),
            Key::Escape | Key::Backspace => self.candidates = None,
            Key::Function(6) => self.take_kana(false),
            Key::Function(7) => self.take_kana(true),
            Key::Character(character) if shown && ('1'..='9').contains(&character) => {
                let chosen = character
                    .to_digit(10)
                    .and_then(|digit| usize::try_from(digit).ok())
                    .map_or(0, |digit| digit - 1);
                if chosen < count {
                    if let Some(candidates) = &mut self.candidates {
                        candidates.selected = chosen;
                    }
                    self.commit_selected();
                }
            }
            Key::Character(character) if types(character) => {
                self.commit_selected();
                self.type_character(character);
            }
            _ => {}
        }
    }

    /// Shows the list of candidates, and moves the selection by a step, round
    /// from the last to the first.
    fn step(&mut self, by: usize) {
        if let Some(candidates) = &mut self.candidates {
            candidates.shown = true;
            candidates.selected = (candidates.selected + by) % candidates.texts.len().max(1);
        }
    }

    /// Types a character into the composition, and drafts the reading where it
    /// changed.
    fn type_character(&mut self, character: char) {
        self.composer.push(character);
        self.composed();
    }

    /// Drafts the composition's reading where it changed, and posts it to the
    /// worker.
    fn composed(&mut self) {
        let reading = self.composer.reading();
        if reading == self.drafts.reading() {
            return;
        }
        if reading.is_empty() {
            self.drafts.clear();
        } else {
            let dictionary = self.shared.dictionary.dictionary();
            self.drafts.compose(&dictionary, &reading, &self.context);
        }
    }

    /// Takes the keys that wait, and drafts the reading they make where it
    /// changed, posting nothing.
    fn flush(&mut self) {
        self.composer.flush();
        let reading = self.composer.reading();
        if reading != self.drafts.reading() {
            let dictionary = self.shared.dictionary.dictionary();
            self.drafts.redraft(&dictionary, &reading, &self.context);
        }
    }

    /// Converts what is typed: waits for its refinement up to the deadline, and
    /// shows the conversion, its candidates behind it.
    fn convert(&mut self) {
        self.flush();
        let conversion = self.drafts.convert().to_owned();
        self.candidates = Some(self.candidates_of(conversion));
    }

    /// Shows the kana, or the katakana, of what is typed, among its candidates.
    fn take_kana(&mut self, katakana_wanted: bool) {
        if self.candidates.is_none() {
            self.flush();
            let conversion = self.drafts.text().to_owned();
            self.candidates = Some(self.candidates_of(conversion));
        }
        let reading = self.drafts.reading();
        let wanted = if katakana_wanted {
            katakana(reading)
        } else {
            reading.to_owned()
        };
        if let Some(candidates) = &mut self.candidates
            && let Some(at) = candidates.texts.iter().position(|text| *text == wanted)
        {
            candidates.selected = at;
        }
    }

    /// The candidates of the reading converted: the conversion, the
    /// dictionary's own, the kana and the katakana, each once.
    fn candidates_of(&self, conversion: String) -> Candidates {
        let dictionary = self.shared.dictionary.dictionary();
        let reading = self.drafts.reading();
        let lattice = Lattice::new(&dictionary, reading);
        let own = lattice.text(&lattice.best(&dictionary));
        let mut texts = vec![conversion];
        for text in [own, reading.to_owned(), katakana(reading)] {
            if !texts.contains(&text) {
                texts.push(text);
            }
        }
        Candidates {
            texts,
            selected: 0,
            shown: false,
        }
    }

    /// Shows a refinement that has come: in place of the draft while the text
    /// is typed, and of a conversion shown alone, its list not shown. Whether
    /// what is shown changed.
    fn refined(&mut self) -> bool {
        if let Some(candidates) = &mut self.candidates {
            if candidates.selected != 0 || candidates.shown {
                return false;
            }
            let first = self.drafts.text().to_owned();
            let rest: Vec<String> = candidates.texts.drain(1..).filter(|text| *text != first).collect();
            candidates.texts = vec![first];
            candidates.texts.extend(rest);
        }
        self.show();
        true
    }

    fn commit_selected(&mut self) {
        if let Some(candidates) = self.candidates.take() {
            let text = candidates
                .texts
                .into_iter()
                .nth(candidates.selected)
                .unwrap_or_default();
            self.finish(text);
        }
    }

    /// Commits a text: the composition ends, and the text is the left context
    /// of the next.
    fn finish(&mut self, text: String) {
        self.context.push_str(&text);
        keep_context(&mut self.context);
        self.commit = text;
        self.composer.clear();
        self.candidates = None;
        self.drafts.clear();
    }

    /// Deletes what is typed.
    fn cancel(&mut self) {
        self.composer.clear();
        self.candidates = None;
        self.drafts.clear();
    }

    /// Makes what is shown of what there is.
    fn show(&mut self) {
        let text = match &self.candidates {
            Some(candidates) => candidates.texts.get(candidates.selected).cloned().unwrap_or_default(),
            None if self.composer.is_empty() => String::new(),
            None => {
                let mut text = self.drafts.text().to_owned();
                text.push_str(&self.composer.pending_shown());
                text
            }
        };
        let highlight = self.candidates.as_ref().map(|_| 0..text.len());
        self.preedit = Preedit {
            cursor: text.len(),
            highlight,
            text,
        };
    }
}
