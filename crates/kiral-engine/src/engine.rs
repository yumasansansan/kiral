// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later
//
// This file is part of KIRAL. The SPDX lines name its copyright holder and its
// license, in the machine-readable form of the REUSE specification: the GNU
// General Public License, version 3 or any later version
// (LICENSES/GPL-3.0-or-later.txt).

//! The engine: a reading drafted on the input thread, and refined on a worker.
//!
//! Each change of the reading -- a key -- makes a new generation, a counter
//! that only the input threads raise. The input thread drafts the reading from
//! its lattice, within the constraint the last refinement shown left
//! ([`refine::draft`]), shows the draft, and posts a job of the generation to
//! the worker. It never waits for the worker, but on an explicit conversion,
//! which waits up to a deadline and then shows what there is. The worker sleeps
//! on the jobs' channel, takes the latest job, and refines its reading by a
//! round of verification -- by several for an explicit conversion -- asking
//! before each block of the model whether the generation is still the latest,
//! and dropping the job where it is not; it gives back what it found to the
//! channel the job names, and calls the job's wake-up where it has one, and
//! what it found is shown only while its generation is the latest.
//!
//! The channels are std's, whose sending never blocks: the input thread takes
//! the worker's results without waiting (`try_recv`), and waits on them only
//! for an explicit conversion (`recv_timeout`). The worker holds the model and
//! its cache of keys and values, so that a key's job feeds the model only the
//! end of the prompt that changed -- the reading's new kana and what follows
//! them -- and the draft. [`Engine::run`] scopes a worker to a body that types
//! into it; [`crate::session`] gives each frontend's input context a session of
//! an engine that owns its files and its worker.

use std::fmt;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{self, Receiver, Sender};
use std::time::{Duration, Instant};

use kiral_dict::Dictionary;
use kiral_lattice::{Constraint, Lattice};
use kiral_nn::Error;
use kiral_nn::zenz::{self, Zenz};

use crate::refine::{self, Refined, ZenzJudge};

/// The settings of an engine.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Settings {
    /// The positions the model's cache has room for: the longest prompt and
    /// draft it refines, the left context left out where they would not fit
    /// with it.
    pub capacity: usize,
    /// The rounds of verification of a key's job.
    pub key_rounds: usize,
    /// The rounds of an explicit conversion's job.
    pub conversion_rounds: usize,
    /// How long an explicit conversion waits for its refinement.
    pub deadline: Duration,
    /// The processor the worker runs on, pinned; none to leave it to the
    /// system.
    pub worker_processor: Option<usize>,
}

impl Default for Settings {
    /// Room for 256 positions, 19 MB of keys and values for zenz-v3.1-small;
    /// a round for a key and up to eight for a conversion, which waits 30 ms.
    fn default() -> Self {
        Self {
            capacity: 256,
            key_rounds: 1,
            conversion_rounds: 8,
            deadline: Duration::from_millis(30),
            worker_processor: None,
        }
    }
}

/// A function the worker calls on its own thread once it has given back an
/// outcome: a frontend's wake-up, which has its own thread take the outcome.
#[derive(Clone)]
pub struct Notify(pub Arc<dyn Fn() + Send + Sync>);

impl fmt::Debug for Notify {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("Notify")
    }
}

/// What the input threads ask of the worker.
#[derive(Debug)]
pub(crate) enum Message {
    /// A reading to refine.
    Job(Job),
    /// Leave: the engine is closed.
    Stop,
}

/// A job for the worker: a generation's reading, the text to its left, the
/// constraint its draft is found within, and where to give back what it finds.
#[derive(Debug)]
pub(crate) struct Job {
    generation: u64,
    reading: String,
    context: String,
    carried: Constraint,
    rounds: usize,
    reply: Sender<Outcome>,
    notify: Option<Notify>,
}

/// What the worker gives back for a job.
#[derive(Debug, Clone)]
pub struct Outcome {
    /// The generation it is of.
    pub generation: u64,
    /// The refined conversion, or why there is none.
    pub refined: Result<Refined, String>,
    /// The time the worker took over the job, from taking it to giving it back.
    pub took: Duration,
}

/// The input thread's side of a reading's conversion: what it shows, and the
/// channels to and from the worker, where there is one -- an engine's, and
/// each session's.
#[derive(Debug)]
pub(crate) struct Drafts {
    settings: Settings,
    generation: Arc<AtomicU64>,
    /// The worker's jobs; none where there is no worker, and the drafts are all
    /// there is.
    jobs: Option<Sender<Message>>,
    reply: Sender<Outcome>,
    outcomes: Receiver<Outcome>,
    notify: Option<Notify>,
    /// The generation of the latest job.
    posted: u64,
    reading: String,
    context: String,
    /// The constraint of the refinement shown, which the next draft is found
    /// within.
    carried: Constraint,
    /// The text shown.
    text: String,
    /// The latest job's outcome, once it has come.
    answer: Option<Outcome>,
}

impl Drafts {
    pub(crate) fn new(settings: Settings, generation: Arc<AtomicU64>, jobs: Option<Sender<Message>>) -> Self {
        let (reply, outcomes) = mpsc::channel();
        Self {
            settings,
            generation,
            jobs,
            reply,
            outcomes,
            notify: None,
            posted: 0,
            reading: String::new(),
            context: String::new(),
            carried: Constraint::default(),
            text: String::new(),
            answer: None,
        }
    }

    /// A new generation, the latest job's: whatever the worker is doing stops
    /// at its next block.
    fn raise(&mut self) -> u64 {
        self.posted = self.generation.fetch_add(1, Ordering::AcqRel) + 1;
        self.posted
    }

    /// Drafts a reading, the text to its left given, within the constraint the
    /// last refinement left, and shows the draft.
    fn draft(&mut self, dictionary: &Dictionary<'_>, reading: &str, context: &str) {
        let lattice = Lattice::new(dictionary, reading);
        let (path, carried) = refine::draft(&lattice, dictionary, &self.carried);
        self.text = lattice.text(&path);
        self.carried = carried;
        self.reading.clear();
        self.reading.push_str(reading);
        self.context.clear();
        self.context.push_str(context);
        self.answer = None;
    }

    /// Composes a reading: drafts it, shows the draft, and posts the reading
    /// to the worker for a key's rounds. The text shown.
    pub(crate) fn compose(&mut self, dictionary: &Dictionary<'_>, reading: &str, context: &str) -> &str {
        let generation = self.raise();
        self.draft(dictionary, reading, context);
        self.post(generation, self.settings.key_rounds);
        &self.text
    }

    /// Drafts a reading and shows the draft, posting nothing: what is shown
    /// before an explicit conversion, which posts it.
    pub(crate) fn redraft(&mut self, dictionary: &Dictionary<'_>, reading: &str, context: &str) {
        self.raise();
        self.draft(dictionary, reading, context);
    }

    /// Converts the reading explicitly: posts it again, for a refinement of the
    /// conversion's rounds, waits for it up to the deadline, and shows what
    /// there is then -- the draft at once where there is no worker. The text
    /// shown.
    pub(crate) fn convert(&mut self) -> &str {
        let generation = self.raise();
        self.answer = None;
        if self.post(generation, self.settings.conversion_rounds) {
            self.wait_until(Instant::now() + self.settings.deadline);
        }
        &self.text
    }

    /// Forgets the reading, a composition ended: nothing carried to the next,
    /// and the jobs posted dropped.
    pub(crate) fn clear(&mut self) {
        self.raise();
        self.reading.clear();
        self.context.clear();
        self.carried = Constraint::default();
        self.text.clear();
        self.answer = None;
    }

    /// Takes the worker's outcomes that have come, without waiting, and shows
    /// the latest job's. Whether it came now.
    pub(crate) fn poll(&mut self) -> bool {
        let mut came = false;
        while let Ok(outcome) = self.outcomes.try_recv() {
            came |= self.take(outcome);
        }
        came
    }

    /// Waits for the latest job's outcome up to an instant, and shows it.
    /// Whether it has come.
    pub(crate) fn wait_until(&mut self, until: Instant) -> bool {
        self.poll();
        while self.answer.is_none() {
            let Some(left) = until.checked_duration_since(Instant::now()) else {
                break;
            };
            match self.outcomes.recv_timeout(left) {
                Ok(outcome) => {
                    self.take(outcome);
                }
                Err(_) => break,
            }
        }
        self.answer.is_some()
    }

    /// Posts the reading to the worker: whether a worker took it. One that has
    /// gone, or none, takes no jobs, and the drafts are shown alone.
    fn post(&self, generation: u64, rounds: usize) -> bool {
        self.jobs.as_ref().is_some_and(|jobs| {
            jobs.send(Message::Job(Job {
                generation,
                reading: self.reading.clone(),
                context: self.context.clone(),
                carried: self.carried.clone(),
                rounds,
                reply: self.reply.clone(),
                notify: self.notify.clone(),
            }))
            .is_ok()
        })
    }

    /// Shows an outcome of the latest job, and drops any other. Whether it
    /// was shown.
    fn take(&mut self, outcome: Outcome) -> bool {
        if outcome.generation != self.posted {
            return false;
        }
        if let Ok(refined) = &outcome.refined {
            self.text.clone_from(&refined.text);
            self.carried.clone_from(&refined.constraint);
        }
        self.answer = Some(outcome);
        true
    }

    /// The wake-up the worker calls once it has given back an outcome.
    pub(crate) fn set_notify(&mut self, notify: Option<Notify>) {
        self.notify = notify;
    }

    /// The reading drafted.
    pub(crate) fn reading(&self) -> &str {
        &self.reading
    }

    /// The text shown.
    pub(crate) fn text(&self) -> &str {
        &self.text
    }

    /// The latest job's outcome, once it has come.
    pub(crate) fn answer(&self) -> Option<&Outcome> {
        self.answer.as_ref()
    }

    /// The generation of the latest job.
    pub(crate) fn posted(&self) -> u64 {
        self.posted
    }
}

/// The input thread's side of an engine whose worker [`Engine::run`] scopes.
#[derive(Debug)]
pub struct Engine<'e> {
    dictionary: &'e Dictionary<'e>,
    drafts: Drafts,
}

impl Engine<'_> {
    /// Runs `body` with an engine whose worker, started first and stopped when
    /// `body` returns, refines with `zenz`.
    pub fn run<R>(
        zenz: &Zenz<'_>,
        dictionary: &Dictionary<'_>,
        settings: &Settings,
        body: impl FnOnce(&mut Engine<'_>) -> R,
    ) -> R {
        let generation = Arc::new(AtomicU64::new(0));
        let (jobs, job_receiver) = mpsc::channel();
        std::thread::scope(|scope| {
            let worker_generation = Arc::clone(&generation);
            scope.spawn(move || work(zenz, dictionary, settings, &worker_generation, &job_receiver));
            let mut engine = Engine {
                dictionary,
                drafts: Drafts::new(settings.clone(), Arc::clone(&generation), Some(jobs)),
            };
            let result = body(&mut engine);
            // A job under way stops at its next block, and the worker leaves when
            // the channel of jobs closes with the engine.
            generation.fetch_add(1, Ordering::AcqRel);
            drop(engine);
            result
        })
    }

    /// Composes a reading, the text to its left given: drafts it within the
    /// constraint the last refinement left, shows the draft, and posts the
    /// reading to the worker. The text shown.
    pub fn compose(&mut self, reading: &str, context: &str) -> &str {
        self.drafts.compose(self.dictionary, reading, context)
    }

    /// Takes the worker's outcomes that have come, without waiting, and shows
    /// the latest generation's. Whether it has come.
    pub fn poll(&mut self) -> bool {
        self.drafts.poll();
        self.drafts.answer().is_some()
    }

    /// Waits for the latest generation's outcome up to an instant, and shows
    /// it. Whether it came.
    pub fn wait_until(&mut self, until: Instant) -> bool {
        self.drafts.wait_until(until)
    }

    /// Converts the reading explicitly: posts it again, for a refinement of the
    /// conversion's rounds, waits for it up to the deadline, and shows what
    /// there is then. The text shown.
    pub fn convert(&mut self) -> &str {
        self.drafts.convert()
    }

    /// The text shown.
    #[must_use]
    pub fn text(&self) -> &str {
        self.drafts.text()
    }

    /// The latest generation's outcome, once it has come.
    #[must_use]
    pub fn answer(&self) -> Option<&Outcome> {
        self.drafts.answer()
    }

    /// The latest generation.
    #[must_use]
    pub fn generation(&self) -> u64 {
        self.drafts.posted()
    }
}

/// The worker: takes the latest job, refines it unless its generation has
/// passed, and gives it back to the channel the job names, until it is told to
/// stop or the channel of jobs closes.
pub(crate) fn work(
    zenz: &Zenz<'_>,
    dictionary: &Dictionary<'_>,
    settings: &Settings,
    generation: &AtomicU64,
    jobs: &Receiver<Message>,
) {
    if let Some(processor) = settings.worker_processor {
        // Where the system refuses, the worker runs unpinned, only not where it
        // was asked to.
        let _ = kiral_sys::affinity::pin_current_thread(processor);
    }
    let mut judge = ZenzJudge::new(zenz, settings.capacity.min(zenz.model.config().positions));
    while let Ok(Message::Job(mut job)) = jobs.recv() {
        loop {
            match jobs.try_recv() {
                Ok(Message::Job(newer)) => job = newer,
                Ok(Message::Stop) => return,
                Err(_) => break,
            }
        }
        if job.generation != generation.load(Ordering::Acquire) {
            continue;
        }
        let started = Instant::now();
        let mut stop = || generation.load(Ordering::Acquire) != job.generation;
        let refined = match refine_reading(
            zenz,
            dictionary,
            &mut judge,
            &job.reading,
            &job.context,
            &job.carried,
            job.rounds,
            &mut stop,
        ) {
            Ok(None) => continue,
            Ok(Some(refined)) => Ok(refined),
            Err(error) => Err(error.to_string()),
        };
        let outcome = Outcome {
            generation: job.generation,
            refined,
            took: started.elapsed(),
        };
        // A session that has gone takes nothing, and the others are served on.
        if job.reply.send(outcome).is_ok()
            && let Some(notify) = &job.notify
        {
            (notify.0)();
        }
    }
}

/// The worker's refinement of a reading, the text to its left given, from the
/// constraint its draft is found within, by up to `rounds` rounds: its lattice,
/// its prompt -- without the left context where the prompt and the first
/// draft would not fit the room of the judge's cache with it -- and its rounds
/// ([`refine::refine`]); none where `stop` said to stop.
///
/// # Errors
///
/// Where the model cannot compute the tokens: past its room, say.
#[expect(
    clippy::too_many_arguments,
    reason = "the parts a refinement takes, each of its own kind"
)]
pub fn refine_reading(
    zenz: &Zenz<'_>,
    dictionary: &Dictionary<'_>,
    judge: &mut ZenzJudge<'_>,
    reading: &str,
    context: &str,
    carried: &Constraint,
    rounds: usize,
    stop: &mut dyn FnMut() -> bool,
) -> Result<Option<Refined>, Error> {
    let tokenizer = &zenz.tokenizer;
    let lattice = Lattice::new(dictionary, reading);
    let (path, _) = refine::draft(&lattice, dictionary, carried);
    let mut draft = Vec::new();
    tokenizer.encode(&zenz::preprocess(&lattice.text(&path)), &mut draft);
    let mut prompt = refine::prompt_tokens(tokenizer, reading, context);
    if prompt.len() + draft.len() + 1 > judge.capacity() && !context.is_empty() {
        prompt = refine::prompt_tokens(tokenizer, reading, "");
    }
    refine::refine(dictionary, &lattice, tokenizer, judge, &prompt, carried, rounds, stop)
}

#[cfg(test)]
mod tests {
    use kiral_dict::file::{Unknown, build};
    use kiral_dict::mozc::Matrix;
    use kiral_dict::{Metadata, Word};

    use super::*;
    use crate::refine::Ending;

    #[test]
    fn outcomes_of_a_generation_the_keys_have_passed_are_dropped() {
        let words = [Word {
            reading: "か".to_owned(),
            surface: "甲".to_owned(),
            lid: 1,
            rid: 1,
            cost: 100,
            flags: 0,
        }];
        let matrix = Matrix {
            ids: 2,
            costs: vec![0; 4],
        };
        let metadata = Metadata {
            name: "test",
            license: "CC0-1.0",
            source: "made by the test",
            notice: "",
        };
        let bytes = build(&words, &matrix, Unknown { id: 1, cost: 8000 }, &metadata).expect("a dictionary");
        let dictionary = Dictionary::parse(&bytes).expect("a dictionary");
        let (jobs, posted) = mpsc::channel();
        let mut drafts = Drafts::new(Settings::default(), Arc::new(AtomicU64::new(0)), Some(jobs));
        let job = || match posted.try_recv() {
            Ok(Message::Job(job)) => job,
            other => panic!("a job, not {other:?}"),
        };
        assert_eq!(drafts.compose(&dictionary, "か", ""), "甲");
        let first = job();
        assert_eq!(drafts.compose(&dictionary, "かか", ""), "甲甲");
        let latest = job();
        let outcome = |generation, text: &str| Outcome {
            generation,
            refined: Ok(Refined {
                text: text.to_owned(),
                constraint: Constraint {
                    prefix: text.as_bytes().to_vec(),
                    complete: false,
                },
                rounds: 1,
                ending: Ending::Accepted,
            }),
            took: Duration::ZERO,
        };
        first.reply.send(outcome(first.generation, "old")).expect("sent");
        assert!(!drafts.poll());
        assert_eq!(drafts.text(), "甲甲");
        latest.reply.send(outcome(latest.generation, "new")).expect("sent");
        latest.reply.send(outcome(first.generation, "old")).expect("sent");
        assert!(drafts.poll());
        assert_eq!(drafts.text(), "new");
        assert_eq!(drafts.carried.prefix, b"new");
        assert_eq!(drafts.answer().map(|answer| answer.generation), Some(latest.generation));
        // A constraint that no path of the next reading meets is not carried to
        // its job.
        drafts.compose(&dictionary, "かかか", "");
        assert_eq!(job().carried.prefix, b"");
        // A composition cleared drops what was carried, and the jobs posted.
        drafts.clear();
        assert!(drafts.text().is_empty() && drafts.carried.prefix.is_empty());
        let passed = drafts.posted() - 1;
        latest.reply.send(outcome(passed, "late")).expect("sent");
        assert!(!drafts.poll());
        // Without a worker, a conversion shows the draft at once.
        let mut alone = Drafts::new(Settings::default(), Arc::new(AtomicU64::new(0)), None);
        alone.compose(&dictionary, "か", "");
        let started = Instant::now();
        assert_eq!(alone.convert(), "甲");
        assert!(started.elapsed() < Settings::default().deadline);
        assert!(alone.answer().is_none() && alone.jobs.is_none());
    }
}
