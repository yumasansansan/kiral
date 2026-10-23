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
//! that only the input thread raises. The input thread drafts the reading from
//! its lattice, within the constraint the last refinement shown left
//! ([`refine::draft`]), shows the draft, and posts a job of the generation to
//! the worker. It never waits for the worker, but on an explicit conversion,
//! which waits up to a deadline and then shows what there is. The worker sleeps
//! on the jobs' channel, takes the latest job, and refines its reading by a
//! round of verification -- by several for an explicit conversion -- asking
//! before each block of the model whether the generation is still the latest,
//! and dropping the job where it is not; it gives back what it found, which is
//! shown only while its generation is the latest.
//!
//! The channels are std's, whose sending never blocks: the input thread takes
//! the worker's results without waiting (`try_recv`), and waits on them only
//! for an explicit conversion (`recv_timeout`). The worker holds the model and
//! its cache of keys and values, so that a key's job feeds the model only the
//! end of the prompt that changed -- the reading's new kana and what follows
//! them -- and the draft.

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

/// A job for the worker: a generation's reading, the text to its left, and the
/// constraint its draft is found within.
#[derive(Debug, Clone)]
struct Job {
    generation: u64,
    reading: String,
    context: String,
    carried: Constraint,
    rounds: usize,
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

/// The input thread's side of an engine.
#[derive(Debug)]
pub struct Engine<'e> {
    dictionary: &'e Dictionary<'e>,
    settings: Settings,
    generation: &'e AtomicU64,
    jobs: Sender<Job>,
    outcomes: Receiver<Outcome>,
    reading: String,
    context: String,
    /// The constraint of the refinement shown, which the next draft is found
    /// within.
    carried: Constraint,
    /// The text shown.
    text: String,
    /// The latest generation's outcome, once it has come.
    answer: Option<Outcome>,
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
        let generation = AtomicU64::new(0);
        let (jobs, job_receiver) = mpsc::channel();
        let (outcome_sender, outcomes) = mpsc::channel();
        std::thread::scope(|scope| {
            let generation = &generation;
            scope.spawn(move || work(zenz, dictionary, settings, generation, &job_receiver, &outcome_sender));
            let mut engine = Engine {
                dictionary,
                settings: settings.clone(),
                generation,
                jobs,
                outcomes,
                reading: String::new(),
                context: String::new(),
                carried: Constraint::default(),
                text: String::new(),
                answer: None,
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
        let generation = self.generation.fetch_add(1, Ordering::AcqRel) + 1;
        let lattice = Lattice::new(self.dictionary, reading);
        let (path, carried) = refine::draft(&lattice, self.dictionary, &self.carried);
        self.text = lattice.text(&path);
        self.carried = carried;
        self.reading.clear();
        self.reading.push_str(reading);
        self.context.clear();
        self.context.push_str(context);
        self.answer = None;
        self.post(generation, self.settings.key_rounds);
        &self.text
    }

    /// Takes the worker's outcomes that have come, without waiting, and shows
    /// the latest generation's. Whether it has come.
    pub fn poll(&mut self) -> bool {
        while let Ok(outcome) = self.outcomes.try_recv() {
            self.take(outcome);
        }
        self.answer.is_some()
    }

    /// Waits for the latest generation's outcome up to an instant, and shows
    /// it. Whether it came.
    pub fn wait_until(&mut self, until: Instant) -> bool {
        while self.answer.is_none() {
            let Some(left) = until.checked_duration_since(Instant::now()) else {
                break;
            };
            match self.outcomes.recv_timeout(left) {
                Ok(outcome) => self.take(outcome),
                Err(_) => break,
            }
        }
        self.answer.is_some()
    }

    /// Converts the reading explicitly: posts it again, for a refinement of the
    /// conversion's rounds, waits for it up to the deadline, and shows what
    /// there is then. The text shown.
    pub fn convert(&mut self) -> &str {
        let generation = self.generation.fetch_add(1, Ordering::AcqRel) + 1;
        self.answer = None;
        self.post(generation, self.settings.conversion_rounds);
        self.wait_until(Instant::now() + self.settings.deadline);
        &self.text
    }

    /// The text shown.
    #[must_use]
    pub fn text(&self) -> &str {
        &self.text
    }

    /// The latest generation's outcome, once it has come.
    #[must_use]
    pub fn answer(&self) -> Option<&Outcome> {
        self.answer.as_ref()
    }

    /// The latest generation.
    #[must_use]
    pub fn generation(&self) -> u64 {
        self.generation.load(Ordering::Acquire)
    }

    /// Posts the reading to the worker; a worker that has gone takes no jobs,
    /// and the drafts are shown alone.
    fn post(&self, generation: u64, rounds: usize) {
        let _ = self.jobs.send(Job {
            generation,
            reading: self.reading.clone(),
            context: self.context.clone(),
            carried: self.carried.clone(),
            rounds,
        });
    }

    /// Shows an outcome of the latest generation; drops any other.
    fn take(&mut self, outcome: Outcome) {
        if outcome.generation != self.generation.load(Ordering::Acquire) {
            return;
        }
        if let Ok(refined) = &outcome.refined {
            self.text.clone_from(&refined.text);
            self.carried.clone_from(&refined.constraint);
        }
        self.answer = Some(outcome);
    }
}

/// The worker: takes the latest job, refines it unless its generation has
/// passed, and gives back what it found, until the channel of jobs closes.
fn work(
    zenz: &Zenz<'_>,
    dictionary: &Dictionary<'_>,
    settings: &Settings,
    generation: &AtomicU64,
    jobs: &Receiver<Job>,
    outcomes: &Sender<Outcome>,
) {
    if let Some(processor) = settings.worker_processor {
        // Where the system refuses, the worker runs unpinned, only not where it
        // was asked to.
        let _ = kiral_sys::affinity::pin_current_thread(processor);
    }
    let mut judge = ZenzJudge::new(zenz, settings.capacity.min(zenz.model.config().positions));
    while let Ok(mut job) = jobs.recv() {
        while let Ok(newer) = jobs.try_recv() {
            job = newer;
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
        if outcomes.send(outcome).is_err() {
            break;
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
        let generation = AtomicU64::new(0);
        let (jobs, posted) = mpsc::channel();
        let (given, outcomes) = mpsc::channel();
        let mut engine = Engine {
            dictionary: &dictionary,
            settings: Settings::default(),
            generation: &generation,
            jobs,
            outcomes,
            reading: String::new(),
            context: String::new(),
            carried: Constraint::default(),
            text: String::new(),
            answer: None,
        };
        assert_eq!(engine.compose("か", ""), "甲");
        let first = posted.try_recv().expect("a job").generation;
        assert_eq!(engine.compose("かか", ""), "甲甲");
        let latest = posted.try_recv().expect("a job").generation;
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
        given.send(outcome(first, "old")).expect("sent");
        assert!(!engine.poll());
        assert_eq!(engine.text(), "甲甲");
        given.send(outcome(latest, "new")).expect("sent");
        given.send(outcome(first, "old")).expect("sent");
        assert!(engine.poll());
        assert_eq!(engine.text(), "new");
        assert_eq!(engine.carried.prefix, b"new");
        assert_eq!(engine.answer().map(|answer| answer.generation), Some(latest));
        // A constraint that no path of the next reading meets is not carried to
        // its job.
        engine.compose("かかか", "");
        assert_eq!(posted.try_recv().expect("a job").carried.prefix, b"");
    }
}
