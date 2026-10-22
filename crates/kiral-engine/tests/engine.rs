// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later
//
// This file is part of KIRAL. The SPDX lines name its copyright holder and its
// license, in the machine-readable form of the REUSE specification: the GNU
// General Public License, version 3 or any later version
// (LICENSES/GPL-3.0-or-later.txt).

//! Scripted sessions of the engine, with a small zenz of random weights: a key
//! shows its draft at once, a path of its lattice; the refinement that comes is
//! of the latest reading, never of one the keys have passed; an explicit
//! conversion shows a path of its lattice, refined or not by its deadline; and
//! the worker leaves when the session ends.

mod common;

use std::time::{Duration, Instant};

use common::{KANA, Random, dictionary, texts, zenz};
use kiral_dict::Dictionary;
use kiral_engine::engine::{Engine, Settings};
use kiral_lattice::Lattice;

/// Whether a text is the text of a path of a reading's lattice.
fn is_path(dictionary: &Dictionary<'_>, reading: &str, text: &str) -> bool {
    texts(&Lattice::new(dictionary, reading))
        .iter()
        .any(|path| path == text)
}

#[test]
fn keys_show_drafts_at_once_and_take_refinements_of_the_latest_reading() {
    let mut random = Random(61);
    let zenz = zenz(&mut random);
    let bytes = dictionary(&mut random);
    let dictionary = Dictionary::parse(&bytes).expect("a dictionary");
    let settings = Settings {
        capacity: 128,
        deadline: Duration::from_secs(5),
        ..Settings::default()
    };
    Engine::run(&zenz, &dictionary, &settings, |engine| {
        for item in 0..6 {
            let reading: String = (0..=random.below(5)).map(|_| KANA[random.below(KANA.len())]).collect();
            let context = if item % 2 == 0 { "" } else { "左の文" };
            let kana: Vec<char> = reading.chars().collect();
            for length in 1..=kana.len() {
                let prefix: String = kana[..length].iter().collect();
                let shown = engine.compose(&prefix, context).to_owned();
                assert!(is_path(&dictionary, &prefix, &shown), "{prefix}: the draft {shown}");
                assert!(engine.answer().is_none(), "no refinement of a reading just composed");
                assert!(
                    engine.wait_until(Instant::now() + Duration::from_secs(5)),
                    "{prefix}: refined"
                );
                let answer = engine.answer().expect("an answer");
                assert_eq!(answer.generation, engine.generation());
                let refined = answer.refined.as_ref().expect("refined");
                assert_eq!(refined.text, engine.text());
                assert!(refined.rounds <= settings.key_rounds);
                assert!(
                    is_path(&dictionary, &prefix, &refined.text),
                    "{prefix}: {}",
                    refined.text
                );
            }
            let converted = engine.convert().to_owned();
            assert!(
                is_path(&dictionary, &reading, &converted),
                "{reading}: converted {converted}"
            );
            let answer = engine.answer().expect("a conversion refined by its deadline of 5 s");
            assert_eq!(answer.generation, engine.generation());
            assert!(answer.refined.as_ref().expect("refined").rounds <= settings.conversion_rounds);
        }
    });
}

#[test]
fn refinements_of_a_reading_the_keys_have_passed_are_dropped() {
    let mut random = Random(62);
    let zenz = zenz(&mut random);
    let bytes = dictionary(&mut random);
    let dictionary = Dictionary::parse(&bytes).expect("a dictionary");
    let settings = Settings {
        capacity: 128,
        ..Settings::default()
    };
    Engine::run(&zenz, &dictionary, &settings, |engine| {
        for _ in 0..10 {
            // Keys as fast as they come: the worker's jobs of the first readings
            // are passed before, or after, it gives them back.
            let readings = ["かな", "かなん", "かなんこ"];
            for reading in readings {
                engine.compose(reading, "");
            }
            assert!(engine.wait_until(Instant::now() + Duration::from_secs(5)), "refined");
            let answer = engine.answer().expect("an answer");
            assert_eq!(answer.generation, engine.generation());
            let refined = answer.refined.as_ref().expect("refined");
            assert!(is_path(&dictionary, "かなんこ", &refined.text), "{}", refined.text);
            // Nothing that comes later is of another reading.
            std::thread::sleep(Duration::from_millis(5));
            engine.poll();
            assert_eq!(
                engine.answer().map(|answer| answer.generation),
                Some(engine.generation())
            );
            assert!(is_path(&dictionary, "かなんこ", engine.text()));
        }
    });
}
