// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later
//
// This file is part of KIRAL. The SPDX lines name its copyright holder and its
// license, in the machine-readable form of the REUSE specification: the GNU
// General Public License, version 3 or any later version
// (LICENSES/GPL-3.0-or-later.txt).

//! Verification and refinement against an oracle, a judge that wants a text of
//! its own: the verdict is at the first token the draft has otherwise; the
//! rounds reach the oracle's text wherever it is a path of the lattice, and
//! end with a path, within what the oracle accepted, where it is not; a token
//! the draft spells otherwise ends them. And zenz as a judge, keeping what a
//! sequence shares with the one fed before, answers as a session fed afresh and
//! feeds only what changed.

mod common;

use common::{END, KANA, Random, dictionary, texts, tokenizer, zenz};
use kiral_dict::Dictionary;
use kiral_dict::file::{Unknown, build};
use kiral_dict::mozc::Matrix;
use kiral_dict::{Metadata, Word};
use kiral_engine::refine::{Ending, Judge, Verdict, ZenzJudge, katakana, prompt_tokens, refine, verify};
use kiral_kernels::vector::argmax;
use kiral_lattice::{Constraint, Lattice};
use kiral_nn::Error;
use kiral_nn::gpt2::Session;

/// A judge that wants a sequence of its own: after each position of a sequence
/// whose tokens so far are its own, the token its sequence has next, or the end
/// of text after its last; the end of text after any other.
struct Oracle {
    wanted: Vec<u32>,
    asked: usize,
}

impl Judge for Oracle {
    fn likeliest(
        &mut self,
        tokens: &[u32],
        from: usize,
        likeliest: &mut Vec<u32>,
        _stop: &mut dyn FnMut() -> bool,
    ) -> Result<bool, Error> {
        self.asked += 1;
        let shared = tokens
            .iter()
            .zip(&self.wanted)
            .take_while(|(given, wanted)| given == wanted)
            .count();
        likeliest.clear();
        likeliest.extend((from..tokens.len()).map(|position| {
            if position < shared {
                self.wanted.get(position + 1).copied().unwrap_or(END)
            } else {
                END
            }
        }));
        Ok(true)
    }
}

fn never() -> bool {
    false
}

#[test]
fn readings_are_given_in_katakana() {
    assert_eq!(katakana("かなゔゝー、カ"), "カナヴヽー、カ");
}

#[test]
fn a_verdict_is_at_the_first_token_the_draft_has_otherwise() {
    let tokenizer = tokenizer("");
    let prompt = prompt_tokens(&tokenizer, "かな", "");
    let oracle = |text: &str| {
        let mut wanted = prompt.clone();
        tokenizer.encode(text, &mut wanted);
        Oracle { wanted, asked: 0 }
    };
    let bytes = |text: &str| text.as_bytes().to_vec();
    let verdict = |text: &str, draft: &str| {
        verify(&mut oracle(text), &tokenizer, &prompt, draft, &mut never)
            .expect("a verdict")
            .expect("not stopped")
    };
    assert_eq!(verdict("甲乙", "甲乙"), Verdict::Accepted);
    // 丙 (E4 B8 99) and 乙 (E4 B9 99) part at their second byte.
    assert_eq!(
        verdict("甲乙", "甲丙"),
        Verdict::Fix(Constraint {
            prefix: [&bytes("甲")[..], &[0xe4, 0xb9]].concat(),
            complete: false,
        })
    );
    // The oracle ends where the draft goes on, and goes on where it ends.
    assert_eq!(
        verdict("甲乙", "甲乙丁"),
        Verdict::Fix(Constraint {
            prefix: bytes("甲乙"),
            complete: true,
        })
    );
    assert_eq!(
        verdict("甲乙", "甲"),
        Verdict::Fix(Constraint {
            prefix: [&bytes("甲")[..], &[0xe4]].concat(),
            complete: false,
        })
    );
}

#[test]
fn the_rounds_reach_the_oracles_text_where_it_is_a_path() {
    let mut random = Random(51);
    let tokenizer = tokenizer("");
    for round in 0..20 {
        let bytes = dictionary(&mut random);
        let dictionary = Dictionary::parse(&bytes).expect("a dictionary");
        for _ in 0..10 {
            let reading: String = (0..=random.below(6)).map(|_| KANA[random.below(KANA.len())]).collect();
            let lattice = Lattice::new(&dictionary, &reading);
            let all = texts(&lattice);
            let prompt = prompt_tokens(&tokenizer, &reading, "");
            let refine_toward = |text: &str| {
                let mut wanted = prompt.clone();
                tokenizer.encode(text, &mut wanted);
                let mut oracle = Oracle { wanted, asked: 0 };
                let refined = refine(
                    &dictionary,
                    &lattice,
                    &tokenizer,
                    &mut oracle,
                    &prompt,
                    &Constraint::default(),
                    100,
                    &mut never,
                )
                .expect("refined")
                .expect("not stopped");
                assert_eq!(refined.rounds, oracle.asked, "a feeding a round");
                assert!(refined.constraint.admits(&refined.text));
                assert!(all.contains(&refined.text), "the text of a path");
                refined
            };
            let target = &all[random.below(all.len())];
            let refined = refine_toward(target);
            let context = format!("round {round}, {reading}: {target}");
            assert_eq!(refined.ending, Ending::Accepted, "{context}");
            assert_eq!(&refined.text, target, "{context}");
            // 戊 is no word's: the rounds end where the lattice cannot follow.
            let cut = target
                .char_indices()
                .map(|(at, _)| at)
                .nth(random.below(target.chars().count()))
                .unwrap_or(0);
            let stranger = format!("{}戊{}", &target[..cut], &target[cut..]);
            let refined = refine_toward(&stranger);
            assert_eq!(refined.ending, Ending::NoPath, "{context}, {stranger}");
            assert!(
                refined.text.as_bytes().starts_with(&stranger.as_bytes()[..cut]),
                "{context}: {} keeps what the oracle accepted",
                refined.text
            );
        }
    }
}

#[test]
fn a_token_the_draft_spells_otherwise_ends_the_rounds() {
    // The tokenizer makes a word of 甲's bytes; the oracle wants them a byte at
    // a time, as a model whose likeliest token is a byte of a character its
    // vocabulary has a word for.
    let tokenizer = tokenizer("甲");
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
    let lattice = Lattice::new(&dictionary, "か");
    let prompt = prompt_tokens(&tokenizer, "か", "");
    let mut wanted = prompt.clone();
    wanted.extend("甲".bytes().map(|byte| u32::from(byte) + 2));
    let mut oracle = Oracle { wanted, asked: 0 };
    let refined = refine(
        &dictionary,
        &lattice,
        &tokenizer,
        &mut oracle,
        &prompt,
        &Constraint::default(),
        10,
        &mut never,
    )
    .expect("refined")
    .expect("not stopped");
    assert_eq!(refined.text, "甲");
    assert_eq!(refined.ending, Ending::NoProgress);
    assert_eq!(refined.rounds, 1);
}

/// The likeliest token after each position from `from` on, of a session fed
/// afresh, where the top two logits are apart by more than what the order of a
/// computation's sums moves them by; none where they are not.
fn fresh(zenz: &kiral_nn::zenz::Zenz<'_>, tokens: &[u32], from: usize) -> Vec<Option<u32>> {
    let vocabulary = zenz.model.config().vocabulary;
    let mut logits = vec![0.0f32; tokens.len() * vocabulary];
    Session::new(&zenz.model, common::POSITIONS)
        .feed_all(tokens, &mut logits)
        .expect("fed");
    logits
        .chunks_exact(vocabulary)
        .skip(from)
        .map(|row| {
            let mut sorted = row.to_vec();
            sorted.sort_by(|a, b| b.total_cmp(a));
            (sorted[0] - sorted[1] > 1e-4 * sorted[0].abs().max(1.0))
                .then(|| argmax(row).and_then(|token| u32::try_from(token).ok()))
                .flatten()
        })
        .collect()
}

#[test]
fn zenz_judges_as_a_session_fed_afresh_and_feeds_only_what_changed() {
    let mut random = Random(52);
    let zenz = zenz(&mut random);
    let tokenizer = &zenz.tokenizer;
    let mut judge = ZenzJudge::new(&zenz, 128);
    // The rounds of a reading, then a key that adds a kana to it.
    let sequence = |reading: &str, draft: &str| {
        let prompt = prompt_tokens(tokenizer, reading, "");
        let mut tokens = prompt.clone();
        tokenizer.encode(draft, &mut tokens);
        (tokens, prompt.len() - 1)
    };
    let sequences = [
        sequence("かなこ", "甲乙"),
        sequence("かなこ", "甲丙"),
        sequence("かなこ", "甲丙丁"),
        sequence("かなこん", "甲丙丁"),
        sequence("かなこん", "甲丙丁"),
    ];
    let mut previous: Vec<u32> = Vec::new();
    let mut likeliest = Vec::new();
    for (tokens, from) in &sequences {
        let before = judge.tokens_fed();
        assert!(
            judge
                .likeliest(tokens, *from, &mut likeliest, &mut never)
                .expect("judged")
        );
        let shared = previous.iter().zip(tokens).take_while(|(a, b)| a == b).count();
        assert_eq!(
            judge.tokens_fed() - before,
            tokens.len() - shared,
            "fed only what changed"
        );
        for (position, (judged, fresh)) in likeliest.iter().zip(fresh(&zenz, tokens, *from)).enumerate() {
            if let Some(fresh) = fresh {
                assert_eq!(*judged, fresh, "position {}", from + position);
            }
        }
        previous.clone_from(tokens);
    }
    // Stopped before its first block, a feeding keeps nothing, and the next
    // answers as a fresh session.
    let (tokens, from) = sequence("こん", "乙");
    let mut asked = 0;
    let mut stop = || {
        asked += 1;
        true
    };
    assert!(
        !judge
            .likeliest(&tokens, from, &mut likeliest, &mut stop)
            .expect("stopped")
    );
    assert!(asked > 0);
    assert!(
        judge
            .likeliest(&tokens, from, &mut likeliest, &mut never)
            .expect("judged")
    );
    for (judged, fresh) in likeliest.iter().zip(fresh(&zenz, &tokens, from)) {
        if let Some(fresh) = fresh {
            assert_eq!(*judged, fresh);
        }
    }
}
