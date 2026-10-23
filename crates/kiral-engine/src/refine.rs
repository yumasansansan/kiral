// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later
//
// This file is part of KIRAL. The SPDX lines name its copyright holder and its
// license, in the machine-readable form of the REUSE specification: the GNU
// General Public License, version 3 or any later version
// (LICENSES/GPL-3.0-or-later.txt).

//! The dictionary's drafts verified by the model, as Zenzai verifies them.
//!
//! A draft is the cheapest path of the reading's lattice. The model is fed the
//! prompt and the draft's tokens at once, and its likeliest token after each
//! position is compared with the draft's next token, and with the end of text
//! after the last ([`verify`]). Where every one is the model's likeliest, the
//! draft is the model's own greedy conversion, and it stands. At the first that
//! is not, the text the model accepted and the token it wants instead make a
//! constraint -- where the model wants the text to end, the text it accepted
//! alone -- and the lattice is searched again for the cheapest path whose text
//! begins so, or is that text: the next round's draft ([`refine`]). A round
//! adds at least a token to the constraint, or the refinement ends: where no
//! path's text is what the model wants, where the model's token is the start of
//! one the draft already has, spelt in other tokens, and when the rounds allowed
//! run out. The text is a path of the lattice whatever happens, so that the
//! conversion is of the dictionary's words and covers the reading.
//!
//! The model's side is a [`Judge`]. [`ZenzJudge`] feeds zenz, keeping the keys
//! and values of the tokens a sequence shares with the one fed before, so that
//! a round -- or a key, whose reading the prompt holds before the draft --
//! feeds only what changed.

use kiral_dict::Dictionary;
use kiral_lattice::{Constraint, Lattice, Path};
use kiral_nn::Error;
use kiral_nn::gpt2::Session;
use kiral_nn::tokenizer::Tokenizer;
use kiral_nn::zenz::{self, Zenz};

/// The model's side of verification: its likeliest token after each position
/// of a sequence.
pub trait Judge {
    /// Writes into `likeliest` the likeliest token after each of
    /// `tokens[from..]`, the one after `tokens[i]` being the model's guess of
    /// `tokens[i + 1]`. `stop` is asked as the work goes on; where it says to
    /// stop, `likeliest` is not written and the answer is false.
    ///
    /// # Errors
    ///
    /// Where the model cannot compute the tokens.
    fn likeliest(
        &mut self,
        tokens: &[u32],
        from: usize,
        likeliest: &mut Vec<u32>,
        stop: &mut dyn FnMut() -> bool,
    ) -> Result<bool, Error>;
}

/// The likeliest token after a position where it was not computed.
const NOT_COMPUTED: u32 = u32::MAX;

/// zenz as a judge: a session of the model, and the tokens in its cache with
/// the likeliest token after each.
#[derive(Debug)]
pub struct ZenzJudge<'m> {
    session: Session<'m>,
    /// The tokens whose keys and values the session holds, in order.
    fed: Vec<u32>,
    /// The likeliest token after each of them, [`NOT_COMPUTED`] where its
    /// logits were not computed.
    after: Vec<u32>,
    /// The likeliest tokens a feeding gives.
    likeliest: Vec<Option<usize>>,
    /// The tokens fed to the model so far: what the refinements cost.
    tokens_fed: usize,
}

impl<'m> ZenzJudge<'m> {
    /// A judge of zenz with room for `capacity` positions, at most the model's.
    ///
    /// # Panics
    ///
    /// Where `capacity` is more than the positions of the model.
    #[must_use]
    pub fn new(zenz: &'m Zenz<'m>, capacity: usize) -> Self {
        Self {
            session: Session::new(&zenz.model, capacity),
            fed: Vec::new(),
            after: Vec::new(),
            likeliest: Vec::new(),
            tokens_fed: 0,
        }
    }

    /// The positions it has room for.
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.session.capacity()
    }

    /// The tokens fed to the model so far.
    #[must_use]
    pub fn tokens_fed(&self) -> usize {
        self.tokens_fed
    }
}

impl Judge for ZenzJudge<'_> {
    /// Keeps the tokens the sequence shares with the one fed before, as long as
    /// the likeliest token after each from `from` on is known, and feeds the
    /// rest at once, the output layer computed only from `from` on.
    fn likeliest(
        &mut self,
        tokens: &[u32],
        from: usize,
        likeliest: &mut Vec<u32>,
        stop: &mut dyn FnMut() -> bool,
    ) -> Result<bool, Error> {
        assert!(
            from < tokens.len(),
            "the likeliest tokens from {from} of {}",
            tokens.len()
        );
        let shared = self
            .fed
            .iter()
            .zip(tokens)
            .take_while(|(fed, token)| fed == token)
            .count();
        let keep = (from..shared)
            .find(|&position| self.after[position] == NOT_COMPUTED)
            .unwrap_or(shared);
        self.session.truncate(keep);
        self.fed.truncate(keep);
        self.after.truncate(keep);
        if keep < tokens.len() {
            let feed = &tokens[keep..];
            let first = from.saturating_sub(keep);
            self.likeliest.resize(feed.len() - first, None);
            if !self.session.feed_likeliest(feed, first, &mut self.likeliest, stop)? {
                return Ok(false);
            }
            self.tokens_fed += feed.len();
            self.fed.extend_from_slice(feed);
            self.after.extend(std::iter::repeat_n(NOT_COMPUTED, first));
            self.after.extend(self.likeliest.iter().map(|token| {
                token
                    .and_then(|token| u32::try_from(token).ok())
                    .unwrap_or(NOT_COMPUTED)
            }));
        }
        likeliest.clear();
        likeliest.extend_from_slice(&self.after[from..tokens.len()]);
        Ok(true)
    }
}

/// A reading as zenz-v3's prompt takes it: its hiragana as katakana.
#[must_use]
pub fn katakana(reading: &str) -> String {
    reading
        .chars()
        .map(|character| match character {
            'ぁ'..='ゖ' | 'ゝ' | 'ゞ' => char::from_u32(u32::from(character) + 0x60).unwrap_or(character),
            _ => character,
        })
        .collect()
}

/// The tokens of zenz-v3's prompt for a reading in kana and the text to its
/// left.
#[must_use]
pub fn prompt_tokens(tokenizer: &Tokenizer, reading: &str, context: &str) -> Vec<u32> {
    let mut tokens = Vec::new();
    tokenizer.encode(&zenz::prompt(&katakana(reading), context), &mut tokens);
    tokens
}

/// What the model makes of a draft.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Verdict {
    /// Each token of the draft, and the end of text after the last, is the
    /// model's likeliest.
    Accepted,
    /// At the first that is not: the draft's text before it and the model's
    /// token, or, where the model's token is the end of text, the draft's text
    /// before it as the whole text.
    Fix(Constraint),
}

/// Verifies a draft after a prompt's tokens, in one feeding of the model; none
/// where `stop` said to stop.
///
/// # Errors
///
/// Where the model cannot compute the tokens: past its room, say.
///
/// # Panics
///
/// Where the prompt is empty.
pub fn verify(
    judge: &mut dyn Judge,
    tokenizer: &Tokenizer,
    prompt: &[u32],
    draft: &str,
    stop: &mut dyn FnMut() -> bool,
) -> Result<Option<Verdict>, Error> {
    assert!(!prompt.is_empty(), "a draft verified after no prompt");
    let mut tokens = prompt.to_vec();
    tokenizer.encode(&zenz::preprocess(draft), &mut tokens);
    let mut likeliest = Vec::with_capacity(tokens.len() + 1 - prompt.len());
    if !judge.likeliest(&tokens, prompt.len() - 1, &mut likeliest, stop)? {
        return Ok(None);
    }
    let end = tokenizer.specials().end;
    let drafted = &tokens[prompt.len()..];
    for (position, &wanted) in likeliest.iter().enumerate() {
        if drafted.get(position).copied().unwrap_or(end) == wanted {
            continue;
        }
        let mut prefix: Vec<u8> = drafted[..position]
            .iter()
            .flat_map(|&token| tokenizer.piece(token))
            .copied()
            .collect();
        let complete = wanted == end;
        if !complete {
            prefix.extend_from_slice(tokenizer.piece(wanted));
        }
        return Ok(Some(Verdict::Fix(Constraint { prefix, complete })));
    }
    Ok(Some(Verdict::Accepted))
}

/// How a refinement ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Ending {
    /// The model accepted the draft, the end of text after it included: the
    /// draft is the model's own greedy conversion.
    Accepted,
    /// The model wants a text that no path of the lattice has; the last draft
    /// stands.
    NoPath,
    /// The model's token starts one that the draft already has, spelt in other
    /// tokens; the draft stands.
    NoProgress,
    /// The rounds allowed ran out before the model accepted a draft; the last
    /// draft, found within the model's last constraint, stands.
    Limit,
}

/// A conversion refined by the model.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Refined {
    /// Its text: the last draft's.
    pub text: String,
    /// The constraint the last draft was found within, for a refinement of the
    /// reading made longer to start from.
    pub constraint: Constraint,
    /// The rounds of verification it took: the model's feedings.
    pub rounds: usize,
    /// How it ended.
    pub ending: Ending,
}

/// The draft of a lattice within a constraint carried from a refinement: the
/// cheapest path the constraint admits; or the cheapest of all, and no
/// constraint, where no path is admitted or the admitted one costs at least a
/// character of no reading more than the cheapest of all -- a constraint made
/// for a shorter reading, which the reading's new characters have undone, as
/// 岸 for きし when きしゃ is 記者.
#[must_use]
pub fn draft(lattice: &Lattice<'_>, dictionary: &Dictionary<'_>, carried: &Constraint) -> (Path, Constraint) {
    let best = lattice.best(dictionary);
    if carried.prefix.is_empty() && !carried.complete {
        return (best, Constraint::default());
    }
    match lattice.best_within(dictionary, carried) {
        Some(path) if path.cost - best.cost < i64::from(dictionary.unknown().cost) => (path, carried.clone()),
        _ => (best, Constraint::default()),
    }
}

/// Refines a reading's conversion by up to `rounds` rounds of verification,
/// from its draft within a carried constraint ([`draft`]); none where `stop`
/// said to stop.
///
/// # Errors
///
/// Where the model cannot compute the tokens: past its room, say.
#[expect(
    clippy::too_many_arguments,
    reason = "the parts a refinement takes, each of its own kind"
)]
pub fn refine(
    dictionary: &Dictionary<'_>,
    lattice: &Lattice<'_>,
    tokenizer: &Tokenizer,
    judge: &mut dyn Judge,
    prompt: &[u32],
    carried: &Constraint,
    rounds: usize,
    stop: &mut dyn FnMut() -> bool,
) -> Result<Option<Refined>, Error> {
    let (path, mut constraint) = draft(lattice, dictionary, carried);
    let mut text = lattice.text(&path);
    let refined = |text: String, constraint: Constraint, rounds: usize, ending: Ending| {
        Ok(Some(Refined {
            text,
            constraint,
            rounds,
            ending,
        }))
    };
    for round in 1..=rounds {
        let Some(verdict) = verify(judge, tokenizer, prompt, &text, stop)? else {
            return Ok(None);
        };
        let Verdict::Fix(next) = verdict else {
            return refined(text, constraint, round, Ending::Accepted);
        };
        if next.admits(&text) {
            return refined(text, constraint, round, Ending::NoProgress);
        }
        let Some(path) = lattice.best_within(dictionary, &next) else {
            return refined(text, constraint, round, Ending::NoPath);
        };
        text = lattice.text(&path);
        constraint = next;
    }
    refined(text, constraint, rounds, Ending::Limit)
}
