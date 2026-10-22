// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later
//
// This file is part of KIRAL. The SPDX lines name its copyright holder and its
// license, in the machine-readable form of the REUSE specification: the GNU
// General Public License, version 3 or any later version
// (LICENSES/GPL-3.0-or-later.txt).

//! The forms of a reading: as it is looked up, and as it is shown.
//!
//! A reading is looked up with its katakana as hiragana, the full-width forms
//! of ASCII as ASCII and the ideographic space as a space -- the readings of
//! the dictionary are made so when it is built, so that a reading typed in
//! either form finds the same words. A character becomes one character, so
//! that the positions of a reading are those of its lookup form. Shown, a
//! reading keeps its full-width forms, which a word that is its own reading
//! is written in (a parenthesis typed full-width stays full-width), and only
//! its katakana become hiragana.

/// A character as readings are looked up.
#[must_use]
pub fn lookup(character: char) -> char {
    match character {
        '\u{3000}' => ' ',
        '\u{ff01}'..='\u{ff5e}' => char::from_u32(u32::from(character) - 0xfee0).unwrap_or(character),
        _ => display(character),
    }
}

/// A character as a reading is shown: katakana as hiragana (ヴ as ゔ, ヵ and
/// ヶ as ゕ and ゖ, ヽ and ヾ as ゝ and ゞ), anything else as it is.
#[must_use]
pub fn display(character: char) -> char {
    match character {
        '\u{30a1}'..='\u{30f6}' | '\u{30fd}'..='\u{30fe}' => {
            char::from_u32(u32::from(character) - 0x60).unwrap_or(character)
        }
        _ => character,
    }
}

/// A reading as it is looked up.
#[must_use]
pub fn lookup_form(reading: &str) -> String {
    reading.chars().map(lookup).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn readings_are_looked_up_in_one_form() {
        assert_eq!(lookup_form("カケイヲ、（ヴ）ー～　ヶヽ"), "かけいを、(ゔ)ー~ ゖゝ");
        assert_eq!(lookup_form("かな123abc"), "かな123abc");
        assert_eq!(display('（'), '（');
        assert_eq!(display('ア'), 'あ');
        assert_eq!(display('ー'), 'ー');
        assert_eq!(display('・'), '・');
    }
}
