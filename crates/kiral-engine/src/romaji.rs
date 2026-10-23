// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later
//
// This file is part of KIRAL. The SPDX lines name its copyright holder and its
// license, in the machine-readable form of the REUSE specification: the GNU
// General Public License, version 3 or any later version
// (LICENSES/GPL-3.0-or-later.txt).

//! Romaji composed into kana, as Mozc's default romaji table composes them, a
//! key at a time.
//!
//! A rule of the table takes keys to kana, and may leave keys to be typed
//! again: `ka` makes か; `kk` makes っ and leaves `k`, so that `kka` is っか;
//! `ww` before any letter but w makes っ (`wwa`: っわ) while `www` stays three
//! w, as `wwww` is typed for laughter. Keys wait while a longer rule begins
//! with them -- `n` for `na`, `ny` for `nya` -- and are taken by the rule they
//! make when the next key makes none (`nk`: ん, and `k` waits); keys that make
//! no rule are kept as they are, full-width (`q` before `w`: ｑ, then `wa`
//! makes わ). The symbols of ASCII make their full-width forms, or the marks
//! the table gives them: `-` ー, `,` 、, `.` 。, `[` 「, `z[` 『, `z.` …, `/` ・.
//!
//! An uppercase letter is taken as it is typed, and so is every key after it
//! until the composition ends or the letter is deleted (`iPhone`: いPhone),
//! as Mozc's shift makes them; deleting the letter, the keys that waited before
//! it wait again. The table and the way its keys are taken agree with Mozc
//! 2.29, typed through its helper with a profile of its own, on every sequence
//! of up to three lowercase letters, of three consonants and a vowel, and on
//! every other printable character of ASCII alone and after `z`: 64,719
//! sequences, besides the uppercase letters' and backspaces' of the tests.

/// The rules, sorted by their keys: the keys, the kana they make, and the keys
/// they leave to be typed again.
static RULES: [(&str, &str, &str); 375] = [
    ("!", "！", ""),
    ("\"", "”", ""),
    ("#", "＃", ""),
    ("$", "＄", ""),
    ("%", "％", ""),
    ("&", "＆", ""),
    ("'", "’", ""),
    ("(", "（", ""),
    (")", "）", ""),
    ("*", "＊", ""),
    ("+", "＋", ""),
    (",", "、", ""),
    ("-", "ー", ""),
    (".", "。", ""),
    ("/", "・", ""),
    ("0", "０", ""),
    ("1", "１", ""),
    ("2", "２", ""),
    ("3", "３", ""),
    ("4", "４", ""),
    ("5", "５", ""),
    ("6", "６", ""),
    ("7", "７", ""),
    ("8", "８", ""),
    ("9", "９", ""),
    (":", "：", ""),
    (";", "；", ""),
    ("<", "＜", ""),
    ("=", "＝", ""),
    (">", "＞", ""),
    ("?", "？", ""),
    ("@", "＠", ""),
    ("[", "「", ""),
    ("\\", "￥", ""),
    ("]", "」", ""),
    ("^", "＾", ""),
    ("_", "＿", ""),
    ("`", "｀", ""),
    ("a", "あ", ""),
    ("ba", "ば", ""),
    ("bb", "っ", "b"),
    ("be", "べ", ""),
    ("bi", "び", ""),
    ("bo", "ぼ", ""),
    ("bu", "ぶ", ""),
    ("bya", "びゃ", ""),
    ("bye", "びぇ", ""),
    ("byi", "びぃ", ""),
    ("byo", "びょ", ""),
    ("byu", "びゅ", ""),
    ("ca", "か", ""),
    ("cc", "っ", "c"),
    ("ce", "せ", ""),
    ("cha", "ちゃ", ""),
    ("che", "ちぇ", ""),
    ("chi", "ち", ""),
    ("cho", "ちょ", ""),
    ("chu", "ちゅ", ""),
    ("ci", "し", ""),
    ("co", "こ", ""),
    ("cu", "く", ""),
    ("cya", "ちゃ", ""),
    ("cye", "ちぇ", ""),
    ("cyi", "ちぃ", ""),
    ("cyo", "ちょ", ""),
    ("cyu", "ちゅ", ""),
    ("da", "だ", ""),
    ("dd", "っ", "d"),
    ("de", "で", ""),
    ("dha", "でゃ", ""),
    ("dhe", "でぇ", ""),
    ("dhi", "でぃ", ""),
    ("dho", "でょ", ""),
    ("dhu", "でゅ", ""),
    ("di", "ぢ", ""),
    ("do", "ど", ""),
    ("du", "づ", ""),
    ("dwa", "どぁ", ""),
    ("dwe", "どぇ", ""),
    ("dwi", "どぃ", ""),
    ("dwo", "どぉ", ""),
    ("dwu", "どぅ", ""),
    ("dya", "ぢゃ", ""),
    ("dye", "ぢぇ", ""),
    ("dyi", "ぢぃ", ""),
    ("dyo", "ぢょ", ""),
    ("dyu", "ぢゅ", ""),
    ("e", "え", ""),
    ("fa", "ふぁ", ""),
    ("fe", "ふぇ", ""),
    ("ff", "っ", "f"),
    ("fi", "ふぃ", ""),
    ("fo", "ふぉ", ""),
    ("fu", "ふ", ""),
    ("fya", "ふゃ", ""),
    ("fyo", "ふょ", ""),
    ("fyu", "ふゅ", ""),
    ("ga", "が", ""),
    ("ge", "げ", ""),
    ("gg", "っ", "g"),
    ("gi", "ぎ", ""),
    ("go", "ご", ""),
    ("gu", "ぐ", ""),
    ("gwa", "ぐぁ", ""),
    ("gwe", "ぐぇ", ""),
    ("gwi", "ぐぃ", ""),
    ("gwo", "ぐぉ", ""),
    ("gwu", "ぐぅ", ""),
    ("gya", "ぎゃ", ""),
    ("gye", "ぎぇ", ""),
    ("gyi", "ぎぃ", ""),
    ("gyo", "ぎょ", ""),
    ("gyu", "ぎゅ", ""),
    ("ha", "は", ""),
    ("he", "へ", ""),
    ("hh", "っ", "h"),
    ("hi", "ひ", ""),
    ("ho", "ほ", ""),
    ("hu", "ふ", ""),
    ("hwa", "ふぁ", ""),
    ("hwe", "ふぇ", ""),
    ("hwi", "ふぃ", ""),
    ("hwo", "ふぉ", ""),
    ("hwyu", "ふゅ", ""),
    ("hya", "ひゃ", ""),
    ("hye", "ひぇ", ""),
    ("hyi", "ひぃ", ""),
    ("hyo", "ひょ", ""),
    ("hyu", "ひゅ", ""),
    ("i", "い", ""),
    ("ja", "じゃ", ""),
    ("je", "じぇ", ""),
    ("ji", "じ", ""),
    ("jj", "っ", "j"),
    ("jo", "じょ", ""),
    ("ju", "じゅ", ""),
    ("jya", "じゃ", ""),
    ("jye", "じぇ", ""),
    ("jyi", "じぃ", ""),
    ("jyo", "じょ", ""),
    ("jyu", "じゅ", ""),
    ("ka", "か", ""),
    ("ke", "け", ""),
    ("ki", "き", ""),
    ("kk", "っ", "k"),
    ("ko", "こ", ""),
    ("ku", "く", ""),
    ("kwa", "くぁ", ""),
    ("kwe", "くぇ", ""),
    ("kwi", "くぃ", ""),
    ("kwo", "くぉ", ""),
    ("kwu", "くぅ", ""),
    ("kya", "きゃ", ""),
    ("kye", "きぇ", ""),
    ("kyi", "きぃ", ""),
    ("kyo", "きょ", ""),
    ("kyu", "きゅ", ""),
    ("la", "ぁ", ""),
    ("le", "ぇ", ""),
    ("li", "ぃ", ""),
    ("lka", "ヵ", ""),
    ("lke", "ヶ", ""),
    ("ll", "っ", "l"),
    ("lo", "ぉ", ""),
    ("ltsu", "っ", ""),
    ("ltu", "っ", ""),
    ("lu", "ぅ", ""),
    ("lwa", "ゎ", ""),
    ("lya", "ゃ", ""),
    ("lye", "ぇ", ""),
    ("lyi", "ぃ", ""),
    ("lyo", "ょ", ""),
    ("lyu", "ゅ", ""),
    ("ma", "ま", ""),
    ("me", "め", ""),
    ("mi", "み", ""),
    ("mm", "っ", "m"),
    ("mo", "も", ""),
    ("mu", "む", ""),
    ("mya", "みゃ", ""),
    ("mye", "みぇ", ""),
    ("myi", "みぃ", ""),
    ("myo", "みょ", ""),
    ("myu", "みゅ", ""),
    ("n", "ん", ""),
    ("n'", "ん", ""),
    ("na", "な", ""),
    ("ne", "ね", ""),
    ("ni", "に", ""),
    ("nn", "ん", ""),
    ("no", "の", ""),
    ("nu", "ぬ", ""),
    ("nya", "にゃ", ""),
    ("nye", "にぇ", ""),
    ("nyi", "にぃ", ""),
    ("nyo", "にょ", ""),
    ("nyu", "にゅ", ""),
    ("o", "お", ""),
    ("pa", "ぱ", ""),
    ("pe", "ぺ", ""),
    ("pi", "ぴ", ""),
    ("po", "ぽ", ""),
    ("pp", "っ", "p"),
    ("pu", "ぷ", ""),
    ("pya", "ぴゃ", ""),
    ("pye", "ぴぇ", ""),
    ("pyi", "ぴぃ", ""),
    ("pyo", "ぴょ", ""),
    ("pyu", "ぴゅ", ""),
    ("qa", "くぁ", ""),
    ("qe", "くぇ", ""),
    ("qi", "くぃ", ""),
    ("qo", "くぉ", ""),
    ("qq", "っ", "q"),
    ("qu", "く", ""),
    ("ra", "ら", ""),
    ("re", "れ", ""),
    ("ri", "り", ""),
    ("ro", "ろ", ""),
    ("rr", "っ", "r"),
    ("ru", "る", ""),
    ("rya", "りゃ", ""),
    ("rye", "りぇ", ""),
    ("ryi", "りぃ", ""),
    ("ryo", "りょ", ""),
    ("ryu", "りゅ", ""),
    ("sa", "さ", ""),
    ("se", "せ", ""),
    ("sha", "しゃ", ""),
    ("she", "しぇ", ""),
    ("shi", "し", ""),
    ("sho", "しょ", ""),
    ("shu", "しゅ", ""),
    ("si", "し", ""),
    ("so", "そ", ""),
    ("ss", "っ", "s"),
    ("su", "す", ""),
    ("swa", "すぁ", ""),
    ("swe", "すぇ", ""),
    ("swi", "すぃ", ""),
    ("swo", "すぉ", ""),
    ("swu", "すぅ", ""),
    ("sya", "しゃ", ""),
    ("sye", "しぇ", ""),
    ("syi", "しぃ", ""),
    ("syo", "しょ", ""),
    ("syu", "しゅ", ""),
    ("ta", "た", ""),
    ("te", "て", ""),
    ("tha", "てゃ", ""),
    ("the", "てぇ", ""),
    ("thi", "てぃ", ""),
    ("tho", "てょ", ""),
    ("thu", "てゅ", ""),
    ("ti", "ち", ""),
    ("to", "と", ""),
    ("tsa", "つぁ", ""),
    ("tse", "つぇ", ""),
    ("tsi", "つぃ", ""),
    ("tso", "つぉ", ""),
    ("tsu", "つ", ""),
    ("tt", "っ", "t"),
    ("tu", "つ", ""),
    ("twa", "とぁ", ""),
    ("twe", "とぇ", ""),
    ("twi", "とぃ", ""),
    ("two", "とぉ", ""),
    ("twu", "とぅ", ""),
    ("tya", "ちゃ", ""),
    ("tye", "ちぇ", ""),
    ("tyi", "ちぃ", ""),
    ("tyo", "ちょ", ""),
    ("tyu", "ちゅ", ""),
    ("u", "う", ""),
    ("va", "ゔぁ", ""),
    ("ve", "ゔぇ", ""),
    ("vi", "ゔぃ", ""),
    ("vo", "ゔぉ", ""),
    ("vu", "ゔ", ""),
    ("vv", "っ", "v"),
    ("vya", "ゔゃ", ""),
    ("vye", "ゔぇ", ""),
    ("vyi", "ゔぃ", ""),
    ("vyo", "ゔょ", ""),
    ("vyu", "ゔゅ", ""),
    ("wa", "わ", ""),
    ("we", "うぇ", ""),
    ("wha", "うぁ", ""),
    ("whe", "うぇ", ""),
    ("whi", "うぃ", ""),
    ("who", "うぉ", ""),
    ("whu", "う", ""),
    ("wi", "うぃ", ""),
    ("wo", "を", ""),
    ("wu", "う", ""),
    ("ww", "ｗ", "w"),
    ("wwa", "っ", "wa"),
    ("wwb", "っ", "wb"),
    ("wwc", "っ", "wc"),
    ("wwd", "っ", "wd"),
    ("wwe", "っ", "we"),
    ("wwf", "っ", "wf"),
    ("wwg", "っ", "wg"),
    ("wwh", "っ", "wh"),
    ("wwi", "っ", "wi"),
    ("wwj", "っ", "wj"),
    ("wwk", "っ", "wk"),
    ("wwl", "っ", "wl"),
    ("wwm", "っ", "wm"),
    ("wwn", "っ", "wn"),
    ("wwo", "っ", "wo"),
    ("wwp", "っ", "wp"),
    ("wwq", "っ", "wq"),
    ("wwr", "っ", "wr"),
    ("wws", "っ", "ws"),
    ("wwt", "っ", "wt"),
    ("wwu", "っ", "wu"),
    ("wwv", "っ", "wv"),
    ("wwx", "っ", "wx"),
    ("wwy", "っ", "wy"),
    ("wwz", "っ", "wz"),
    ("wye", "ゑ", ""),
    ("wyi", "ゐ", ""),
    ("xa", "ぁ", ""),
    ("xe", "ぇ", ""),
    ("xi", "ぃ", ""),
    ("xka", "ヵ", ""),
    ("xke", "ヶ", ""),
    ("xn", "ん", ""),
    ("xo", "ぉ", ""),
    ("xtsu", "っ", ""),
    ("xtu", "っ", ""),
    ("xu", "ぅ", ""),
    ("xwa", "ゎ", ""),
    ("xx", "っ", "x"),
    ("xya", "ゃ", ""),
    ("xye", "ぇ", ""),
    ("xyi", "ぃ", ""),
    ("xyo", "ょ", ""),
    ("xyu", "ゅ", ""),
    ("ya", "や", ""),
    ("ye", "いぇ", ""),
    ("yo", "よ", ""),
    ("yu", "ゆ", ""),
    ("yy", "っ", "y"),
    ("z,", "‥", ""),
    ("z-", "〜", ""),
    ("z.", "…", ""),
    ("z/", "・", ""),
    ("z[", "『", ""),
    ("z]", "』", ""),
    ("za", "ざ", ""),
    ("ze", "ぜ", ""),
    ("zh", "←", ""),
    ("zi", "じ", ""),
    ("zj", "↓", ""),
    ("zk", "↑", ""),
    ("zl", "→", ""),
    ("zo", "ぞ", ""),
    ("zu", "ず", ""),
    ("zwa", "ずぁ", ""),
    ("zwe", "ずぇ", ""),
    ("zwi", "ずぃ", ""),
    ("zwo", "ずぉ", ""),
    ("zwu", "ずぅ", ""),
    ("zya", "じゃ", ""),
    ("zye", "じぇ", ""),
    ("zyi", "じぃ", ""),
    ("zyo", "じょ", ""),
    ("zyu", "じゅ", ""),
    ("zz", "っ", "z"),
    ("{", "｛", ""),
    ("|", "｜", ""),
    ("}", "｝", ""),
    ("~", "〜", ""),
];

/// The rule of keys, where the table has one: its kana, and the keys it leaves.
fn rule(keys: &str) -> Option<(&'static str, &'static str)> {
    RULES
        .binary_search_by(|(rule, _, _)| (*rule).cmp(keys))
        .ok()
        .map(|at| (RULES[at].1, RULES[at].2))
}

/// Whether a longer rule begins with the keys: the rules that begin with them
/// follow them in order, the first right after.
fn longer(keys: &str) -> bool {
    let after = RULES.partition_point(|(rule, _, _)| *rule <= keys);
    RULES.get(after).is_some_and(|(rule, _, _)| rule.starts_with(keys))
}

/// Whether the table takes a key: the printable characters of ASCII, but the
/// uppercase letters.
fn in_table(key: char) -> bool {
    ('!'..='~').contains(&key) && !key.is_ascii_uppercase()
}

/// A character of ASCII in its full-width form; any other as it is.
fn full_width(character: char) -> char {
    if ('!'..='~').contains(&character) {
        char::from_u32(u32::from(character) + 0xfee0).unwrap_or(character)
    } else {
        character
    }
}

/// A full-width letter of ASCII as the letter; any other character as it is.
fn half_width_letter(character: char) -> char {
    match character {
        'ａ'..='ｚ' | 'Ａ'..='Ｚ' => char::from_u32(u32::from(character) - 0xfee0).unwrap_or(character),
        _ => character,
    }
}

/// What is typed of a composition: its kana, and the keys that wait for a rule.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Composer {
    /// The kana made, the marks the table makes, and the keys kept as they are.
    kana: String,
    /// The keys that wait for a rule.
    pending: String,
    /// Where the keys taken as they are typed begin in the kana, from an
    /// uppercase letter on.
    typed_from: Option<usize>,
    /// The keys that waited when the uppercase letter came, kept as they are
    /// before it, and waiting again once it is deleted.
    suspended: String,
}

impl Composer {
    /// A composition of nothing.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Types a key.
    pub fn push(&mut self, key: char) {
        if self.typed_from.is_some() {
            self.kana.push(key);
            return;
        }
        if !in_table(key) {
            // The keys that wait stay as they were typed.
            let pending = std::mem::take(&mut self.pending);
            self.kana.extend(pending.chars().map(full_width));
            if key.is_ascii_uppercase() {
                self.typed_from = Some(self.kana.len());
                self.suspended = pending;
            }
            self.kana.push(key);
            return;
        }
        self.feed(key);
    }

    /// Feeds a key of the table to the keys that wait.
    fn feed(&mut self, key: char) {
        let mut keys = std::mem::take(&mut self.pending);
        keys.push(key);
        if longer(&keys) {
            self.pending = keys;
            return;
        }
        if let Some((kana, again)) = rule(&keys) {
            self.kana.push_str(kana);
            again.chars().for_each(|key| self.feed(key));
            return;
        }
        // No rule begins with the keys: those before the last are taken by the
        // rule they make, or kept as they are, and the last is fed again.
        keys.pop();
        if keys.is_empty() {
            self.kana.push(full_width(key));
            return;
        }
        match rule(&keys) {
            Some((kana, again)) => {
                self.kana.push_str(kana);
                again.chars().for_each(|key| self.feed(key));
            }
            None => self.kana.extend(keys.chars().map(full_width)),
        }
        self.feed(key);
    }

    /// Takes the keys that wait as the rule they make, or as they are where
    /// they make none: `n` as ん, `k` as ｋ -- the composition being converted
    /// or committed.
    pub fn flush(&mut self) {
        let keys = std::mem::take(&mut self.pending);
        if keys.is_empty() {
            return;
        }
        match rule(&keys) {
            Some((kana, again)) => {
                self.kana.push_str(kana);
                again.chars().for_each(|key| self.feed(key));
                self.flush();
            }
            None => self.kana.extend(keys.chars().map(full_width)),
        }
    }

    /// Deletes the last character: a key that waits, or else a character of the
    /// kana -- the uppercase letter that began the keys taken as typed among
    /// them, and with it the taking, the keys that waited before it waiting
    /// again. Whether there was one.
    pub fn backspace(&mut self) -> bool {
        if self.pending.pop().is_some() {
            return true;
        }
        if self.kana.pop().is_none() {
            return false;
        }
        if let Some(from) = self.typed_from
            && self.kana.len() <= from
        {
            self.typed_from = None;
            let shown: usize = self.suspended.chars().map(|key| full_width(key).len_utf8()).sum();
            self.kana.truncate(from.saturating_sub(shown));
            self.pending = std::mem::take(&mut self.suspended);
        }
        true
    }

    /// Deletes everything.
    pub fn clear(&mut self) {
        self.kana.clear();
        self.pending.clear();
        self.typed_from = None;
        self.suspended.clear();
    }

    /// Whether nothing is typed.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.kana.is_empty() && self.pending.is_empty()
    }

    /// Whether keys are taken as they are typed, after an uppercase letter.
    #[must_use]
    pub fn is_typed_as_is(&self) -> bool {
        self.typed_from.is_some()
    }

    /// The kana, the keys that wait not among them.
    #[must_use]
    pub fn kana(&self) -> &str {
        &self.kana
    }

    /// The keys that wait, as they were typed.
    #[must_use]
    pub fn pending(&self) -> &str {
        &self.pending
    }

    /// The keys that wait as Mozc shows them while they are typed: full-width.
    #[must_use]
    pub fn pending_shown(&self) -> String {
        self.pending.chars().map(full_width).collect()
    }

    /// What is typed as Mozc shows it: the kana, then the keys that wait.
    #[must_use]
    pub fn preedit(&self) -> String {
        let mut preedit = self.kana.clone();
        preedit.extend(self.pending.chars().map(full_width));
        preedit
    }

    /// The kana as a reading to convert: the letters kept as they were typed in
    /// their half-width forms, as a conversion shows them.
    #[must_use]
    pub fn reading(&self) -> String {
        self.kana.chars().map(half_width_letter).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// What typing keys shows; `⌫` is a backspace.
    fn typed(keys: &str) -> String {
        let mut composer = Composer::new();
        for key in keys.chars() {
            if key == '⌫' {
                composer.backspace();
            } else {
                composer.push(key);
            }
        }
        composer.preedit()
    }

    #[test]
    fn the_rules_are_sorted_and_each_once() {
        assert!(RULES.windows(2).all(|pair| pair[0].0 < pair[1].0));
        assert!(RULES.iter().all(|(keys, kana, _)| !keys.is_empty() && !kana.is_empty()));
        assert!(
            RULES
                .iter()
                .all(|(_, _, again)| again.chars().all(|key| key.is_ascii_lowercase()))
        );
    }

    #[test]
    fn keys_compose_as_mozc_composes_them() {
        // As Mozc 2.29 showed them, typed through its helper.
        for (keys, shown) in [
            ("konnichiha", "こんいちは"),
            ("konnnichiha", "こんにちは"),
            ("kon'nichiha", "こんにちは"),
            ("shinbun", "しんぶｎ"),
            ("kanji", "かんじ"),
            ("nk", "んｋ"),
            ("nyn", "ｎｙｎ"),
            ("nnn", "んｎ"),
            ("kka", "っか"),
            ("kkya", "っきゃ"),
            ("tte", "って"),
            ("ww", "ｗｗ"),
            ("wwa", "っわ"),
            ("wwwa", "ｗっわ"),
            ("wwwka", "ｗっｗか"),
            ("wwh", "っｗｈ"),
            ("lwi", "ｌｗい"),
            ("lwwa", "ｌｗわ"),
            ("qwa", "ｑわ"),
            ("qwwa", "ｑっわ"),
            ("hwyu", "ふゅ"),
            ("hwya", "ｈｗｙあ"),
            ("ltsu", "っ"),
            ("xxtu", "っっ"),
            ("chi", "ち"),
            ("tsu", "つ"),
            ("fu", "ふ"),
            ("ja", "じゃ"),
            ("dhi", "でぃ"),
            ("twu", "とぅ"),
            ("yi", "ｙい"),
            ("ye", "いぇ"),
            ("wo", "を"),
            ("vu", "ゔ"),
            ("xka", "ヵ"),
            ("1-2", "１ー２"),
            ("a,b.", "あ、ｂ。"),
            ("[z[z.z]]", "「『…』」"),
            ("z/zhzjzkzlz-z,~", "・←↓↑→〜‥〜"),
            ("\\'\"`", "￥’”｀"),
            ("!?#", "！？＃"),
        ] {
            assert_eq!(typed(keys), shown, "{keys}");
        }
    }

    #[test]
    fn an_uppercase_letter_and_what_follows_it_are_taken_as_typed() {
        for (keys, shown) in [
            ("iPhone", "いPhone"),
            ("A1,-", "A1,-"),
            ("kA", "ｋA"),
            ("nA", "ｎA"),
            ("Ab⌫a", "Aa"),
            ("A⌫a", "あ"),
            ("kA⌫a", "か"),
            ("nA⌫a", "な"),
            ("kyA⌫a", "きゃ"),
            ("kaA⌫", "か"),
            ("kya⌫", "き"),
            ("ky⌫", "ｋ"),
            ("kk⌫", "っ"),
            ("nn⌫", ""),
        ] {
            assert_eq!(typed(keys), shown, "{keys}");
        }
    }

    #[test]
    fn a_composition_flushed_takes_its_waiting_keys() {
        let mut composer = Composer::new();
        "shinbun".chars().for_each(|key| composer.push(key));
        assert_eq!(composer.kana(), "しんぶ");
        assert_eq!(composer.pending(), "n");
        assert_eq!(composer.reading(), "しんぶ");
        composer.flush();
        assert_eq!(composer.preedit(), "しんぶん");
        let mut composer = Composer::new();
        "kany".chars().for_each(|key| composer.push(key));
        composer.flush();
        assert_eq!(composer.preedit(), "かｎｙ");
        assert_eq!(composer.reading(), "かny");
        let mut composer = Composer::new();
        "nA⌫".chars().for_each(|key| {
            if key == '⌫' {
                composer.backspace();
            } else {
                composer.push(key);
            }
        });
        composer.flush();
        assert_eq!(composer.preedit(), "ん");
        let mut composer = Composer::new();
        "www".chars().for_each(|key| composer.push(key));
        composer.flush();
        assert_eq!(composer.reading(), "www");
        assert!(!composer.is_empty());
        composer.clear();
        assert!(composer.is_empty() && !composer.backspace());
    }
}
