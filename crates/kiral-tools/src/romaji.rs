// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later
//
// This file is part of KIRAL. The SPDX lines name its copyright holder and its
// license, in the machine-readable form of the REUSE specification: the GNU
// General Public License, version 3 or any later version
// (LICENSES/GPL-3.0-or-later.txt).

//! The keys that type a reading into an input method with Mozc's default
//! romaji table, and the preedit they make.
//!
//! A kana is typed as its romaji, two kana that make one syllable as one
//! (`kya`, `fa`, `dhu`); ン is always `nn`, which the table reads no other way;
//! ッ doubles the consonant after it where the table doubles that consonant, and
//! is `xtu` elsewhere; a small kana that makes no syllable with the one before is
//! `x` and its own. The punctuation of the readings is typed as the keys the
//! table maps to it: `,` and `.` for 、 and 。 after a kana, `[` and `]` for 「
//! and 」, `z[` and `z]` for 『 and 』, `z.` for …, `/` for ・, `(` and `)` for
//! （ and ）, `-` for ー.

/// The romaji of the syllables of two kana.
const PAIRS: [(&str, &str); 64] = [
    ("キャ", "kya"),
    ("キュ", "kyu"),
    ("キョ", "kyo"),
    ("ギャ", "gya"),
    ("ギュ", "gyu"),
    ("ギョ", "gyo"),
    ("シャ", "sya"),
    ("シュ", "syu"),
    ("ショ", "syo"),
    ("シェ", "sye"),
    ("ジャ", "zya"),
    ("ジュ", "zyu"),
    ("ジョ", "zyo"),
    ("ジェ", "zye"),
    ("チャ", "tya"),
    ("チュ", "tyu"),
    ("チョ", "tyo"),
    ("チェ", "tye"),
    ("ヂャ", "dya"),
    ("ヂュ", "dyu"),
    ("ヂョ", "dyo"),
    ("ニャ", "nya"),
    ("ニュ", "nyu"),
    ("ニョ", "nyo"),
    ("ヒャ", "hya"),
    ("ヒュ", "hyu"),
    ("ヒョ", "hyo"),
    ("ビャ", "bya"),
    ("ビュ", "byu"),
    ("ビョ", "byo"),
    ("ピャ", "pya"),
    ("ピュ", "pyu"),
    ("ピョ", "pyo"),
    ("ミャ", "mya"),
    ("ミュ", "myu"),
    ("ミョ", "myo"),
    ("リャ", "rya"),
    ("リュ", "ryu"),
    ("リョ", "ryo"),
    ("ファ", "fa"),
    ("フィ", "fi"),
    ("フェ", "fe"),
    ("フォ", "fo"),
    ("フュ", "fyu"),
    ("ティ", "thi"),
    ("テュ", "thu"),
    ("ディ", "dhi"),
    ("デュ", "dhu"),
    ("トゥ", "twu"),
    ("ドゥ", "dwu"),
    ("ウィ", "wi"),
    ("ウェ", "we"),
    ("ウォ", "who"),
    ("イェ", "ye"),
    ("ヴァ", "va"),
    ("ヴィ", "vi"),
    ("ヴェ", "ve"),
    ("ヴォ", "vo"),
    ("ヴュ", "vyu"),
    ("クァ", "kwa"),
    ("ツァ", "tsa"),
    ("ツィ", "tsi"),
    ("ツェ", "tse"),
    ("ツォ", "tso"),
];

/// The romaji of one kana, or the keys of a punctuation mark.
fn single(character: char) -> Option<&'static str> {
    Some(match character {
        'ア' => "a",
        'イ' => "i",
        'ウ' => "u",
        'エ' => "e",
        'オ' => "o",
        'カ' => "ka",
        'キ' => "ki",
        'ク' => "ku",
        'ケ' => "ke",
        'コ' => "ko",
        'ガ' => "ga",
        'ギ' => "gi",
        'グ' => "gu",
        'ゲ' => "ge",
        'ゴ' => "go",
        'サ' => "sa",
        'シ' => "si",
        'ス' => "su",
        'セ' => "se",
        'ソ' => "so",
        'ザ' => "za",
        'ジ' => "zi",
        'ズ' => "zu",
        'ゼ' => "ze",
        'ゾ' => "zo",
        'タ' => "ta",
        'チ' => "ti",
        'ツ' => "tu",
        'テ' => "te",
        'ト' => "to",
        'ダ' => "da",
        'ヂ' => "di",
        'ヅ' => "du",
        'デ' => "de",
        'ド' => "do",
        'ナ' => "na",
        'ニ' => "ni",
        'ヌ' => "nu",
        'ネ' => "ne",
        'ノ' => "no",
        'ハ' => "ha",
        'ヒ' => "hi",
        'フ' => "hu",
        'ヘ' => "he",
        'ホ' => "ho",
        'バ' => "ba",
        'ビ' => "bi",
        'ブ' => "bu",
        'ベ' => "be",
        'ボ' => "bo",
        'パ' => "pa",
        'ピ' => "pi",
        'プ' => "pu",
        'ペ' => "pe",
        'ポ' => "po",
        'マ' => "ma",
        'ミ' => "mi",
        'ム' => "mu",
        'メ' => "me",
        'モ' => "mo",
        'ヤ' => "ya",
        'ユ' => "yu",
        'ヨ' => "yo",
        'ラ' => "ra",
        'リ' => "ri",
        'ル' => "ru",
        'レ' => "re",
        'ロ' => "ro",
        'ワ' => "wa",
        'ヰ' => "wyi",
        'ヱ' => "wye",
        'ヲ' => "wo",
        'ン' => "nn",
        'ヴ' => "vu",
        'ァ' => "xa",
        'ィ' => "xi",
        'ゥ' => "xu",
        'ェ' => "xe",
        'ォ' => "xo",
        'ャ' => "xya",
        'ュ' => "xyu",
        'ョ' => "xyo",
        'ッ' => "xtu",
        'ヮ' => "xwa",
        'ヵ' => "xka",
        'ヶ' => "xke",
        'ー' => "-",
        '、' => ",",
        '。' => ".",
        '「' => "[",
        '」' => "]",
        '『' => "z[",
        '』' => "z]",
        '…' => "z.",
        '・' => "/",
        '（' => "(",
        '）' => ")",
        _ => return None,
    })
}

/// Hiragana as katakana; anything else as it is.
fn katakana(character: char) -> char {
    match character {
        'ぁ'..='ゖ' => char::from_u32(u32::from(character) + 0x60).unwrap_or(character),
        _ => character,
    }
}

/// The romaji of the syllable at the start of `rest`, and the kana it takes.
fn syllable(rest: &[char]) -> Option<(&'static str, usize)> {
    if let [first, second, ..] = rest {
        let pair: String = [*first, *second].iter().collect();
        if let Some(&(_, romaji)) = PAIRS.iter().find(|(kana, _)| *kana == pair) {
            return Some((romaji, 2));
        }
    }
    single(*rest.first()?).map(|romaji| (romaji, 1))
}

/// The keys that type a reading, in katakana or hiragana, into Mozc's default
/// romaji table: ASCII, one key a byte.
///
/// # Errors
///
/// The first character the table has no keys for.
pub fn keys(reading: &str) -> Result<String, char> {
    let kana: Vec<char> = reading.chars().map(katakana).collect();
    let mut keys = String::with_capacity(3 * kana.len());
    let mut at = 0;
    while at < kana.len() {
        if kana[at] == 'ッ' {
            // The consonant of what follows, doubled where the table doubles it.
            let doubled = syllable(&kana[at + 1..])
                .and_then(|(romaji, _)| romaji.chars().next())
                .filter(|consonant| "bcdfghjkprstvz".contains(*consonant));
            match doubled {
                Some(consonant) => keys.push(consonant),
                None => keys.push_str("xtu"),
            }
            at += 1;
            continue;
        }
        let (romaji, taken) = syllable(&kana[at..]).ok_or(kana[at])?;
        keys.push_str(romaji);
        at += taken;
    }
    Ok(keys)
}

/// The preedit that typing a reading makes: its kana in hiragana -- but ヵ and
/// ヶ, which the table makes in katakana -- and its punctuation as it is.
pub fn preedit(reading: &str) -> String {
    reading
        .chars()
        .map(|character| match character {
            'ァ'..='ヴ' => char::from_u32(u32::from(character) - 0x60).unwrap_or(character),
            character => character,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn readings_are_typed_as_the_table_reads_them() {
        assert_eq!(keys("カケイヲタスケルタメ、"), Ok("kakeiwotasukerutame,".to_owned()));
        assert_eq!(keys("ガッコウ"), Ok("gakkou".to_owned()));
        assert_eq!(keys("キッチン"), Ok("kittinn".to_owned()));
        assert_eq!(keys("ハッパ"), Ok("happa".to_owned()));
        assert_eq!(keys("アッ。"), Ok("axtu.".to_owned()));
        assert_eq!(keys("ヨッン"), Ok("yoxtunn".to_owned()));
        assert_eq!(keys("ジェット"), Ok("zyetto".to_owned()));
        assert_eq!(keys("キャンセル"), Ok("kyannseru".to_owned()));
        assert_eq!(keys("フィードバック"), Ok("fi-dobakku".to_owned()));
        assert_eq!(keys("デュエット・トゥ"), Ok("dhuetto/twu".to_owned()));
        assert_eq!(keys("「…『（）』」"), Ok("[z.z[()z]]".to_owned()));
        assert_eq!(keys("ァャ"), Ok("xaxya".to_owned()));
        assert_eq!(keys("かな"), Ok("kana".to_owned()));
        assert_eq!(keys("カナ1"), Err('1'));
    }

    #[test]
    fn the_preedit_is_the_reading_in_hiragana() {
        assert_eq!(preedit("カケイヲ、「ヴ」ー…"), "かけいを、「ゔ」ー…");
        assert_eq!(preedit("ヵヶ"), "ヵヶ");
    }
}
