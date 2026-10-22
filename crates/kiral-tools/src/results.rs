// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later
//
// This file is part of KIRAL. The SPDX lines name its copyright holder and its
// license, in the machine-readable form of the REUSE specification: the GNU
// General Public License, version 3 or any later version
// (LICENSES/GPL-3.0-or-later.txt).

//! The results of a run, a line for each item, kept as text separated by tabs
//! in `bench/results/<bench>/` (which is not committed), named for the time of
//! the run and the system it measured, for the next run to be compared with.

use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

/// What a run did with an item.
#[derive(Debug, Clone, PartialEq)]
pub struct Line {
    /// The item's number in the benchmark.
    pub index: String,
    /// Whether the item has a left context.
    pub context: bool,
    /// Whether the conversion is one the item accepts.
    pub correct: bool,
    /// The least character error rate against those it accepts.
    pub error_rate: f64,
    /// How the conversion ended: `end`, `tag` or `limit`.
    pub stop: String,
    /// The conversion.
    pub conversion: String,
}

const COLUMNS: &str = "index\tcontext\tcorrect\tcer\tstop\tconversion";

/// The text of a field: tabs, line ends and backslashes escaped.
fn escape(text: &str) -> String {
    let mut escaped = String::with_capacity(text.len());
    for character in text.chars() {
        match character {
            '\\' => escaped.push_str("\\\\"),
            '\t' => escaped.push_str("\\t"),
            '\n' => escaped.push_str("\\n"),
            '\r' => escaped.push_str("\\r"),
            character => escaped.push(character),
        }
    }
    escaped
}

fn unescape(text: &str) -> Option<String> {
    let mut unescaped = String::with_capacity(text.len());
    let mut characters = text.chars();
    while let Some(character) = characters.next() {
        if character == '\\' {
            unescaped.push(match characters.next()? {
                '\\' => '\\',
                't' => '\t',
                'n' => '\n',
                'r' => '\r',
                _ => return None,
            });
        } else {
            unescaped.push(character);
        }
    }
    Some(unescaped)
}

/// Writes the lines, after a title, as text separated by tabs.
///
/// # Errors
///
/// Where the file cannot be written.
pub fn write(path: &Path, title: &str, lines: &[Line]) -> std::io::Result<()> {
    let mut text = format!("# {}\n{COLUMNS}\n", escape(title));
    for line in lines {
        let _ = writeln!(
            text,
            "{}\t{}\t{}\t{:.6}\t{}\t{}",
            escape(&line.index),
            u8::from(line.context),
            u8::from(line.correct),
            line.error_rate,
            escape(&line.stop),
            escape(&line.conversion)
        );
    }
    std::fs::write(path, text)
}

/// The lines of a file that [`write()`] wrote.
///
/// # Errors
///
/// Where the file cannot be read, or is not one of results.
pub fn read(path: &Path) -> Result<Vec<Line>, String> {
    let text = std::fs::read_to_string(path).map_err(|error| format!("{}: {error}", path.display()))?;
    let wrong = |number: usize| format!("{}: line {} is not one of results", path.display(), number + 1);
    let mut lines = Vec::new();
    for (number, line) in text.lines().enumerate() {
        if line.starts_with('#') || line == COLUMNS {
            continue;
        }
        let fields: Vec<&str> = line.split('\t').collect();
        let [index, context, correct, error_rate, stop, conversion] = fields[..] else {
            return Err(wrong(number));
        };
        let flag = |field: &str| match field {
            "0" => Some(false),
            "1" => Some(true),
            _ => None,
        };
        let parsed = (|| {
            Some(Line {
                index: unescape(index)?,
                context: flag(context)?,
                correct: flag(correct)?,
                error_rate: error_rate.parse().ok()?,
                stop: unescape(stop)?,
                conversion: unescape(conversion)?,
            })
        })();
        lines.push(parsed.ok_or_else(|| wrong(number))?);
    }
    Ok(lines)
}

/// The year, month and day of a day counted from 1970-01-01 (Howard Hinnant's
/// `civil_from_days`, for days after it).
fn civil_from_days(days: u64) -> (u64, u64, u64) {
    let z = days + 719_468;
    let era = z / 146_097;
    let day_of_era = z % 146_097;
    let year_of_era = (day_of_era - day_of_era / 1460 + day_of_era / 36_524 - day_of_era / 146_096) / 365;
    let day_of_year = day_of_era - (365 * year_of_era + year_of_era / 4 - year_of_era / 100);
    let shifted_month = (5 * day_of_year + 2) / 153;
    let day = day_of_year - (153 * shifted_month + 2) / 5 + 1;
    let month = if shifted_month < 10 {
        shifted_month + 3
    } else {
        shifted_month - 9
    };
    let year = year_of_era + era * 400 + u64::from(month <= 2);
    (year, month, day)
}

/// The time of a number of seconds since 1970, in UTC: `20261022T131502Z`.
fn timestamp_of(seconds: u64) -> String {
    let (year, month, day) = civil_from_days(seconds / 86_400);
    let time = seconds % 86_400;
    format!(
        "{year:04}{month:02}{day:02}T{:02}{:02}{:02}Z",
        time / 3600,
        time / 60 % 60,
        time % 60
    )
}

/// The time now, in UTC, as [`timestamp_of`] writes it.
pub fn timestamp() -> String {
    timestamp_of(
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |since| since.as_secs()),
    )
}

/// A label as it can go in a file's name: what is not a letter, a digit, a
/// dot, a hyphen or an underscore is a hyphen.
pub fn label_for_files(label: &str) -> String {
    label
        .chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() || matches!(character, '.' | '-' | '_') {
                character
            } else {
                '-'
            }
        })
        .collect()
}

/// The file of the latest run of a label in a directory: the last of the names
/// `<time>-<label>.tsv`, whose times sort as their names do.
pub fn latest(directory: &Path, label: &str) -> Option<PathBuf> {
    let suffix = format!("-{}.tsv", label_for_files(label));
    std::fs::read_dir(directory)
        .ok()?
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .filter(|path| {
            path.file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name.ends_with(&suffix) && name.len() == 16 + suffix.len())
        })
        .max()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn times_are_written_in_utc() {
        assert_eq!(timestamp_of(0), "19700101T000000Z");
        assert_eq!(timestamp_of(951_782_400), "20000229T000000Z");
        assert_eq!(timestamp_of(1_792_674_902), "20261022T131502Z");
        assert_eq!(timestamp().len(), 16);
    }

    #[test]
    fn lines_go_back_to_what_they_were() {
        let directory = std::env::temp_dir().join(format!("kiral-results-{}", std::process::id()));
        std::fs::create_dir_all(&directory).expect("a directory");
        let lines = vec![
            Line {
                index: "7".to_owned(),
                context: true,
                correct: false,
                error_rate: 0.125,
                stop: "end".to_owned(),
                conversion: "a\tb\\c\nd".to_owned(),
            },
            Line {
                index: "12".to_owned(),
                context: false,
                correct: true,
                error_rate: 0.0,
                stop: "limit".to_owned(),
                conversion: "の依頼".to_owned(),
            },
        ];
        let path = directory.join("20261022T131502Z-zenz.tsv");
        write(&path, "a title", &lines).expect("written");
        assert_eq!(read(&path), Ok(lines));
        std::fs::write(directory.join("20261022T131501Z-zenz.tsv"), "").expect("written");
        std::fs::write(directory.join("20261022T131503Z-zenz-other.tsv"), "").expect("written");
        assert_eq!(latest(&directory, "zenz"), Some(path));
        std::fs::remove_dir_all(&directory).expect("removed");
        assert_eq!(label_for_files("zenz v3.1/small"), "zenz-v3.1-small");
    }
}
