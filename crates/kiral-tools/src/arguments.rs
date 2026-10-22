// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later
//
// This file is part of KIRAL. The SPDX lines name its copyright holder and its
// license, in the machine-readable form of the REUSE specification: the GNU
// General Public License, version 3 or any later version
// (LICENSES/GPL-3.0-or-later.txt).

//! The options of a command: `--name value`, `--name=value` and `--flag`.

use std::ffi::OsString;
use std::path::PathBuf;
use std::str::FromStr;

/// The options a command was given.
#[derive(Debug, Default)]
pub struct Options {
    values: Vec<(String, OsString)>,
    flags: Vec<String>,
}

/// The options of the arguments, of those a command takes: the names in
/// `valued` take a value, those in `flags` none.
///
/// # Errors
///
/// An option the command does not take, given twice, or without its value, or
/// an argument that is not an option.
pub fn parse(
    arguments: impl IntoIterator<Item = OsString>,
    valued: &[&str],
    flags: &[&str],
) -> Result<Options, String> {
    let mut options = Options::default();
    let mut arguments = arguments.into_iter();
    while let Some(argument) = arguments.next() {
        let text = argument.to_string_lossy().into_owned();
        let Some(option) = text.strip_prefix("--") else {
            return Err(format!("an argument that is not an option: {text}"));
        };
        let (name, inline) = match option.split_once('=') {
            Some((name, value)) => (name.to_owned(), Some(OsString::from(value))),
            None => (option.to_owned(), None),
        };
        if options.values.iter().any(|(given, _)| *given == name) || options.flags.contains(&name) {
            return Err(format!("--{name} given twice"));
        }
        if valued.contains(&name.as_str()) {
            let value = match inline {
                Some(value) => value,
                None => arguments.next().ok_or_else(|| format!("--{name} without its value"))?,
            };
            options.values.push((name, value));
        } else if flags.contains(&name.as_str()) && inline.is_none() {
            options.flags.push(name);
        } else {
            return Err(format!("no option --{name}"));
        }
    }
    Ok(options)
}

impl Options {
    /// The value of an option, where it was given.
    pub fn value(&self, name: &str) -> Option<&OsString> {
        self.values
            .iter()
            .find(|(given, _)| given == name)
            .map(|(_, value)| value)
    }

    /// The value of an option as a path.
    pub fn path(&self, name: &str) -> Option<PathBuf> {
        self.value(name).map(PathBuf::from)
    }

    /// The value of an option as text.
    pub fn text(&self, name: &str) -> Option<String> {
        self.value(name).map(|value| value.to_string_lossy().into_owned())
    }

    /// The value of an option read as a `T`.
    ///
    /// # Errors
    ///
    /// Where the value does not read as one.
    pub fn parsed<T: FromStr>(&self, name: &str) -> Result<Option<T>, String> {
        self.text(name)
            .map(|text| {
                text.parse()
                    .map_err(|_| format!("--{name} {text}: not a value it takes"))
            })
            .transpose()
    }

    /// Whether a flag was given.
    pub fn flag(&self, name: &str) -> bool {
        self.flags.iter().any(|given| given == name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn arguments(text: &[&str]) -> Vec<OsString> {
        text.iter().map(OsString::from).collect()
    }

    #[test]
    fn options_are_read_in_both_forms() {
        let options = parse(
            arguments(&["--model", "m", "--threads=4", "--quiet"]),
            &["model", "threads"],
            &["quiet"],
        )
        .expect("options");
        assert_eq!(options.text("model").as_deref(), Some("m"));
        assert_eq!(options.parsed::<usize>("threads"), Ok(Some(4)));
        assert!(options.flag("quiet"));
        assert_eq!(options.parsed::<usize>("limit"), Ok(None));
    }

    #[test]
    fn what_a_command_does_not_take_is_refused() {
        assert!(parse(arguments(&["--model"]), &["model"], &[]).is_err());
        assert!(parse(arguments(&["--other", "x"]), &["model"], &[]).is_err());
        assert!(parse(arguments(&["--model", "a", "--model", "b"]), &["model"], &[]).is_err());
        assert!(parse(arguments(&["model"]), &["model"], &[]).is_err());
        assert!(parse(arguments(&["--quiet=yes"]), &[], &["quiet"]).is_err());
        let options = parse(arguments(&["--threads", "four"]), &["threads"], &[]).expect("options");
        assert!(options.parsed::<usize>("threads").is_err());
    }
}
