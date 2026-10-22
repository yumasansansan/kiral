// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later
//
// This file is part of KIRAL. The SPDX lines name its copyright holder and its
// license, in the machine-readable form of the REUSE specification: the GNU
// General Public License, version 3 or any later version
// (LICENSES/GPL-3.0-or-later.txt).

//! AJIMEE-Bench's items: `JWTD_v2/v1/evaluation_items.json`, 200 readings in
//! katakana, 100 of them with the text to their left, each with the
//! conversions it accepts.

use kiral_model::json::{self, Value};

/// Where `scripts/fetch_datasets.py` puts the items, from the repository's root.
pub const ITEMS: &str = "data/ajimee/evaluation_items.json";

/// An item of the benchmark.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Item {
    /// Its number in the benchmark (its line in the dataset it was drawn from).
    pub index: String,
    /// The text to the left of the reading: empty for the items without one.
    pub context: String,
    /// The reading, in katakana.
    pub reading: String,
    /// The conversions it accepts.
    pub expected: Vec<String>,
}

/// The items of the benchmark's file.
///
/// # Errors
///
/// Where the bytes are not an array of items, each with the fields read.
pub fn read(bytes: &[u8]) -> Result<Vec<Item>, String> {
    let items = json::parse(bytes).map_err(|error| format!("the items are not JSON: {error}"))?;
    let items = items.as_array().ok_or("the items are not an array")?;
    items
        .iter()
        .enumerate()
        .map(|(number, item)| {
            let item = item
                .as_object()
                .ok_or_else(|| format!("item {number} is not an object"))?;
            let text = |name: &str| {
                item.get(name)
                    .and_then(Value::as_str)
                    .map(str::to_owned)
                    .ok_or_else(|| format!("item {number} has no {name} that is a string"))
            };
            let expected = item
                .get("expected_output")
                .and_then(Value::as_array)
                .ok_or_else(|| format!("item {number} has no expected_output that is an array"))?
                .iter()
                .map(|value| value.as_str().map(str::to_owned))
                .collect::<Option<Vec<_>>>()
                .filter(|expected| !expected.is_empty())
                .ok_or_else(|| format!("item {number} accepts no conversion, or one that is not a string"))?;
            Ok(Item {
                index: text("index")?,
                context: text("context_text")?,
                reading: text("input")?,
                expected,
            })
        })
        .collect()
}

/// The items of a file -- the benchmark's own where none is named -- the first
/// `limit` of them where a limit is given.
///
/// # Errors
///
/// Where the file cannot be read or holds no items, with how to fetch it.
pub fn load(path: Option<std::path::PathBuf>, limit: Option<usize>) -> Result<Vec<Item>, String> {
    let path = path.unwrap_or_else(|| std::path::PathBuf::from(ITEMS));
    let bytes = std::fs::read(&path).map_err(|error| {
        format!(
            "{}: {error} (scripts/fetch_datasets.py fetch ajimee fetches it)",
            path.display()
        )
    })?;
    let mut items = read(&bytes)?;
    if let Some(limit) = limit {
        items.truncate(limit);
    }
    if items.is_empty() {
        return Err(format!("{}: no items", path.display()));
    }
    Ok(items)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn items_are_read() {
        let bytes = r#"[{"index": "1890", "context_text": "", "input": "カケイヲ",
            "expected_output": ["家計を", "家計ヲ"], "original_text": "", "splitted_input_for_limited_input_length": []}]"#;
        let items = read(bytes.as_bytes()).expect("items");
        assert_eq!(
            items,
            [Item {
                index: "1890".to_owned(),
                context: String::new(),
                reading: "カケイヲ".to_owned(),
                expected: vec!["家計を".to_owned(), "家計ヲ".to_owned()],
            }]
        );
        assert!(read(br#"[{"index": "1", "context_text": "", "input": "", "expected_output": []}]"#).is_err());
        assert!(read(br#"{"index": "1"}"#).is_err());
    }
}
