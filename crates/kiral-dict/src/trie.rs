// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later
//
// This file is part of KIRAL. The SPDX lines name its copyright holder and its
// license, in the machine-readable form of the REUSE specification: the GNU
// General Public License, version 3 or any later version
// (LICENSES/GPL-3.0-or-later.txt).

//! A double-array trie of the readings, over the codes of their characters.
//!
//! A unit is a base and a check, four bytes each, little-endian. The child of
//! the unit `s` by the code `c` is the unit `base(s) + c`, where it is one:
//! where its check is `s`. Code 0 ends a reading: the unit it leads to holds
//! the reading's number in its base. The root is unit 0, whose check no unit's
//! number equals. A unit that is no node's has the check [`FREE`].
//!
//! The units are built as darts does: the children of a node are placed at the
//! first base where they all fit, searched from the first free unit on, a start
//! moved past the stretches that are nearly full.

use std::ops::Range;

/// The check of a unit that is no node's.
pub const FREE: u32 = u32::MAX;

/// The check of the root, which no unit's number is.
pub const ROOT: u32 = u32::MAX - 1;

/// The bytes of a unit.
pub const UNIT: usize = 8;

/// The units of the trie of keys: each key its codes, none of them 0, the keys
/// sorted and distinct; a key's number is its place among them.
///
/// # Panics
///
/// Where the keys are not sorted and distinct, hold a code 0, or need more
/// units than a check can number.
#[must_use]
pub fn build(keys: &[Vec<u16>]) -> Vec<(u32, u32)> {
    assert!(
        keys.windows(2).all(|pair| pair[0] < pair[1]),
        "keys that are not sorted and distinct"
    );
    assert!(keys.iter().all(|key| !key.contains(&0)), "a key with the code 0");
    let mut builder = Builder {
        base: vec![0],
        check: vec![ROOT],
        used: vec![false],
        next_search: 1,
    };
    if !keys.is_empty() {
        builder.insert(keys, 0..keys.len(), 0, 0);
    }
    builder.base.into_iter().zip(builder.check).collect()
}

struct Builder {
    base: Vec<u32>,
    check: Vec<u32>,
    /// Whether a base is taken, so that no two nodes are given the same.
    used: Vec<bool>,
    /// Where the search for a base starts: units before it are all taken, or
    /// nearly.
    next_search: usize,
}

impl Builder {
    fn grow(&mut self, length: usize) {
        if length > self.check.len() {
            let length = length.max(2 * self.check.len());
            self.base.resize(length, 0);
            self.check.resize(length, FREE);
            self.used.resize(length, false);
        }
    }

    /// The first base at which every code fits.
    fn find_base(&mut self, codes: &[u16]) -> usize {
        let first = usize::from(codes[0]);
        let last = usize::from(codes[codes.len() - 1]);
        let mut position = self.next_search.max(first + 1) - 1;
        let start = position;
        let mut taken = 0usize;
        let mut first_free = true;
        loop {
            position += 1;
            self.grow(position + 1);
            if self.check[position] != FREE {
                taken += 1;
                continue;
            }
            if first_free {
                self.next_search = position;
                first_free = false;
            }
            let base = position - first;
            self.grow(base + last + 1);
            if self.used[base] {
                continue;
            }
            if codes.iter().all(|&code| self.check[base + usize::from(code)] == FREE) {
                // A stretch nearly all taken is not searched again.
                if taken * 20 >= (position - start) * 19 {
                    self.next_search = position;
                }
                return base;
            }
        }
    }

    fn insert(&mut self, keys: &[Vec<u16>], range: Range<usize>, depth: usize, unit: usize) {
        let mut children: Vec<(u16, Range<usize>)> = Vec::new();
        for index in range {
            let code = keys[index].get(depth).copied().unwrap_or(0);
            match children.last_mut() {
                Some((last, group)) if *last == code => group.end = index + 1,
                _ => children.push((code, index..index + 1)),
            }
        }
        let codes: Vec<u16> = children.iter().map(|(code, _)| *code).collect();
        let base = self.find_base(&codes);
        self.used[base] = true;
        self.base[unit] = u32::try_from(base).expect("units within a base's range");
        let parent = u32::try_from(unit).expect("units within a check's range");
        assert!(parent < ROOT, "more units than a check can number");
        for &code in &codes {
            self.check[base + usize::from(code)] = parent;
        }
        for (code, group) in children {
            let child = base + usize::from(code);
            if code == 0 {
                self.base[child] = u32::try_from(group.start).expect("keys within a base's range");
            } else {
                self.insert(keys, group, depth + 1, child);
            }
        }
    }
}

/// Reads a little-endian word of four bytes at `offset`, none past the end.
fn word(bytes: &[u8], offset: usize) -> Option<u32> {
    let bytes: [u8; 4] = bytes.get(offset..offset.checked_add(4)?)?.try_into().ok()?;
    Some(u32::from_le_bytes(bytes))
}

/// The child of a unit by a code, in the bytes of the units; none where there
/// is none, or the units end first.
#[must_use]
pub fn child(units: &[u8], unit: u32, code: u16) -> Option<u32> {
    let unit_offset = usize::try_from(unit).ok()?.checked_mul(UNIT)?;
    let next = word(units, unit_offset)?.checked_add(u32::from(code))?;
    let next_offset = usize::try_from(next).ok()?.checked_mul(UNIT)?;
    (word(units, next_offset.checked_add(4)?)? == unit).then_some(next)
}

/// The number of the reading that ends at a unit, where one does.
#[must_use]
pub fn reading_at(units: &[u8], unit: u32) -> Option<u32> {
    let end = child(units, unit, 0)?;
    word(units, usize::try_from(end).ok()?.checked_mul(UNIT)?)
}

/// The units as bytes.
#[must_use]
pub fn to_bytes(units: &[(u32, u32)]) -> Vec<u8> {
    units
        .iter()
        .flat_map(|&(base, check)| base.to_le_bytes().into_iter().chain(check.to_le_bytes()))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn find(bytes: &[u8], key: &[u16]) -> Option<u32> {
        let mut unit = 0;
        for &code in key {
            unit = child(bytes, unit, code)?;
        }
        reading_at(bytes, unit)
    }

    #[test]
    fn keys_are_found_and_others_are_not() {
        let keys: Vec<Vec<u16>> = vec![
            vec![1],
            vec![1, 2],
            vec![1, 2, 3],
            vec![2, 1],
            vec![3],
            vec![3, 3, 3, 3],
        ];
        let bytes = to_bytes(&build(&keys));
        for (number, key) in keys.iter().enumerate() {
            assert_eq!(
                find(&bytes, key),
                Some(u32::try_from(number).expect("small")),
                "{key:?}"
            );
        }
        for key in [&[2][..], &[1, 3], &[3, 3], &[4], &[1, 2, 3, 4], &[]] {
            assert_eq!(find(&bytes, key), None, "{key:?}");
        }
        assert_eq!(find(&to_bytes(&build(&[])), &[1]), None);
    }

    #[test]
    fn units_that_end_early_find_nothing() {
        let keys: Vec<Vec<u16>> = vec![vec![1, 2], vec![7, 300]];
        let bytes = to_bytes(&build(&keys));
        for length in [0, 3, 8, 15, bytes.len() / 2] {
            let _ = find(&bytes[..length], &[1, 2]);
            let _ = find(&bytes[..length], &[7, 300]);
        }
        assert_eq!(child(&bytes, u32::MAX, 1), None);
    }
}
