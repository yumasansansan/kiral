// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later
//
// This file is part of KIRAL. The SPDX lines name its copyright holder and its
// license, in the machine-readable form of the REUSE specification: the GNU
// General Public License, version 3 or any later version
// (LICENSES/GPL-3.0-or-later.txt).

//! The processors of the machine: which are performance cores and which
//! efficiency cores, the clock one runs at, and the processor's name, for the
//! measurements to say what they ran on.
//!
//! Linux tells the kinds apart on Intel's hybrid processors, in
//! `/sys/devices/cpu_core/cpus` and `/sys/devices/cpu_atom/cpus`; where it does
//! not, every processor online counts as a performance core, and so does every
//! processor on the other systems, which are not read yet.

/// The processors of the machine, by kind, as the system numbers them.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Topology {
    /// The performance cores' processors (Intel's `cpu_core`), in order.
    pub performance: Vec<usize>,
    /// The efficiency cores' processors (Intel's `cpu_atom`), in order.
    pub efficiency: Vec<usize>,
}

/// The processors of a list as Linux writes them: `0-7`, `0-3,8-11`, `5`;
/// none where it is not one.
#[must_use]
pub fn parse_list(text: &str) -> Option<Vec<usize>> {
    let mut processors = Vec::new();
    for range in text.trim().split(',').filter(|range| !range.is_empty()) {
        let (first, last) = range.split_once('-').unwrap_or((range, range));
        let (first, last) = (first.parse::<usize>().ok()?, last.parse::<usize>().ok()?);
        if last < first || last - first > 1 << 16 {
            return None;
        }
        processors.extend(first..=last);
    }
    Some(processors)
}

impl Topology {
    /// The processors of the machine this runs on.
    #[must_use]
    pub fn detect() -> Self {
        let list = |path: &str| {
            std::fs::read_to_string(path)
                .ok()
                .and_then(|text| parse_list(&text))
                .filter(|list| !list.is_empty())
        };
        if let Some(performance) = list("/sys/devices/cpu_core/cpus") {
            return Self {
                performance,
                efficiency: list("/sys/devices/cpu_atom/cpus").unwrap_or_default(),
            };
        }
        let online = list("/sys/devices/system/cpu/online").unwrap_or_else(|| {
            let count = std::thread::available_parallelism().map_or(1, usize::from);
            (0..count).collect()
        });
        Self {
            performance: online,
            efficiency: Vec::new(),
        }
    }

    /// The processor of a name: `p<n>`, the n-th performance core's, `e<n>` the
    /// n-th efficiency core's, or a processor's number as the system gives it.
    #[must_use]
    pub fn processor(&self, name: &str) -> Option<usize> {
        let nth = |kind: &[usize], rest: &str| rest.parse::<usize>().ok().and_then(|n| kind.get(n).copied());
        if let Some(rest) = name.strip_prefix('p') {
            nth(&self.performance, rest)
        } else if let Some(rest) = name.strip_prefix('e') {
            nth(&self.efficiency, rest)
        } else {
            name.parse().ok()
        }
    }

    /// The kind of a processor: `P` for a performance core, `E` for an
    /// efficiency core, none for one the machine does not have.
    #[must_use]
    pub fn kind(&self, processor: usize) -> Option<char> {
        if self.performance.contains(&processor) {
            Some('P')
        } else if self.efficiency.contains(&processor) {
            Some('E')
        } else {
            None
        }
    }
}

/// The clock a processor runs at now, in kHz, as Linux's cpufreq reports it;
/// none where it does not.
#[must_use]
pub fn clock_khz(processor: usize) -> Option<u64> {
    std::fs::read_to_string(format!(
        "/sys/devices/system/cpu/cpu{processor}/cpufreq/scaling_cur_freq"
    ))
    .ok()?
    .trim()
    .parse()
    .ok()
}

/// The processor's name, as Linux's `/proc/cpuinfo` gives it; none elsewhere.
#[must_use]
pub fn processor_name() -> Option<String> {
    let info = std::fs::read_to_string("/proc/cpuinfo").ok()?;
    info.lines().find_map(|line| {
        let (key, value) = line.split_once(':')?;
        (key.trim() == "model name").then(|| value.trim().to_owned())
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lists_are_read_as_linux_writes_them() {
        assert_eq!(parse_list("0-7\n"), Some((0..8).collect()));
        assert_eq!(parse_list("0-3,8-11"), Some(vec![0, 1, 2, 3, 8, 9, 10, 11]));
        assert_eq!(parse_list("5"), Some(vec![5]));
        assert_eq!(parse_list(""), Some(Vec::new()));
        assert_eq!(parse_list("3-1"), None);
        assert_eq!(parse_list("a"), None);
        assert_eq!(parse_list("0-99999999"), None);
    }

    #[test]
    fn processors_are_named_by_kind_or_number() {
        let topology = Topology {
            performance: (0..8).collect(),
            efficiency: (8..16).collect(),
        };
        assert_eq!(topology.processor("p0"), Some(0));
        assert_eq!(topology.processor("p7"), Some(7));
        assert_eq!(topology.processor("e0"), Some(8));
        assert_eq!(topology.processor("e8"), None);
        assert_eq!(topology.processor("12"), Some(12));
        assert_eq!(topology.processor("x"), None);
        assert_eq!(topology.kind(3), Some('P'));
        assert_eq!(topology.kind(15), Some('E'));
        assert_eq!(topology.kind(16), None);
    }

    #[test]
    fn the_machine_has_processors() {
        let topology = Topology::detect();
        println!("{topology:?}, {:?}", processor_name());
        assert_ne!(topology.performance, []);
    }
}
