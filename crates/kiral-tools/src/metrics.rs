// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later
//
// This file is part of KIRAL. The SPDX lines name its copyright holder and its
// license, in the machine-readable form of the REUSE specification: the GNU
// General Public License, version 3 or any later version
// (LICENSES/GPL-3.0-or-later.txt).

//! The measures of conversions, as AJIMEE-Bench's own code takes them: Acc@1,
//! whether the conversion is one of those an item accepts; the character error
//! rate, the edit distance to the nearest of them over its length in
//! characters. And McNemar's exact test, of whether two runs differ on the
//! items more than by chance.

/// The edit distance of two texts in characters: the fewest insertions,
/// deletions and substitutions of one character that make one the other.
pub fn edit_distance(a: &str, b: &str) -> usize {
    let b: Vec<char> = b.chars().collect();
    let mut previous: Vec<usize> = (0..=b.len()).collect();
    let mut current = vec![0; b.len() + 1];
    for (i, a) in a.chars().enumerate() {
        current[0] = i + 1;
        for (j, &b) in b.iter().enumerate() {
            let substitution = previous[j] + usize::from(a != b);
            current[j + 1] = substitution.min(previous[j + 1] + 1).min(current[j] + 1);
        }
        std::mem::swap(&mut previous, &mut current);
    }
    previous[b.len()]
}

/// A count as a double, exactly: every count here is far below 2^53.
pub fn count(value: usize) -> f64 {
    f64::from(u32::try_from(value).expect("a count below 2^32"))
}

/// The character error rate of a conversion against an accepted one: the edit
/// distance over the accepted one's length; infinite for an empty one, as
/// AJIMEE-Bench's code has it.
pub fn character_error_rate(accepted: &str, conversion: &str) -> f64 {
    let length = accepted.chars().count();
    if length == 0 {
        return f64::INFINITY;
    }
    count(edit_distance(accepted, conversion)) / count(length)
}

/// The least character error rate against the accepted conversions.
pub fn least_error_rate(accepted: &[String], conversion: &str) -> f64 {
    accepted
        .iter()
        .map(|accepted| character_error_rate(accepted, conversion))
        .fold(f64::INFINITY, f64::min)
}

/// The two-sided p of McNemar's exact test: of the items on which two runs
/// differ, `better` went one way and `worse` the other; the probability, were
/// each way as likely, of a split at least as uneven.
pub fn mcnemar(better: usize, worse: usize) -> f64 {
    let differing = better + worse;
    let fewer = better.min(worse);
    // The binomial probabilities of 0, 1, ... successes of `differing` fair
    // trials, one from the one before.
    let mut probability = 0.5f64.powi(i32::try_from(differing).expect("a count of items"));
    let mut tail = 0.0;
    for successes in 0..=fewer {
        tail += probability;
        probability *= count(differing - successes) / count(successes + 1);
    }
    (2.0 * tail).min(1.0)
}

/// The `p`-th percentile of sorted times, by the nearest rank: the time no
/// fewer than `p` percent of them are at or below, `ceil(p n / 100)`-th of `n`;
/// zero for none.
pub fn percentile(sorted: &[std::time::Duration], p: usize) -> std::time::Duration {
    if sorted.is_empty() {
        return std::time::Duration::ZERO;
    }
    let rank = (p * sorted.len()).div_ceil(100).clamp(1, sorted.len());
    sorted[rank - 1]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn percentiles_are_of_the_nearest_rank() {
        let times: Vec<std::time::Duration> = (1..=200).map(std::time::Duration::from_millis).collect();
        assert_eq!(percentile(&times, 50), std::time::Duration::from_millis(100));
        assert_eq!(percentile(&times, 99), std::time::Duration::from_millis(198));
        assert_eq!(percentile(&times, 100), std::time::Duration::from_millis(200));
        assert_eq!(percentile(&times[..1], 99), std::time::Duration::from_millis(1));
        assert_eq!(percentile(&[], 50), std::time::Duration::ZERO);
    }

    #[test]
    fn distances_are_those_of_ajimees_tests() {
        assert_eq!(character_error_rate("hello", "hello"), 0.0);
        assert_eq!(character_error_rate("hello", "hell"), 0.2);
        assert_eq!(character_error_rate("hello", "hellao"), 0.2);
        assert_eq!(character_error_rate("hello", "hallo"), 0.2);
        assert_eq!(
            least_error_rate(&["hello".to_owned(), "hallo".to_owned()], "hello"),
            0.0
        );
        assert_eq!(
            least_error_rate(&["hallo".to_owned(), "hollo".to_owned()], "hello"),
            0.2
        );
        assert_eq!(character_error_rate("", "a"), f64::INFINITY);
        assert_eq!(edit_distance("家計を助ける", "家計を扶ける為"), 2);
        assert_eq!(edit_distance("", "かな"), 2);
    }

    #[test]
    fn mcnemar_is_the_binomial_tail() {
        assert_eq!(mcnemar(0, 0), 1.0);
        assert_eq!(mcnemar(3, 3), 1.0);
        // 0 of 3: 2 (1/8) = 0.25; 0 of 8: 2 / 256.
        assert!((mcnemar(0, 3) - 0.25).abs() < 1e-15);
        assert!((mcnemar(8, 0) - 2.0 / 256.0).abs() < 1e-15);
        // 2 of 10: 2 (1 + 10 + 45) / 1024.
        assert!((mcnemar(2, 8) - 112.0 / 1024.0).abs() < 1e-15);
        assert_eq!(mcnemar(100, 100), 1.0);
        assert!(mcnemar(0, 200) > 0.0);
    }
}
