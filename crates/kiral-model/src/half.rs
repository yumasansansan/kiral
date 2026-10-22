// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later
//
// This file is part of KIRAL. The SPDX lines name its copyright holder and its
// license, in the machine-readable form of the REUSE specification: the GNU
// General Public License, version 3 or any later version
// (LICENSES/GPL-3.0-or-later.txt).

//! Floats of other widths, to single precision: bfloat16 and IEEE half
//! precision exactly, every value of theirs being one of single precision, and
//! double precision to the nearest.

/// A bfloat16 as single precision: its bits are the upper half of the float's.
#[must_use]
pub fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits(u32::from(bits) << 16)
}

/// A float of IEEE half precision as single precision, NaNs keeping their
/// payloads.
#[must_use]
pub fn f16_to_f32(bits: u16) -> f32 {
    let sign = u32::from(bits & 0x8000) << 16;
    let exponent = u32::from((bits >> 10) & 0x1f);
    let fraction = u32::from(bits & 0x03ff);
    let magnitude = match (exponent, fraction) {
        (0, 0) => 0,
        // A subnormal, fraction 2^-24: normalized, its highest bit p becomes the
        // implicit one and its exponent p - 24.
        (0, _) => {
            let p = fraction.ilog2();
            ((p + 103) << 23) | ((fraction << (23 - p)) & 0x007f_ffff)
        }
        (0x1f, _) => 0x7f80_0000 | (fraction << 13),
        _ => ((exponent + 112) << 23) | (fraction << 13),
    };
    f32::from_bits(sign | magnitude)
}

/// A double rounded to the nearest float of single precision, ties to even.
#[must_use]
pub fn f64_to_f32(value: f64) -> f32 {
    #[expect(clippy::as_conversions, reason = "the conversion that rounds to nearest")]
    #[expect(clippy::cast_possible_truncation, reason = "the conversion that rounds to nearest")]
    let rounded = value as f32;
    rounded
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_half_converts_to_its_value() {
        for bits in 0..=u16::MAX {
            let sign = if bits & 0x8000 == 0 { 1.0 } else { -1.0 };
            let exponent = i32::from((bits >> 10) & 0x1f);
            let fraction = f64::from(bits & 0x03ff);
            let converted = f16_to_f32(bits);
            match exponent {
                0x1f if fraction == 0.0 => assert_eq!(f64::from(converted), sign * f64::INFINITY),
                0x1f => assert!(converted.is_nan(), "{bits:#06x}"),
                0 => assert_eq!(f64::from(converted), sign * fraction * 2f64.powi(-24), "{bits:#06x}"),
                _ => assert_eq!(
                    f64::from(converted),
                    sign * (1.0 + fraction / 1024.0) * 2f64.powi(exponent - 15),
                    "{bits:#06x}"
                ),
            }
            assert_eq!(converted.is_sign_negative(), bits & 0x8000 != 0);
        }
    }

    #[test]
    fn bfloat16_is_the_upper_half() {
        assert_eq!(bf16_to_f32(0x3f80), 1.0);
        assert_eq!(bf16_to_f32(0xc040), -3.0);
        assert_eq!(bf16_to_f32(0x0001), f32::from_bits(0x0001_0000));
        assert!(bf16_to_f32(0x7fc0).is_nan());
    }

    #[test]
    fn doubles_round_to_nearest() {
        assert_eq!(f64_to_f32(0.1), 0.1f32);
        assert_eq!(f64_to_f32(1.0 + f64::EPSILON), 1.0);
        assert_eq!(f64_to_f32(1e300), f32::INFINITY);
    }
}
