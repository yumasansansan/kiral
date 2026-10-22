// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later
//
// This file is part of KIRAL. The SPDX lines name its copyright holder and its
// license, in the machine-readable form of the REUSE specification: the GNU
// General Public License, version 3 or any later version
// (LICENSES/GPL-3.0-or-later.txt).

//! SHA-256, as FIPS 180-4 gives it: the digest a model file carries of itself.
//!
//! The message is taken a block of 64 bytes at a time, each block's sixteen
//! words big-endian, expanded to the 64 of its schedule and mixed into the
//! eight words of the state in 64 rounds; the last block is padded with a byte
//! 0x80, zeros, and the message's length in bits, in a block or two.

/// The digest of bytes.
#[must_use]
pub fn digest(bytes: &[u8]) -> [u8; 32] {
    let mut state = INITIAL;
    let (blocks, rest) = bytes.as_chunks::<64>();
    for block in blocks {
        compress(&mut state, block);
    }
    // The bytes left, 0x80, zeros, and the length in bits: one block where the
    // eight bytes of the length fit after the 0x80, two where they do not.
    let mut last = [0u8; 128];
    last[..rest.len()].copy_from_slice(rest);
    last[rest.len()] = 0x80;
    let end = if rest.len() < 56 { 64 } else { 128 };
    last[end - 8..end].copy_from_slice(&length_in_bits(bytes.len()).to_be_bytes());
    for block in last[..end].as_chunks::<64>().0 {
        compress(&mut state, block);
    }
    let mut digest = [0u8; 32];
    for (bytes, word) in digest.as_chunks_mut::<4>().0.iter_mut().zip(state) {
        *bytes = word.to_be_bytes();
    }
    digest
}

/// A digest in lowercase hexadecimal, as `sha256sum` writes it.
#[must_use]
pub fn hex(digest: &[u8; 32]) -> String {
    const DIGITS: &[u8; 16] = b"0123456789abcdef";
    let mut text = String::with_capacity(64);
    for byte in digest {
        text.push(char::from(DIGITS[usize::from(byte >> 4)]));
        text.push(char::from(DIGITS[usize::from(byte & 0xf)]));
    }
    text
}

/// A message's length in bits, modulo 2^64, as the padding gives it.
#[expect(clippy::as_conversions, reason = "a usize is of 64 bits at most on every target")]
fn length_in_bits(length: usize) -> u64 {
    (length as u64).wrapping_mul(8)
}

/// The first state: the first 32 bits of the fractional parts of the square
/// roots of the first eight primes.
const INITIAL: [u32; 8] = [
    0x6a09_e667,
    0xbb67_ae85,
    0x3c6e_f372,
    0xa54f_f53a,
    0x510e_527f,
    0x9b05_688c,
    0x1f83_d9ab,
    0x5be0_cd19,
];

/// The rounds' constants: the first 32 bits of the fractional parts of the
/// cube roots of the first 64 primes.
const ROUNDS: [u32; 64] = [
    0x428a_2f98,
    0x7137_4491,
    0xb5c0_fbcf,
    0xe9b5_dba5,
    0x3956_c25b,
    0x59f1_11f1,
    0x923f_82a4,
    0xab1c_5ed5,
    0xd807_aa98,
    0x1283_5b01,
    0x2431_85be,
    0x550c_7dc3,
    0x72be_5d74,
    0x80de_b1fe,
    0x9bdc_06a7,
    0xc19b_f174,
    0xe49b_69c1,
    0xefbe_4786,
    0x0fc1_9dc6,
    0x240c_a1cc,
    0x2de9_2c6f,
    0x4a74_84aa,
    0x5cb0_a9dc,
    0x76f9_88da,
    0x983e_5152,
    0xa831_c66d,
    0xb003_27c8,
    0xbf59_7fc7,
    0xc6e0_0bf3,
    0xd5a7_9147,
    0x06ca_6351,
    0x1429_2967,
    0x27b7_0a85,
    0x2e1b_2138,
    0x4d2c_6dfc,
    0x5338_0d13,
    0x650a_7354,
    0x766a_0abb,
    0x81c2_c92e,
    0x9272_2c85,
    0xa2bf_e8a1,
    0xa81a_664b,
    0xc24b_8b70,
    0xc76c_51a3,
    0xd192_e819,
    0xd699_0624,
    0xf40e_3585,
    0x106a_a070,
    0x19a4_c116,
    0x1e37_6c08,
    0x2748_774c,
    0x34b0_bcb5,
    0x391c_0cb3,
    0x4ed8_aa4a,
    0x5b9c_ca4f,
    0x682e_6ff3,
    0x748f_82ee,
    0x78a5_636f,
    0x84c8_7814,
    0x8cc7_0208,
    0x90be_fffa,
    0xa450_6ceb,
    0xbef9_a3f7,
    0xc671_78f2,
];

/// Mixes a block into the state.
#[expect(
    clippy::many_single_char_names,
    reason = "the standard's names of the eight working variables"
)]
fn compress(state: &mut [u32; 8], block: &[u8; 64]) {
    let mut schedule = [0u32; 64];
    for (word, bytes) in schedule.iter_mut().zip(block.as_chunks::<4>().0) {
        *word = u32::from_be_bytes(*bytes);
    }
    for t in 16..64 {
        let (early, late) = (schedule[t - 15], schedule[t - 2]);
        let sigma0 = early.rotate_right(7) ^ early.rotate_right(18) ^ (early >> 3);
        let sigma1 = late.rotate_right(17) ^ late.rotate_right(19) ^ (late >> 10);
        schedule[t] = schedule[t - 16]
            .wrapping_add(sigma0)
            .wrapping_add(schedule[t - 7])
            .wrapping_add(sigma1);
    }
    let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h] = *state;
    for (&constant, &word) in ROUNDS.iter().zip(&schedule) {
        let sum1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
        let choice = (e & f) ^ (!e & g);
        let first = h
            .wrapping_add(sum1)
            .wrapping_add(choice)
            .wrapping_add(constant)
            .wrapping_add(word);
        let sum0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
        let majority = (a & b) ^ (a & c) ^ (b & c);
        let second = sum0.wrapping_add(majority);
        (h, g, f, e) = (g, f, e, d.wrapping_add(first));
        (d, c, b, a) = (c, b, a, first.wrapping_add(second));
    }
    for (word, mixed) in state.iter_mut().zip([a, b, c, d, e, f, g, h]) {
        *word = word.wrapping_add(mixed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_standards_examples() {
        // FIPS 180-4's examples and NIST's long messages.
        let examples: [(&[u8], &str); 4] = [
            (b"", "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"),
            (b"abc", "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"),
            (
                b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq",
                "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1",
            ),
            (
                b"abcdefghbcdefghicdefghijdefghijkefghijklfghijklmghijklmnhijklmnoijklmnopjklmnopqklmnopqrlmnopqrsmnopqrstnopqrstu",
                "cf5b16a778af8380036ce59e7b0492370b249b11e8f07a51afac45037afee9d1",
            ),
        ];
        for (message, expected) in examples {
            assert_eq!(hex(&digest(message)), expected, "{}", String::from_utf8_lossy(message));
        }
        assert_eq!(
            hex(&digest(&vec![b'a'; 1_000_000])),
            "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0"
        );
    }

    #[test]
    fn lengths_about_the_blocks_ends() {
        // The bytes i mod 251, of the lengths at which the padding takes one
        // block or two, against Python's hashlib.
        let expected = [
            (55, "463eb28e72f82e0a96c0a4cc53690c571281131f672aa229e0d45ae59b598b59"),
            (56, "da2ae4d6b36748f2a318f23e7ab1dfdf45acdc9d049bd80e59de82a60895f562"),
            (63, "29af2686fd53374a36b0846694cc342177e428d1647515f078784d69cdb9e488"),
            (64, "fdeab9acf3710362bd2658cdc9a29e8f9c757fcf9811603a8c447cd1d9151108"),
            (65, "4bfd2c8b6f1eec7a2afeb48b934ee4b2694182027e6d0fc075074f2fabb31781"),
            (119, "da18797ed7c3a777f0847f429724a2d8cd5138e6ed2895c3fa1a6d39d18f7ec6"),
            (120, "f52b23db1fbb6ded89ef42a23ce0c8922c45f25c50b568a93bf1c075420bbb7c"),
            (127, "92ca0fa6651ee2f97b884b7246a562fa71250fedefe5ebf270d31c546bfea976"),
            (128, "471fb943aa23c511f6f72f8d1652d9c880cfa392ad80503120547703e56a2be5"),
            (1000, "4e4c294b331f7a2099a379bec34b9f9fc03dc46ab465d998f4d683da53487e6d"),
        ];
        for (length, expected) in expected {
            let message: Vec<u8> = (0..length)
                .map(|i: usize| u8::try_from(i % 251).expect("a byte"))
                .collect();
            assert_eq!(hex(&digest(&message)), expected, "{length} bytes");
        }
    }
}
