// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later
//
// This file is part of KIRAL. The SPDX lines name its copyright holder and its
// license, in the machine-readable form of the REUSE specification: the GNU
// General Public License, version 3 or any later version
// (LICENSES/GPL-3.0-or-later.txt).

//! The processor KIRAL runs on: the extensions of its instruction set that the
//! kernels are written for, and the level of the kernels chosen from them.
//!
//! A [`Level`] is a set of extensions that one form of the kernels is written
//! for: `v2`, `avx2`, `avx-vnni` and `avx512-vnni` on x86-64, `neon`, `dotprod`
//! and `i8mm` on AArch64. A processor has some of them, not always all the less
//! preferred ones -- Ice Lake's server cores have AVX512-VNNI without the VEX form
//! of VNNI, Alder Lake's the VEX form without AVX-512 -- so the levels it has are a
//! set, and the level in effect is the most preferred of them. `KIRAL_ISA=<level>`
//! holds it to that level or a less preferred one, and `KIRAL_KERNELS=portable`
//! takes the portable form of every kernel, whatever the level ([`Request`]).
//!
//! The extensions are those std's `is_x86_feature_detected!` and
//! `is_aarch64_feature_detected!` report, which ask the processor and the system:
//! AVX-512 counts only where the system saves its registers.
//!
//! [`topology`] tells the performance cores from the efficiency cores, and reads
//! the clock and the processor's name, for measurements to say what they ran on.
#![forbid(unsafe_code)]

use std::fmt;

pub mod topology;

/// A set of extensions of the instruction set that a form of the kernels is
/// written for, named as `KIRAL_ISA` names it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Level {
    /// `v2`: x86-64-v2, SSE4.2 and POPCNT, the baseline the build is compiled for.
    /// The portable form alone.
    V2,
    /// `avx2`: AVX2 and FMA. The integer products by `vpmaddubsw` and `vpmaddwd`,
    /// exact for inputs of seven bits.
    Avx2,
    /// `avx-vnni`: AVX2, FMA and the VEX form of `vpdpbusd`: Alder Lake, Raptor
    /// Lake, Meteor Lake and after; Zen 5.
    AvxVnni,
    /// `avx512-vnni`: AVX-512 F, BW, VL and VNNI, 16 lanes and masked edges: Ice
    /// Lake's server cores and after, Zen 4 and 5.
    Avx512Vnni,
    /// `neon`: AArch64's baseline.
    Neon,
    /// `dotprod`: NEON and `sdot`/`udot`: Apple M1 and after, most Arm cores since
    /// 2018.
    Dotprod,
    /// `i8mm`: NEON, `sdot`/`udot` and `smmla`/`usmmla`: Apple M2 and after.
    I8mm,
}

/// The levels of the architecture KIRAL is built for, the least preferred first.
#[cfg(target_arch = "x86_64")]
pub const LEVELS: &[Level] = &[Level::V2, Level::Avx2, Level::AvxVnni, Level::Avx512Vnni];
/// The levels of the architecture KIRAL is built for, the least preferred first.
#[cfg(target_arch = "aarch64")]
pub const LEVELS: &[Level] = &[Level::Neon, Level::Dotprod, Level::I8mm];
/// The levels of the architecture KIRAL is built for: none, on an architecture
/// whose kernels are the portable form alone.
#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
pub const LEVELS: &[Level] = &[];

/// Every level, of every architecture.
const ALL: [Level; 7] = [
    Level::V2,
    Level::Avx2,
    Level::AvxVnni,
    Level::Avx512Vnni,
    Level::Neon,
    Level::Dotprod,
    Level::I8mm,
];

impl Level {
    /// The level's name, as `KIRAL_ISA` takes it.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::V2 => "v2",
            Self::Avx2 => "avx2",
            Self::AvxVnni => "avx-vnni",
            Self::Avx512Vnni => "avx512-vnni",
            Self::Neon => "neon",
            Self::Dotprod => "dotprod",
            Self::I8mm => "i8mm",
        }
    }

    /// The level of a name, of any architecture.
    #[must_use]
    pub fn from_name(name: &str) -> Option<Self> {
        ALL.into_iter().find(|level| level.name() == name)
    }

    /// Whether the level is one of the architecture KIRAL is built for.
    #[must_use]
    pub fn is_native(self) -> bool {
        LEVELS.contains(&self)
    }
}

impl fmt::Display for Level {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.name())
    }
}

/// The extensions that the levels need, as the processor and the system report
/// them. The fields of the other architecture stay false.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[expect(
    clippy::struct_excessive_bools,
    reason = "one flag for each extension that the processor reports, as std's macros report them"
)]
pub struct Features {
    /// SSE4.2.
    pub sse4_2: bool,
    /// POPCNT.
    pub popcnt: bool,
    /// AVX2.
    pub avx2: bool,
    /// FMA, the fused multiply-adds of three operands.
    pub fma: bool,
    /// AVX-VNNI, the VEX form of `vpdpbusd`.
    pub avx_vnni: bool,
    /// AVX-512 F, with the system saving its registers.
    pub avx512f: bool,
    /// AVX-512 BW.
    pub avx512bw: bool,
    /// AVX-512 VL.
    pub avx512vl: bool,
    /// AVX-512 VNNI.
    pub avx512_vnni: bool,
    /// NEON (Advanced SIMD).
    pub neon: bool,
    /// `FEAT_DotProd`: `sdot` and `udot`.
    pub dotprod: bool,
    /// `FEAT_I8MM`: `smmla`, `ummla` and `usmmla`.
    pub i8mm: bool,
}

impl Features {
    /// The extensions of the processor this runs on.
    #[must_use]
    pub fn detect() -> Self {
        #[cfg(target_arch = "x86_64")]
        {
            use std::arch::is_x86_feature_detected;
            Self {
                sse4_2: is_x86_feature_detected!("sse4.2"),
                popcnt: is_x86_feature_detected!("popcnt"),
                avx2: is_x86_feature_detected!("avx2"),
                fma: is_x86_feature_detected!("fma"),
                avx_vnni: is_x86_feature_detected!("avxvnni"),
                avx512f: is_x86_feature_detected!("avx512f"),
                avx512bw: is_x86_feature_detected!("avx512bw"),
                avx512vl: is_x86_feature_detected!("avx512vl"),
                avx512_vnni: is_x86_feature_detected!("avx512vnni"),
                ..Self::default()
            }
        }
        #[cfg(target_arch = "aarch64")]
        {
            use std::arch::is_aarch64_feature_detected;
            Self {
                neon: is_aarch64_feature_detected!("neon"),
                dotprod: is_aarch64_feature_detected!("dotprod"),
                i8mm: is_aarch64_feature_detected!("i8mm"),
                ..Self::default()
            }
        }
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            Self::default()
        }
    }

    /// Whether the extensions include every one that the level needs.
    #[must_use]
    pub const fn has(self, level: Level) -> bool {
        let avx2 = self.avx2 && self.fma;
        match level {
            Level::V2 => self.sse4_2 && self.popcnt,
            Level::Avx2 => avx2,
            Level::AvxVnni => avx2 && self.avx_vnni,
            Level::Avx512Vnni => avx2 && self.avx512f && self.avx512bw && self.avx512vl && self.avx512_vnni,
            Level::Neon => self.neon,
            Level::Dotprod => self.neon && self.dotprod,
            Level::I8mm => self.neon && self.dotprod && self.i8mm,
        }
    }

    /// The levels of the architecture that the extensions include, the least
    /// preferred first: those the tests run every kernel at.
    #[must_use]
    pub fn levels(self) -> Vec<Level> {
        LEVELS.iter().copied().filter(|&level| self.has(level)).collect()
    }
}

/// What the environment asks of the kernels: a limit to their level
/// (`KIRAL_ISA`), and the portable form everywhere (`KIRAL_KERNELS=portable`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Request {
    /// The most preferred level that may be taken; any, where there is none.
    pub limit: Option<Level>,
    /// Whether every kernel is taken in its portable form.
    pub portable: bool,
}

/// A request that the environment makes and that cannot be met as it is.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    /// `KIRAL_ISA` names no level.
    UnknownLevel(String),
    /// `KIRAL_ISA` names a level of another architecture.
    OtherArchitecture(Level),
    /// `KIRAL_KERNELS` names something other than `portable`.
    UnknownKernels(String),
}

impl fmt::Display for Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let names = LEVELS.iter().map(|level| level.name()).collect::<Vec<_>>().join(", ");
        match self {
            Self::UnknownLevel(name) => write!(formatter, "KIRAL_ISA={name} names no level; the levels are {names}"),
            Self::OtherArchitecture(level) => write!(
                formatter,
                "KIRAL_ISA={level} is a level of another architecture; the levels of this one are {names}"
            ),
            Self::UnknownKernels(name) => write!(formatter, "KIRAL_KERNELS={name} is not portable, the one it takes"),
        }
    }
}

impl std::error::Error for Error {}

impl Request {
    /// The request of `KIRAL_ISA` and `KIRAL_KERNELS`, of which an empty value
    /// is the same as none.
    ///
    /// # Errors
    ///
    /// Where `KIRAL_ISA` names no level of this architecture, or `KIRAL_KERNELS`
    /// is set to anything but `portable`.
    pub fn from_environment() -> Result<Self, Error> {
        let isa = std::env::var("KIRAL_ISA").ok();
        let kernels = std::env::var("KIRAL_KERNELS").ok();
        Self::parse(isa.as_deref(), kernels.as_deref())
    }

    /// The request of the values of `KIRAL_ISA` and `KIRAL_KERNELS`.
    ///
    /// # Errors
    ///
    /// As [`Request::from_environment`].
    pub fn parse(isa: Option<&str>, kernels: Option<&str>) -> Result<Self, Error> {
        let limit = match isa.map(str::trim).filter(|name| !name.is_empty()) {
            None => None,
            Some(name) => {
                let level = Level::from_name(name).ok_or_else(|| Error::UnknownLevel(name.to_owned()))?;
                if !level.is_native() {
                    return Err(Error::OtherArchitecture(level));
                }
                Some(level)
            }
        };
        let portable = match kernels.map(str::trim).filter(|name| !name.is_empty()) {
            None => false,
            Some("portable") => true,
            Some(name) => return Err(Error::UnknownKernels(name.to_owned())),
        };
        Ok(Self { limit, portable })
    }
}

/// The form of the kernels in effect: the level they are taken at, and whether
/// the portable form is taken whatever the level.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Choice {
    /// The most preferred level that the processor has and the request allows;
    /// none on an architecture without levels.
    pub level: Option<Level>,
    /// Whether every kernel is taken in its portable form.
    pub portable: bool,
}

impl Choice {
    /// The choice for a processor's extensions and a request.
    #[must_use]
    pub fn new(features: Features, request: Request) -> Self {
        let level = LEVELS
            .iter()
            .copied()
            .rfind(|&level| features.has(level) && request.limit.is_none_or(|limit| level <= limit));
        Self {
            level,
            portable: request.portable,
        }
    }
}

impl fmt::Display for Choice {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match (self.level, self.portable) {
            (Some(level), false) => write!(formatter, "{level}"),
            (Some(level), true) => write!(formatter, "portable (level {level})"),
            (None, _) => formatter.write_str("portable"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_go_back_to_their_levels() {
        for level in ALL {
            assert_eq!(Level::from_name(level.name()), Some(level));
        }
        assert_eq!(Level::from_name("avx512"), None);
        assert_eq!(Level::from_name("AVX2"), None);
    }

    #[test]
    fn the_architecture_has_its_own_levels_in_order() {
        assert!(LEVELS.windows(2).all(|pair| pair[0] < pair[1]));
        for level in LEVELS {
            assert!(level.is_native());
        }
    }

    #[test]
    fn requests_are_read_from_their_values() {
        assert_eq!(Request::parse(None, None), Ok(Request::default()));
        assert_eq!(Request::parse(Some(""), Some(" ")), Ok(Request::default()));
        assert_eq!(
            Request::parse(None, Some("portable")),
            Ok(Request {
                limit: None,
                portable: true
            })
        );
        assert_eq!(
            Request::parse(Some("avx2x"), None),
            Err(Error::UnknownLevel("avx2x".to_owned()))
        );
        assert_eq!(
            Request::parse(None, Some("asm")),
            Err(Error::UnknownKernels("asm".to_owned()))
        );
        if let Some(&least) = LEVELS.first() {
            assert_eq!(
                Request::parse(Some(least.name()), None),
                Ok(Request {
                    limit: Some(least),
                    portable: false
                })
            );
        }
        let foreign = ALL.into_iter().find(|level| !level.is_native());
        if let Some(foreign) = foreign {
            assert_eq!(
                Request::parse(Some(foreign.name()), None),
                Err(Error::OtherArchitecture(foreign))
            );
        }
    }

    /// A processor of x86-64-v2 and the extensions named: `avx2` (with FMA),
    /// `avxvnni`, `avx512` (F, BW and VL) and `avx512vnni`.
    #[cfg(target_arch = "x86_64")]
    fn x86(extensions: &[&str]) -> Features {
        let has = |name| extensions.contains(&name);
        Features {
            sse4_2: true,
            popcnt: true,
            avx2: has("avx2"),
            fma: has("avx2"),
            avx_vnni: has("avxvnni"),
            avx512f: has("avx512"),
            avx512bw: has("avx512"),
            avx512vl: has("avx512"),
            avx512_vnni: has("avx512vnni"),
            ..Features::default()
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn the_processors_have_the_levels_of_their_extensions() {
        use Level::{Avx2, Avx512Vnni, AvxVnni, V2};
        let nehalem = x86(&[]);
        let haswell = x86(&["avx2"]);
        let skylake_server = x86(&["avx2", "avx512"]);
        let ice_lake_server = x86(&["avx2", "avx512", "avx512vnni"]);
        let alder_lake = x86(&["avx2", "avxvnni"]);
        let sapphire_rapids = x86(&["avx2", "avxvnni", "avx512", "avx512vnni"]);
        assert_eq!(nehalem.levels(), [V2]);
        assert_eq!(haswell.levels(), [V2, Avx2]);
        assert_eq!(skylake_server.levels(), [V2, Avx2]);
        assert_eq!(ice_lake_server.levels(), [V2, Avx2, Avx512Vnni]);
        assert_eq!(alder_lake.levels(), [V2, Avx2, AvxVnni]);
        assert_eq!(sapphire_rapids.levels(), [V2, Avx2, AvxVnni, Avx512Vnni]);
        // AVX-VNNI without FMA does not make the level: it is AVX2's and FMA's too.
        let without_fma = Features {
            fma: false,
            ..alder_lake
        };
        assert_eq!(without_fma.levels(), [V2]);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn a_limit_takes_the_most_preferred_level_within_it() {
        use Level::{Avx2, Avx512Vnni, AvxVnni, V2};
        let ice_lake_server = x86(&["avx2", "avx512", "avx512vnni"]);
        let sapphire_rapids = x86(&["avx2", "avxvnni", "avx512", "avx512vnni"]);
        let choose = |features, limit| Choice::new(features, Request { limit, portable: false }).level;
        assert_eq!(choose(sapphire_rapids, None), Some(Avx512Vnni));
        assert_eq!(choose(sapphire_rapids, Some(AvxVnni)), Some(AvxVnni));
        assert_eq!(choose(ice_lake_server, None), Some(Avx512Vnni));
        // Ice Lake has no VEX form of VNNI: held to it, it takes AVX2.
        assert_eq!(choose(ice_lake_server, Some(AvxVnni)), Some(Avx2));
        assert_eq!(choose(ice_lake_server, Some(V2)), Some(V2));
        let portable = Choice::new(
            sapphire_rapids,
            Request {
                limit: Some(Avx2),
                portable: true,
            },
        );
        assert_eq!(portable.to_string(), "portable (level avx2)");
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn the_processors_have_the_levels_of_their_extensions() {
        use Level::{Dotprod, I8mm, Neon};
        let cortex_a72 = Features {
            neon: true,
            ..Features::default()
        };
        let apple_m1 = Features {
            dotprod: true,
            ..cortex_a72
        };
        let apple_m2 = Features { i8mm: true, ..apple_m1 };
        assert_eq!(cortex_a72.levels(), [Neon]);
        assert_eq!(apple_m1.levels(), [Neon, Dotprod]);
        assert_eq!(apple_m2.levels(), [Neon, Dotprod, I8mm]);
        let held = Choice::new(
            apple_m2,
            Request {
                limit: Some(Dotprod),
                portable: false,
            },
        );
        assert_eq!(held.level, Some(Dotprod));
    }
}
