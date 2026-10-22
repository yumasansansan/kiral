// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later
//
// This file is part of KIRAL. The SPDX lines name its copyright holder and its
// license, in the machine-readable form of the REUSE specification: the GNU
// General Public License, version 3 or any later version
// (LICENSES/GPL-3.0-or-later.txt).

//! KIRAL's networks.
//!
//! - [`gpt2`]: GPT-2 in single precision and in integers, zenz's network.
//! - [`tokenizer`]: GPT-2's byte-level BPE, zenz's tokenizer.
//! - [`zenz`]: zenz used alone, with zenz-v3's prompt and greedy decoding; and
//!   its model files.
#![forbid(unsafe_code)]

use std::fmt;
use std::path::PathBuf;

use kiral_model::{file, json, safetensors};

pub mod gpt2;
pub mod tokenizer;
pub mod zenz;

/// Why a model was not loaded, or did not compute.
#[derive(Debug)]
pub enum Error {
    /// A file that could not be read.
    Io(PathBuf, std::io::Error),
    /// A file of JSON, named, that is not JSON.
    Json(&'static str, json::Error),
    /// Weights that are not a safetensors file.
    SafeTensors(safetensors::Error),
    /// A configuration of a model other than the one computed.
    Config(String),
    /// A tensor missing, or not of the configuration's shape.
    Tensor(String),
    /// A tokenizer's files that are not what they should be.
    Tokenizer(String),
    /// Tokens a model cannot compute: none, out of its vocabulary, or past its
    /// room.
    Input(String),
    /// A model file not of the architecture and tokenizer read.
    Architecture(String),
    /// A model file that could not be written.
    Write(file::BuildError),
}

impl fmt::Display for Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(path, error) => write!(formatter, "{}: {error}", path.display()),
            Self::Json(name, error) => write!(formatter, "{name}: {error}"),
            Self::SafeTensors(error) => write!(formatter, "the weights: {error}"),
            Self::Config(message) => write!(formatter, "the configuration: {message}"),
            Self::Tensor(message) => write!(formatter, "the weights: {message}"),
            Self::Tokenizer(message) => write!(formatter, "the tokenizer: {message}"),
            Self::Input(message) => write!(formatter, "the input: {message}"),
            Self::Architecture(message) => write!(formatter, "the model file: {message}"),
            Self::Write(error) => write!(formatter, "the model file: {error}"),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(_, error) => Some(error),
            Self::Json(_, error) => Some(error),
            Self::SafeTensors(error) => Some(error),
            Self::Write(error) => Some(error),
            Self::Config(_) | Self::Tensor(_) | Self::Tokenizer(_) | Self::Input(_) | Self::Architecture(_) => None,
        }
    }
}
