// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later
//
// This file is part of KIRAL. The SPDX lines name its copyright holder and its
// license, in the machine-readable form of the REUSE specification: the GNU
// General Public License, version 3 or any later version
// (LICENSES/GPL-3.0-or-later.txt).

//! safetensors files, read by hand: the form zenz's weights are published in.
//!
//! A file is the length of its header (eight bytes, little-endian), the header
//! (a JSON object: for each tensor its element type, its shape and the offsets
//! of its bytes, and an optional `__metadata__` of strings), and the tensors'
//! bytes. Every tensor's offsets are checked against its shape and type, and
//! the tensors against one another: together they cover the bytes after the
//! header, one after another, with nothing between them and nothing over, as the
//! format requires -- a file cannot hide anything in it. Nothing is copied: the
//! tensors are slices of the file's bytes.

use std::fmt;

use crate::half;
use crate::json::{self, Value};

/// The longest header taken, as safetensors' own reader takes it: 100 MB.
pub const MAX_HEADER: u64 = 100_000_000;

/// The element types of safetensors.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Dtype {
    /// Booleans, a byte each.
    Bool,
    /// Unsigned integers of 8 bits.
    U8,
    /// Signed integers of 8 bits.
    I8,
    /// Unsigned integers of 16 bits.
    U16,
    /// Signed integers of 16 bits.
    I16,
    /// IEEE half precision.
    F16,
    /// bfloat16: the upper half of a float of single precision.
    Bf16,
    /// Unsigned integers of 32 bits.
    U32,
    /// Signed integers of 32 bits.
    I32,
    /// IEEE single precision.
    F32,
    /// Unsigned integers of 64 bits.
    U64,
    /// Signed integers of 64 bits.
    I64,
    /// IEEE double precision.
    F64,
}

impl Dtype {
    /// The type of the name the header gives it.
    #[must_use]
    pub fn from_name(name: &str) -> Option<Self> {
        Some(match name {
            "BOOL" => Self::Bool,
            "U8" => Self::U8,
            "I8" => Self::I8,
            "U16" => Self::U16,
            "I16" => Self::I16,
            "F16" => Self::F16,
            "BF16" => Self::Bf16,
            "U32" => Self::U32,
            "I32" => Self::I32,
            "F32" => Self::F32,
            "U64" => Self::U64,
            "I64" => Self::I64,
            "F64" => Self::F64,
            _ => return None,
        })
    }

    /// The bytes of an element.
    #[must_use]
    pub const fn size(self) -> usize {
        match self {
            Self::Bool | Self::U8 | Self::I8 => 1,
            Self::U16 | Self::I16 | Self::F16 | Self::Bf16 => 2,
            Self::U32 | Self::I32 | Self::F32 => 4,
            Self::U64 | Self::I64 | Self::F64 => 8,
        }
    }
}

/// Why bytes are not a safetensors file that can be read.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    /// Fewer than the eight bytes of the header's length.
    Short,
    /// A header longer than the bytes after its length, or than [`MAX_HEADER`].
    HeaderLength(u64),
    /// A header that is not JSON.
    Json(json::Error),
    /// A header that is JSON but not what the format says it is, and why.
    Header(String),
    /// A tensor's element type that this reader does not know.
    Dtype(String),
    /// A tensor's offsets that do not hold its shape, or that leave bytes out or
    /// overlap another's.
    Offsets(String),
    /// A tensor converted to floats that is not of a floating-point type.
    NotFloat(Dtype),
}

impl fmt::Display for Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Short => formatter.write_str("fewer than the eight bytes of a header's length"),
            Self::HeaderLength(length) => write!(formatter, "a header of {length} bytes, longer than can be read"),
            Self::Json(error) => write!(formatter, "a header that is not JSON: {error}"),
            Self::Header(message) | Self::Offsets(message) => formatter.write_str(message),
            Self::Dtype(name) => write!(formatter, "an element type {name} that is not read"),
            Self::NotFloat(dtype) => write!(formatter, "a tensor of {dtype:?}, not of floats"),
        }
    }
}

impl std::error::Error for Error {}

/// A tensor of the file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorInfo {
    /// Its name.
    pub name: String,
    /// The type of its elements.
    pub dtype: Dtype,
    /// Its shape, the outermost dimension first.
    pub shape: Vec<usize>,
    /// The offset of its first byte after the header.
    pub begin: usize,
    /// The offset after its last byte.
    pub end: usize,
}

/// A tensor's type, shape and bytes.
#[derive(Debug, Clone, Copy)]
pub struct Tensor<'a> {
    /// The type of its elements.
    pub dtype: Dtype,
    /// Its shape, the outermost dimension first.
    pub shape: &'a [usize],
    /// Its bytes, little-endian.
    pub bytes: &'a [u8],
}

impl Tensor<'_> {
    /// The number of its elements.
    #[must_use]
    pub fn len(&self) -> usize {
        self.bytes.len() / self.dtype.size()
    }

    /// Whether it has no elements.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.bytes.is_empty()
    }

    /// Its elements in single precision: those of half precision and bfloat16
    /// converted exactly, those of double precision rounded to nearest.
    ///
    /// # Errors
    ///
    /// Where its elements are not floating-point.
    pub fn to_f32(&self) -> Result<Vec<f32>, Error> {
        Ok(match self.dtype {
            Dtype::F32 => self
                .bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|&bytes| f32::from_le_bytes(bytes))
                .collect(),
            Dtype::F16 => self
                .bytes
                .as_chunks::<2>()
                .0
                .iter()
                .map(|&bytes| half::f16_to_f32(u16::from_le_bytes(bytes)))
                .collect(),
            Dtype::Bf16 => self
                .bytes
                .as_chunks::<2>()
                .0
                .iter()
                .map(|&bytes| half::bf16_to_f32(u16::from_le_bytes(bytes)))
                .collect(),
            Dtype::F64 => self
                .bytes
                .as_chunks::<8>()
                .0
                .iter()
                .map(|&bytes| half::f64_to_f32(f64::from_le_bytes(bytes)))
                .collect(),
            dtype => return Err(Error::NotFloat(dtype)),
        })
    }
}

/// A safetensors file read from its bytes.
#[derive(Debug, Clone)]
pub struct SafeTensors<'a> {
    data: &'a [u8],
    tensors: Vec<TensorInfo>,
    metadata: Vec<(String, String)>,
}

impl<'a> SafeTensors<'a> {
    /// The file of `bytes`, its header read and checked.
    ///
    /// # Errors
    ///
    /// Where the bytes are not a safetensors file, or one with element types
    /// this reader does not know.
    pub fn parse(bytes: &'a [u8]) -> Result<Self, Error> {
        let (length, rest) = bytes.split_first_chunk::<8>().ok_or(Error::Short)?;
        let length = u64::from_le_bytes(*length);
        let header_length = usize::try_from(length)
            .ok()
            .filter(|&header| length <= MAX_HEADER && header <= rest.len())
            .ok_or(Error::HeaderLength(length))?;
        let (header, data) = rest.split_at(header_length);
        let header = json::parse(header).map_err(Error::Json)?;
        let header = header
            .as_object()
            .ok_or_else(|| Error::Header("a header that is not an object".to_owned()))?;
        let mut tensors = Vec::with_capacity(header.len());
        let mut metadata = Vec::new();
        for (name, value) in header.iter() {
            if name == "__metadata__" {
                metadata = read_metadata(value)?;
            } else {
                tensors.push(read_tensor(name, value)?);
            }
        }
        check_coverage(&mut tensors, data.len())?;
        Ok(Self {
            data,
            tensors,
            metadata,
        })
    }

    /// The tensors, in the order of their bytes.
    #[must_use]
    pub fn tensors(&self) -> &[TensorInfo] {
        &self.tensors
    }

    /// The tensor of a name.
    #[must_use]
    pub fn tensor(&self, name: &str) -> Option<Tensor<'_>> {
        self.tensors
            .iter()
            .find(|tensor| tensor.name == name)
            .map(|tensor| Tensor {
                dtype: tensor.dtype,
                shape: &tensor.shape,
                bytes: &self.data[tensor.begin..tensor.end],
            })
    }

    /// The strings of the header's `__metadata__`, in their order.
    #[must_use]
    pub fn metadata(&self) -> &[(String, String)] {
        &self.metadata
    }
}

fn read_metadata(value: &Value) -> Result<Vec<(String, String)>, Error> {
    let object = value
        .as_object()
        .ok_or_else(|| Error::Header("__metadata__ that is not an object".to_owned()))?;
    object
        .iter()
        .map(|(name, value)| {
            value
                .as_str()
                .map(|text| (name.to_owned(), text.to_owned()))
                .ok_or_else(|| Error::Header(format!("__metadata__'s {name} that is not a string")))
        })
        .collect()
}

fn read_tensor(name: &str, value: &Value) -> Result<TensorInfo, Error> {
    let fields = value
        .as_object()
        .ok_or_else(|| Error::Header(format!("tensor {name} that is not an object")))?;
    for (field, _) in fields.iter() {
        if !matches!(field, "dtype" | "shape" | "data_offsets") {
            return Err(Error::Header(format!(
                "tensor {name} with a field {field} the format does not have"
            )));
        }
    }
    let field = |field: &str| {
        fields
            .get(field)
            .ok_or_else(|| Error::Header(format!("tensor {name} without its {field}")))
    };
    let dtype_name = field("dtype")?
        .as_str()
        .ok_or_else(|| Error::Header(format!("tensor {name} whose dtype is not a string")))?;
    let dtype = Dtype::from_name(dtype_name).ok_or_else(|| Error::Dtype(dtype_name.to_owned()))?;
    let integer = |value: &Value| value.as_u64().and_then(|value| usize::try_from(value).ok());
    let shape = field("shape")?
        .as_array()
        .ok_or_else(|| Error::Header(format!("tensor {name} whose shape is not an array")))?
        .iter()
        .map(|dimension| {
            integer(dimension).ok_or_else(|| Error::Header(format!("tensor {name} with a dimension out of range")))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let offsets = field("data_offsets")?
        .as_array()
        .ok_or_else(|| Error::Header(format!("tensor {name} whose data_offsets is not an array")))?;
    let [begin, end] = offsets else {
        return Err(Error::Header(format!("tensor {name} with other than two data_offsets")));
    };
    let (Some(begin), Some(end)) = (integer(begin), integer(end)) else {
        return Err(Error::Offsets(format!("tensor {name} with offsets out of range")));
    };
    let size = shape
        .iter()
        .try_fold(dtype.size(), |size, &dimension| size.checked_mul(dimension))
        .ok_or_else(|| Error::Offsets(format!("tensor {name} of more bytes than can be counted")))?;
    if end < begin || end - begin != size {
        return Err(Error::Offsets(format!(
            "tensor {name} of {size} bytes at offsets {begin} to {end}"
        )));
    }
    Ok(TensorInfo {
        name: name.to_owned(),
        dtype,
        shape,
        begin,
        end,
    })
}

/// Sorts the tensors by their offsets, and checks that they cover the bytes of
/// the data one after another, with nothing between them and nothing over.
fn check_coverage(tensors: &mut [TensorInfo], length: usize) -> Result<(), Error> {
    tensors.sort_by_key(|tensor| (tensor.begin, tensor.end));
    let mut covered = 0;
    for tensor in tensors.iter() {
        if tensor.begin != covered {
            return Err(Error::Offsets(format!(
                "tensor {} begins at {}, where the bytes before end at {covered}",
                tensor.name, tensor.begin
            )));
        }
        covered = tensor.end;
    }
    if covered != length {
        return Err(Error::Offsets(format!(
            "tensors that cover {covered} of the {length} bytes after the header"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A file of the header's text and the data's bytes.
    fn file(header: &str, data: &[u8]) -> Vec<u8> {
        let length = u64::try_from(header.len()).expect("a short header");
        [&length.to_le_bytes()[..], header.as_bytes(), data].concat()
    }

    #[test]
    fn a_file_is_read() {
        let header = r#"{"__metadata__":{"format":"pt"},"b":{"dtype":"BF16","shape":[2],"data_offsets":[8,12]},
            "a":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}  "#;
        let mut data = Vec::new();
        data.extend(1.5f32.to_le_bytes());
        data.extend((-2.0f32).to_le_bytes());
        data.extend(0x3f80u16.to_le_bytes());
        data.extend(0xc040u16.to_le_bytes());
        let bytes = file(header, &data);
        let file = SafeTensors::parse(&bytes).expect("a file");
        assert_eq!(file.metadata(), [("format".to_owned(), "pt".to_owned())]);
        assert_eq!(
            file.tensors()
                .iter()
                .map(|tensor| tensor.name.as_str())
                .collect::<Vec<_>>(),
            ["a", "b"]
        );
        let a = file.tensor("a").expect("a");
        assert_eq!((a.dtype, a.shape, a.len()), (Dtype::F32, &[2][..], 2));
        assert_eq!(a.to_f32(), Ok(vec![1.5, -2.0]));
        assert_eq!(file.tensor("b").expect("b").to_f32(), Ok(vec![1.0, -3.0]));
        assert!(file.tensor("c").is_none());
    }

    #[test]
    fn what_the_format_does_not_allow_is_refused() {
        let four = [0u8; 4];
        let cases: [(&str, &[u8]); 12] = [
            (r"[]", &[]),
            (r#"{"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#, &[0; 8]),
            (r#"{"a":{"dtype":"F32","shape":[2],"data_offsets":[0,4]}}"#, &four),
            (r#"{"a":{"dtype":"F32","shape":[1],"data_offsets":[4,0]}}"#, &four),
            (
                r#"{"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},"b":{"dtype":"U8","shape":[2],"data_offsets":[2,4]}}"#,
                &four,
            ),
            (r#"{"a":{"dtype":"U8","shape":[2],"data_offsets":[2,4]}}"#, &four),
            (r#"{"a":{"dtype":"F8_E4M3","shape":[4],"data_offsets":[0,4]}}"#, &four),
            (
                r#"{"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4],"extra":1}}"#,
                &four,
            ),
            (r#"{"a":{"dtype":"F32","shape":[1]}}"#, &four),
            (
                r#"{"a":{"dtype":"U8","shape":[18446744073709551615,2],"data_offsets":[0,4]}}"#,
                &four,
            ),
            (r#"{"a":{"dtype":"U8","shape":[4],"data_offsets":[0,4,8]}}"#, &four),
            (r#"{"__metadata__":{"n":1}}"#, &[]),
        ];
        for (header, data) in cases {
            assert!(SafeTensors::parse(&file(header, data)).is_err(), "{header} was taken");
        }
        assert_eq!(SafeTensors::parse(&[0; 7]).unwrap_err(), Error::Short);
        let mut long = u64::MAX.to_le_bytes().to_vec();
        long.extend(b"{}");
        assert_eq!(SafeTensors::parse(&long).unwrap_err(), Error::HeaderLength(u64::MAX));
        assert!(SafeTensors::parse(&file("{}", &[])).is_ok());
    }

    #[test]
    fn integers_are_not_floats() {
        let bytes = file(
            r#"{"a":{"dtype":"I8","shape":[4],"data_offsets":[0,4]}}"#,
            &[1, 2, 3, 4],
        );
        let file = SafeTensors::parse(&bytes).expect("a file");
        assert_eq!(file.tensor("a").expect("a").to_f32(), Err(Error::NotFloat(Dtype::I8)));
    }
}
