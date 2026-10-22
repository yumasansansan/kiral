// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later
//
// This file is part of KIRAL. The SPDX lines name its copyright holder and its
// license, in the machine-readable form of the REUSE specification: the GNU
// General Public License, version 3 or any later version
// (LICENSES/GPL-3.0-or-later.txt).

//! The model file: what is written is read back as it was, every tensor's bytes
//! on a multiple of 64 bytes of the file; the digest finds a byte changed; and
//! what the format does not take is refused, when it is written and when it is
//! read.

use kiral_model::file::{self, BuildError, Contents, Error, Kind, Metadata, ModelFile, Panels, Specials, Tensor};
use kiral_model::sha256;

const PANELS: Panels = Panels {
    columns: 16,
    group: 64,
    bits: 4,
    fitted: true,
    rotation: 16,
};

const METADATA: Metadata<'static> = Metadata {
    name: "テスト",
    license: "CC-BY-SA-4.0",
    source: "made here",
    notice: "nothing\nto notice",
};

/// The words of a vocabulary: two special words, the bytes of "ab" and "あ",
/// and the words "ab" and "あ".
const WORDS: [&[u8]; 9] = [b"", b"", b"a", b"b", &[0xe3], &[0x81], &[0x82], b"ab", "あ".as_bytes()];

/// The merges that make "ab" and "あ".
const MERGES: [[u32; 3]; 2] = [[2, 3, 7], [4, 5, 8]];

const SPECIALS: Specials = Specials {
    unknown: 0,
    end: 1,
    begin: None,
    padding: Some(0),
};

/// The bytes of floats of single precision.
fn singles(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|value| value.to_le_bytes()).collect()
}

/// A small model's tensors: floats of both precisions, and weights in panels of
/// a layer whose depth and width are not multiples of a group and a panel.
fn tensors() -> Vec<(String, Kind, Vec<usize>, Vec<u8>)> {
    let embedding: Vec<u8> = (0..15u16).flat_map(|i| (0x3f80 + 16 * i).to_le_bytes()).collect();
    let panels = PANELS.length(70, 20).expect("a length");
    vec![
        ("embedding".to_owned(), Kind::Bf16, vec![5, 3], embedding),
        (
            "norm.weight".to_owned(),
            Kind::F32,
            vec![3],
            singles(&[1.0, -2.5, 3.25]),
        ),
        (
            "layer.weight".to_owned(),
            Kind::Panels(PANELS),
            vec![70, 20],
            (0..panels).map(|i| u8::try_from(i % 251).expect("a byte")).collect(),
        ),
        ("cube".to_owned(), Kind::F32, vec![2, 1, 3, 1], singles(&[0.5; 6])),
    ]
}

/// The tensors as the writer takes them.
fn views(tensors: &[(String, Kind, Vec<usize>, Vec<u8>)]) -> Vec<Tensor<'_>> {
    tensors
        .iter()
        .map(|(name, kind, shape, bytes)| Tensor {
            name,
            kind: *kind,
            shape: shape.clone(),
            bytes,
        })
        .collect()
}

fn contents<'a>(tensors: &'a [Tensor<'a>]) -> Contents<'a> {
    Contents {
        architecture: "test-net",
        tokenizer: "byte-bpe",
        metadata: METADATA,
        configuration: &[1, 2, 3, 4, 5],
        words: &WORDS,
        merges: &MERGES,
        specials: SPECIALS,
        tensors,
    }
}

fn written() -> Vec<u8> {
    let tensors = tensors();
    file::write(&contents(&views(&tensors))).expect("a file")
}

/// A section's offset and length.
fn section(bytes: &[u8], number: usize) -> (usize, usize) {
    let at = file::TABLE + 16 * number;
    let read = |at: usize| {
        usize::try_from(u64::from_le_bytes(bytes[at..at + 8].try_into().expect("eight bytes"))).expect("a usize")
    };
    (read(at), read(at + 8))
}

/// The offset in the file of a tensor's entry.
fn entry(bytes: &[u8], number: usize) -> usize {
    section(bytes, file::TENSORS).0 + 8 + file::ENTRY * number
}

#[test]
fn what_is_written_is_read_back() {
    let bytes = written();
    let model = ModelFile::parse(&bytes).expect("a model file");
    assert_eq!(model.architecture(), "test-net");
    assert_eq!(model.tokenizer(), "byte-bpe");
    assert_eq!(model.metadata(), METADATA);
    assert_eq!(model.configuration(), [1, 2, 3, 4, 5]);
    let vocabulary = model.vocabulary();
    assert_eq!(vocabulary.len(), WORDS.len());
    for (number, word) in WORDS.iter().enumerate() {
        assert_eq!(vocabulary.word(number), Some(*word));
    }
    assert_eq!(vocabulary.word(WORDS.len()), None);
    assert_eq!(vocabulary.merges().collect::<Vec<_>>(), MERGES);
    assert_eq!(vocabulary.merge_count(), MERGES.len());
    assert_eq!(vocabulary.specials(), SPECIALS);
    let expected = tensors();
    assert_eq!(model.tensors().len(), expected.len());
    for (tensor, (name, kind, shape, data)) in model.tensors().iter().zip(&expected) {
        assert_eq!(
            (tensor.name, tensor.kind, &tensor.shape, tensor.bytes),
            (name.as_str(), *kind, shape, &data[..])
        );
        let offset = tensor.bytes.as_ptr().addr() - bytes.as_ptr().addr();
        assert_eq!(offset % file::ALIGNMENT, 0, "{name} at {offset}");
    }
    let norm = model.tensor("norm.weight").and_then(Tensor::floats).expect("floats");
    assert_eq!(norm.to_vec(), [1.0, -2.5, 3.25]);
    let embedding = model.tensor("embedding").and_then(Tensor::floats).expect("floats");
    assert_eq!(embedding.len(), 15);
    let mut row = [0.0f32; 3];
    embedding.read(3, &mut row);
    assert_eq!(row, [1.375, 1.5, 1.625]);
    assert!(model.tensor("layer.weight").and_then(Tensor::floats).is_none());
    assert!(model.tensor("missing").is_none());
    assert_eq!(model.digest(), sha256::digest(&bytes[file::DIGESTED..]));
    assert!(model.verify());
}

#[test]
fn the_digest_finds_a_byte_changed() {
    let mut bytes = written();
    let last = bytes.len() - 1;
    bytes[last] ^= 1;
    let model = ModelFile::parse(&bytes).expect("a model file, its digest unchecked");
    assert!(!model.verify());
    let mut bytes = written();
    bytes[20] ^= 0x80;
    assert!(!ModelFile::parse(&bytes).expect("a model file").verify());
}

#[test]
fn what_the_format_does_not_take_is_not_written() {
    let owned = tensors();
    let tensors = views(&owned);
    for name in ["", "seventeen-letters", "モデル", "a b"] {
        let error = file::write(&Contents {
            architecture: name,
            ..contents(&tensors)
        });
        assert_eq!(error, Err(BuildError::Name(name.to_owned())));
    }
    let refused = |change: &dyn Fn(&mut Tensor<'_>)| {
        let mut changed = tensors.clone();
        change(&mut changed[2]);
        file::write(&contents(&changed))
    };
    let tensor_error = |result: Result<Vec<u8>, BuildError>| matches!(result, Err(BuildError::Tensor(..)));
    assert!(tensor_error(refused(&|tensor| tensor.bytes = &tensor.bytes[1..])));
    assert!(tensor_error(refused(&|tensor| tensor.shape = vec![70, 20, 1])));
    assert!(tensor_error(refused(&|tensor| tensor.name = "embedding")));
    assert!(tensor_error(refused(&|tensor| tensor.name = "")));
    for panels in [
        Panels { bits: 9, ..PANELS },
        Panels { bits: 1, ..PANELS },
        Panels { group: 6, ..PANELS },
        Panels { columns: 0, ..PANELS },
        Panels { rotation: 8, ..PANELS },
        Panels { rotation: 1, ..PANELS },
    ] {
        assert!(
            tensor_error(refused(&|tensor| tensor.kind = Kind::Panels(panels))),
            "{panels:?}"
        );
    }
    let merges = [[2, 3, 9]];
    let error = file::write(&Contents {
        merges: &merges,
        ..contents(&tensors)
    });
    assert_eq!(error, Err(BuildError::Word(9)));
    let error = file::write(&Contents {
        specials: Specials {
            begin: Some(12),
            ..SPECIALS
        },
        ..contents(&tensors)
    });
    assert_eq!(error, Err(BuildError::Word(12)));
}

#[test]
fn what_the_format_does_not_take_is_not_read() {
    let written = written();
    let refused = |change: &dyn Fn(&mut Vec<u8>)| {
        let mut bytes = written.clone();
        change(&mut bytes);
        ModelFile::parse(&bytes).map(|_| ()).expect_err("refused")
    };
    let put = |bytes: &mut Vec<u8>, at: usize, value: &[u8]| bytes[at..at + value.len()].copy_from_slice(value);
    assert_eq!(refused(&|bytes| bytes[0] = b'k'), Error::NotModel("no magic"));
    assert_eq!(refused(&|bytes| bytes[8] = 2), Error::NotModel("another version"));
    assert_eq!(
        refused(&|bytes| bytes.truncate(100)),
        Error::NotModel("shorter than a header")
    );
    assert!(matches!(refused(&|bytes| bytes[48] = 0), Error::NotModel(_)));
    assert!(matches!(refused(&|bytes| bytes[70] = b' '), Error::NotModel(_)));
    let table = file::TABLE;
    assert!(matches!(refused(&|bytes| bytes[table] += 8), Error::Section(0, _)));
    assert!(matches!(
        refused(&|bytes| bytes.truncate(bytes.len() - 1)),
        Error::Section(4, _)
    ));
    let (tokenizer, _) = section(&written, file::TOKENIZER);
    // The start of word 2 past that of word 3.
    assert!(matches!(
        refused(&|bytes| put(bytes, tokenizer + 24 + 8, &5u32.to_le_bytes())),
        Error::Section(2, _)
    ));
    assert!(matches!(
        refused(&|bytes| put(bytes, tokenizer + 8, &9u32.to_le_bytes())),
        Error::Section(2, _)
    ));
    let merge = tokenizer + 24 + 4 * (WORDS.len() + 1);
    assert!(matches!(
        refused(&|bytes| put(bytes, merge + 8, &9u32.to_le_bytes())),
        Error::Section(2, _)
    ));
    let (first, second) = (entry(&written, 0), entry(&written, 1));
    // The second's bytes at the first's, its name the first's, and so on.
    let first_offset = written[first + 32..first + 40].to_vec();
    assert_eq!(
        refused(&|bytes| put(bytes, second + 32, &first_offset)),
        Error::Tensor(1, "bytes that overlap another's")
    );
    let first_name = written[first..first + 8].to_vec();
    assert_eq!(
        refused(&|bytes| put(bytes, second, &first_name)),
        Error::Tensor(1, "the name of another")
    );
    assert!(matches!(refused(&|bytes| bytes[second + 40] += 1), Error::Tensor(1, _)));
    assert!(matches!(refused(&|bytes| bytes[second + 32] += 4), Error::Tensor(1, _)));
    assert!(matches!(refused(&|bytes| bytes[second + 8] = 9), Error::Tensor(1, _)));
    assert!(matches!(refused(&|bytes| bytes[second + 12] = 0), Error::Tensor(1, _)));
    assert!(matches!(refused(&|bytes| bytes[second + 20] = 1), Error::Tensor(1, _)));
    assert!(matches!(refused(&|bytes| bytes[second + 50] = 1), Error::Tensor(1, _)));
    let panels = entry(&written, 2);
    assert!(matches!(refused(&|bytes| bytes[panels + 53] = 2), Error::Tensor(2, _)));
    assert!(matches!(refused(&|bytes| bytes[panels + 52] = 9), Error::Tensor(2, _)));
    assert!(matches!(refused(&|bytes| bytes[panels + 56] = 8), Error::Tensor(2, _)));
    assert!(matches!(refused(&|bytes| bytes[panels + 60] = 1), Error::Tensor(2, _)));
}
