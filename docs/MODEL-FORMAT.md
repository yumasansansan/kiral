<!--
SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
SPDX-License-Identifier: GPL-3.0-or-later

This file is part of KIRAL. The SPDX lines name its copyright holder and its
license, in the machine-readable form of the REUSE specification: the GNU
General Public License, version 3 or any later version
(LICENSES/GPL-3.0-or-later.txt).
-->

# The model file

A KIRAL model is one file: a network's weights, its tokenizer, and what the
weights are and where they come from. It is laid out to be mapped from the disk
and read where it lies: the weights of the linear layers are kept quantized, in
the layout the kernels read them in, so that loading a model is mapping its
file and checking its tables, with nothing to quantize, convert or copy but a
few small vectors. `kiral-tools model convert` writes it from a model as
Hugging Face publishes it (`crates/kiral-nn/src/zenz.rs`,
`crates/kiral-model/src/file.rs`), `kiral_model::file::ModelFile::parse` reads
it, and `kiral-tools model check` checks it whole against the digest it
carries.

A model file is not KIRAL's code and is not under its license: it holds
weights under the terms of their source, which its metadata names, with the
notices that go with them. One made from zenz-v3.1-small goes under CC BY-SA
4.0, as zenz-v3.1-small does.

Every number is little-endian. Offsets count bytes from the beginning of the
file, but those of the tensors' bytes, which count them from the beginning of
the data.

## The header

| Offset | Bytes | What |
| ---: | ---: | --- |
| 0 | 8 | the magic, `KIRALMDL` |
| 8 | 4 | the version of the format: 1 |
| 12 | 4 | reserved, 0 |
| 16 | 32 | the SHA-256 of every byte of the file from offset 48 to its end |
| 48 | 16 | the architecture, in ASCII, zeros after it: `gpt2-zenz` |
| 64 | 16 | the tokenizer, in ASCII, zeros after it: `byte-bpe` |
| 80 | 16 × 5 | the sections: for each, its offset and its length, eight bytes each |

The sections follow, in this order, each at an offset that is a multiple of 64,
with zeros between them:

| | Section | Length |
| ---: | --- | --- |
| 0 | the metadata | four strings |
| 1 | the configuration | the architecture's numbers |
| 2 | the tokenizer | its words, its merges and its special words |
| 3 | the tensors | an entry for each, and their names |
| 4 | the data | the tensors' bytes |

A reader checks the header, the sections and the tensors' entries when it
opens a file. The digest is checked where the whole file is read anyway -- by
`kiral-tools model check`, as when a package installs the file -- rather than
at every start of the input method, which would read the whole file before the
first conversion.

## The metadata

Four strings, each its length in bytes (four bytes) and its UTF-8, as in a
dictionary file: the model's name; the license of its weights, as an SPDX
expression; where its weights come from; and the notices that go with any copy
of it, among them what was changed from its source.

## The architecture `gpt2-zenz`

zenz's network: GPT-2 as Hugging Face's `GPT2LMHeadModel` computes it --
learned positions, a layer normalization before each block's attention and
before its expansion and one after the last block, an expansion of four times
the width through GELU in its approximation by tanh (`gelu_new`), attention
scaled by the square root of a head's width and by nothing else -- converting
in zenz-v3's prompt.

Its configuration is six numbers of four bytes:

| Offset | What |
| ---: | --- |
| 0 | the blocks (`n_layer`) |
| 4 | the heads of each attention (`n_head`), which divide the width |
| 8 | the width of the residual stream (`n_embd`) |
| 12 | the positions it has embeddings for (`n_positions`) |
| 16 | the words of its vocabulary (`vocab_size`) |
| 20 | the epsilon of its layer normalizations (`layer_norm_epsilon`), in single precision |

Its tensors, by Hugging Face's names, where *d* is the width, *V* the
vocabulary, *P* the positions, and `N` each block from 0:

| Name | Shape | Type |
| --- | --- | --- |
| `transformer.wte.weight` | *V* × *d* | floats: the token embedding |
| `transformer.wpe.weight` | *P* × *d* | floats: the position embedding |
| `transformer.h.N.ln_1.weight`, `transformer.h.N.ln_1.bias` | *d* | floats |
| `transformer.h.N.attn.c_attn.weight` | *d* × 3*d* | weights in panels |
| `transformer.h.N.attn.c_attn.bias` | 3*d* | floats |
| `transformer.h.N.attn.c_proj.weight` | *d* × *d* | weights in panels |
| `transformer.h.N.attn.c_proj.bias` | *d* | floats |
| `transformer.h.N.ln_2.weight`, `transformer.h.N.ln_2.bias` | *d* | floats |
| `transformer.h.N.mlp.c_fc.weight` | *d* × 4*d* | weights in panels |
| `transformer.h.N.mlp.c_fc.bias` | 4*d* | floats |
| `transformer.h.N.mlp.c_proj.weight` | 4*d* × *d* | weights in panels |
| `transformer.h.N.mlp.c_proj.bias` | *d* | floats |
| `transformer.ln_f.weight`, `transformer.ln_f.bias` | *d* | floats |
| `lm_head.weight` | *d* × *V* | weights in panels, of 8 bits: the output layer |

Floats are of single precision or bfloat16. A linear layer's weights are its
inputs by its outputs, as GPT-2 keeps them (its `Conv1D`). The output layer's
are the token embedding's, transposed -- its rows are the layer's columns --
and quantized at 8 bits whatever the blocks' are, as its errors move the logits
themselves.

## The tokenizer `byte-bpe`

GPT-2's byte-level BPE, applied to each character on its own: each byte of a
character is the word of that byte, and the merges join two adjacent words
into the word they make, the earliest merge first, until none applies. No
merge joins the bytes of two characters -- a reader refuses one that would --
which is what lets the merges be applied a character at a time rather than to
the pieces GPT-2's regular expression splits a text into. A byte that no word
is is the unknown word. The special words have no bytes, and are never made
from text.

| Offset | Bytes | What |
| ---: | ---: | --- |
| 0 | 4 | the words, *W* |
| 4 | 4 | the merges, *M* |
| 8 | 4 | the unknown word |
| 12 | 4 | the word that ends a text |
| 16 | 4 | the word that begins one, or 2³² − 1 where there is none |
| 20 | 4 | the word of padding, or 2³² − 1 where there is none |
| 24 | 4 × (*W* + 1) | the starts of the words' bytes |
| 28 + 4*W* | 12 × *M* | the merges, in their order: for each, the left word, the right word and the word they make, four bytes each |
| 28 + 4*W* + 12*M* | | the words' bytes |

The bytes of the word *w* are those from the *w*-th start to the (*w* + 1)-th:
the starts go from 0 to the bytes' end and never back. The special words have
none, and every other word has some; the word of a byte is the one word whose
bytes are that byte alone, and a merge's word has its left word's bytes and
then its right word's.

## The tensors

| Offset | Bytes | What |
| ---: | ---: | --- |
| 0 | 4 | the tensors, *T* |
| 4 | 4 | reserved, 0 |
| 8 | 64 × *T* | an entry for each |
| 8 + 64*T* | | their names, in UTF-8 |

An entry:

| Offset | Bytes | What |
| ---: | ---: | --- |
| 0 | 4 | the offset of its name in the names |
| 4 | 4 | the length of its name in bytes |
| 8 | 4 | its type: 1 floats of single precision, 2 floats of bfloat16, 3 weights in panels |
| 12 | 4 | its rank, from 1 to 4; 2 for weights in panels |
| 16 | 4 × 4 | its shape, each length at least 1, and 0 past its rank |
| 32 | 8 | the offset of its bytes in the data, a multiple of 64 |
| 40 | 8 | the length of its bytes |
| 48 | 16 | the layout and scheme of weights in panels; zeros for floats |

No two tensors have the same name, or bytes that overlap. The bytes of floats
are their values one after another, the last length of the shape the one whose
index changes fastest: four bytes a value of single precision, two of bfloat16,
the upper half of a single precision's bits.

## Weights in panels

A linear layer's weights, *K* inputs by *N* outputs (its shape), quantized to
integers held in bytes, a scale for each output's group of *G* inputs, and
packed in panels of *C* columns, as the kernels read them
(`crates/kiral-kernels/src/quantized.rs`). The last 16 bytes of the entry:

| Offset | Bytes | What |
| ---: | ---: | --- |
| 48 | 2 | the columns of a panel, *C*: 16 |
| 50 | 2 | the inputs of a group, *G*: 64, a multiple of 4 |
| 52 | 1 | the bits of a weight, from 2 to 8, held in a byte whatever they are |
| 53 | 1 | 1 where each group's scale is the one of least squared error, 0 where it is that of the group's greatest magnitude |
| 54 | 2 | reserved, 0 |
| 56 | 4 | the order of the inputs' rotation, a power of four from 4, or 0 where they are not rotated |
| 60 | 4 | reserved, 0 |

The bytes are the panels of the columns 0 to *C* − 1, *C* to 2*C* − 1 and on,
⌈*N* / *C*⌉ of them; a panel is its groups of the inputs, ⌈*K* / *G*⌉ of them;
and a group is its integers, quad by quad -- *G* / 4 quads, each the four bytes
of each of the *C* columns, a column's four side by side -- then the *C*
columns' scales and then their offsets, in single precision: *G C* + 8*C*
bytes, 1152 for groups of 64 and panels of 16. A column past *N* is zeros, of a
scale of 0, and an input past *K* a zero.

A weight is its scale times its integer, *w* = *s q*, the integer signed and
|*q*| ≤ 2^(*b* − 1) − 1 for *b* bits; a column's offset in a group is its scale
times the sum of its integers there, which the zero points of the inputs take
away. The scheme's bits and its choice of scales are for the reader to know
what it holds; the rotation it must apply too. Rotated, each block of *R*
inputs, *R* the order, is multiplied by the regular Hadamard matrix of that
order divided by √*R* -- the weights' columns before they were quantized, and
the inputs as they come -- which leaves the products as they were; the inputs
after the last whole block are rotated in blocks of the largest powers of four
that fit. The matrix is the Kronecker power of

```text
 1  1  1 -1
 1  1 -1  1
 1 -1  1  1
-1  1  1  1
```

## The data

The tensors' bytes, each at its offset, with zeros between them.
