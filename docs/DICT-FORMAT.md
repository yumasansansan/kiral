<!--
SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
SPDX-License-Identifier: GPL-3.0-or-later

This file is part of KIRAL. The SPDX lines name its copyright holder and its
license, in the machine-readable form of the REUSE specification: the GNU
General Public License, version 3 or any later version
(LICENSES/GPL-3.0-or-later.txt).
-->

# The dictionary file

A KIRAL dictionary is one file: words keyed by their readings, and the costs of
the connections between the words' classes. It is laid out to be read where it
lies -- mapped from the disk, or read whole -- with nothing to copy or decode
before a reading is looked up. `kiral-tools dict build` writes it
(`crates/kiral-dict/src/file.rs`), `kiral_dict::Dictionary::parse` reads it.

A dictionary file is not KIRAL's code and is not under its license: it holds
words under the terms of their source, and those terms' notices, in its
metadata. One built from Mozc's open-source dictionary goes under IPAdic's
terms and Mozc's BSD-3-Clause license, whose text it carries; one with
SudachiDict's nouns, under the Apache License 2.0 and UniDic's BSD-3-Clause
license as well, with SudachiDict's notices.

Every number is little-endian. Offsets count bytes from the beginning of the
file.

## The header

| Offset | Bytes | What |
| ---: | ---: | --- |
| 0 | 8 | the magic, `KIRALDIC` |
| 8 | 4 | the version of the format: 1 |
| 12 | 4 | the number of ids of the classes, *n* (at most 65,535) |
| 16 | 4 | the number of distinct readings, *R* |
| 20 | 4 | the number of entries, *E* |
| 24 | 2 | the id of a word the dictionary does not have |
| 26 | 2 | the cost of a word the dictionary does not have |
| 28 | 4 | reserved, 0 |
| 32 | 16 × 7 | the sections: for each, its offset and its length, eight bytes each |

The sections follow, in this order, each at an offset that is a multiple of 8,
with zeros between them:

| | Section | Length |
| ---: | --- | --- |
| 0 | the metadata | four strings |
| 1 | the alphabet | 8 bytes a character |
| 2 | the units of the trie | 8 bytes a unit |
| 3 | the starts of the readings' entries | 4 × (*R* + 1) |
| 4 | the entries | 12 × *E* |
| 5 | the surfaces | their bytes |
| 6 | the matrix | 2 × *n* × *n* |

## Readings

A reading is kept, and looked up, in its lookup form: its katakana as hiragana
(ヴ as ゔ, ヵ and ヶ as ゕ and ゖ, ヽ and ヾ as ゝ and ゞ), the full-width forms of
ASCII (U+FF01 to U+FF5E) as ASCII, and the ideographic space as a space. A
character becomes one character, so that positions in a reading are positions
in its lookup form. A word whose surface is its own reading in the lookup form
-- a parenthesis, say -- is written as the input wrote it.

## The metadata

Four strings, each its length in bytes (four bytes) and its UTF-8: the
dictionary's name; the license of its words, as an SPDX expression; where its
words come from; and the notices that go with any copy of it.

## The alphabet

The characters of the readings, each a pair of four-byte numbers: the
character's Unicode scalar value and its code in the trie, from 1 (0 ends a
reading). The pairs are sorted by the character, to be searched by halves; the
codes are given in the order of the characters' frequency in the readings, the
commonest first, so that the trie's nodes, whose children are mostly of them,
pack close.

## The trie

A double-array trie of the readings, over their characters' codes. A unit is
two four-byte numbers, a base and a check. The child of the unit *s* by the
code *c* is the unit *base*(*s*) + *c* where its check is *s*, and there is no
such child otherwise. The root is unit 0. Code 0 ends a reading: the unit the
root reaches through a reading's codes and then code 0 holds the reading's
number, from 0 to *R* − 1, in its base. The readings are numbered in the order
of their codes, sorted. A unit that is no node's has the check 2³² − 1, and the
root the check 2³² − 2, which no unit's number is.

The readings that begin an input are found by walking the trie from the root,
a character at a time, and taking each unit along the way that has a child by
code 0.

## The entries

The entries of the reading numbered *r* are those from the *r*-th start to the
(*r* + 1)-th: the starts go from 0 to *E* and never back. A reading's entries
are sorted by their cost, the cheapest first. An entry is twelve bytes:

| Offset | Bytes | What |
| ---: | ---: | --- |
| 0 | 4 | the offset of its surface in the surfaces |
| 4 | 2 | the length of its surface in bytes (the low 12 bits, at most 4,095), and its flags (the high 4 bits) |
| 6 | 2 | the id of its class on its left |
| 8 | 2 | the id of its class on its right |
| 10 | 2 | its cost: the higher, the less likely |

The flags:

| Flag | What |
| ---: | --- |
| 1 | a word whose reading is spelt otherwise than the word is (Mozc's `SPELLING_CORRECTION`: ばんしあ for ヴァンシア) |
| 2 | a digit read in kana, its surface an Arabic numeral (いち for 1): written in kanji (一, and 零 for 0) unless the word after it has flag 4, as Mozc shows it |
| 4 | a suffix of nouns (年, 人前), after which a digit read in kana stays an Arabic numeral |

Flag 8 is reserved, and 0. A surface is UTF-8; entries may share one. A reader leaves
out an entry whose surface lies past the surfaces' end or is not UTF-8.

## The matrix

The costs of the connections: that of a word whose class on its right is *r*
followed by one whose class on its left is *l* is the number of two bytes at
2 × (*r* × *n* + *l*). Class 0 is the beginning and the end of a sentence; the
beginning followed by the end costs 0.

A path's cost is the sum of its words' costs and of its connections', from the
beginning of the sentence before its first word to the end after its last. A
character that no reading of the dictionary covers is taken as a word of one
character, of the header's id and cost, written as the input wrote it.

## Numbers

A number of more than a digit said in kana -- よんじゅうご, せんきゅうひゃくきゅうじゅういち,
いちまん -- is taken as one word made of the numerals that the dictionary has for
its words: an entry whose surface is the one numeral (一 to 九, 十, 百, 千, 万,
億, 兆) for the stretch each word reads, its cheapest. The word costs what its
numerals cost and the connections between them, less one a connection, so that
the number said whole costs less than its parts; its class on the left is its
first numeral's, and on its right its last's. It is written in Arabic numerals
(45), in kanji (四十五), and, from ten thousand on, with its sections in Arabic
numerals and its units in kanji (1万, the number in Arabic numerals alone costing
one more). A digit read in kana and such a number are written in Arabic numerals
where a word of flag 4 follows, and in kanji elsewhere; a conversion within a
model's constraint may take either.
