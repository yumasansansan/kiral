<!--
SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
SPDX-License-Identifier: GPL-3.0-or-later

This file is part of KIRAL. The SPDX lines name its copyright holder and its
license, in the machine-readable form of the REUSE specification: the GNU
General Public License, version 3 or any later version
(LICENSES/GPL-3.0-or-later.txt).
-->

# KIRAL
A Japanese input method in the making: kana-kanji conversion whose candidates come from a dictionary lattice and whose choice among them is made by a small neural network, computed in integers on the CPU by kernels of its own, for Fcitx 5, Windows (TSF) and macOS (InputMethodKit).

English | [日本語](README.ja.md)

## What it is to be

- **No keystroke waits for the network.** The dictionary's conversion is shown at once; the network's, computed on a thread of its own, refines it when it arrives.
- **The accuracy of a neural converter**, measured on [AJIMEE-Bench](https://github.com/azooKey/AJIMEE-Bench), with the network's weights and inputs in 8-bit integers.
- **Light**: one process for the engine on each system, the model and the dictionary mapped from their files, nothing awake while nothing is typed, no GPU.
- **Portable**: every kernel has a form in plain Rust that is correct on any processor, and is the reference; assembly is only a faster form of the same computation, for a level of the instruction set the processor has, chosen at run time.

## What there is

So far, the harness that every later step is measured on:

- `kiral-tools eval` converts AJIMEE-Bench's 200 readings, gives Acc@1 and the character error rate (CER) of them all and of those with and without left context, compares the run item by item with the last one, with McNemar's exact p, and keeps its conversions for the next.
- [zenz](https://huggingface.co/Miwa-Keita/zenz-v3.1-small), the GPT-2 of [Zenzai](https://github.com/azooKey/AzooKeyKanaKanjiConverter) for kana-kanji conversion, computed by KIRAL's own code (`kiral-nn`): its tokenizer, GPT-2's byte-level BPE; the network in single precision and in integers, with a cache of keys and values; greedy decoding after zenz-v3's prompt. zenz-v3.1-small alone, with no dictionary:

  | | Items | Acc@1 | CER |
  | --- | ---: | ---: | ---: |
  | All | 200 | 87.0 | 1.58 |
  | With left context | 100 | 83.0 | 2.47 |
  | Without | 100 | 91.0 | 0.69 |

  Every one of the 200 conversions is the same as that of the network written in double precision with NumPy, apart from KIRAL's code.

  In integers (`--int8`): the weights of the linear layers and of the output layer quantized to bytes as the model is read, or once, into a model file (below), a scale for each output's group of 64 inputs, and their inputs quantized as they come, a scale and a zero point for each token's group (to seven bits for the form of AVX2 without VNNI, below; `kiral_kernels::quantized`, in the layout of stable-diffusion.assembly's `bits=8`), the products of a group summed exactly in 32 bits: Acc@1 87.0 and CER 1.58, the same items right as in single precision, one conversion of the 200 changed and wrong either way. `kiral-tools agree` feeds each item's prompt and single precision's conversion of it to both and compares the likeliest next token at each position of the conversion, what checking a draft depends on: 4395 of the 4396 positions agree.

  Fewer bits take more care. A scheme may rotate each block of a layer's inputs, as ConvRot does for the diffusion transformers, by a regular Hadamard matrix -- each of its rows and columns sums to the square root of its order, so that a row's large numbers spread rather than add up -- the weights when they are packed and the inputs online; fit each group's scale to its least squared error; and the output layer keeps eight bits. The positions of the conversions that agree with single precision, of 4396:

  | Weights | Agree | Acc@1 | CER |
  | --- | ---: | ---: | ---: |
  | 8 bits | 4395 (99.98%) | 87.0 | 1.58 |
  | 8 bits, rotated in blocks of 256 | 4395 (99.98%) | 87.5 | 1.55 |
  | 4 bits | 4367 (99.34%) | 81.5 | 2.20 |
  | 4 bits, scales fitted | 4359 (99.16%) | | |
  | 4 bits, rotated in blocks of 64, scales fitted | 4366 (99.32%) | | |
  | 4 bits, rotated in blocks of 256 | 4378 (99.59%) | 83.0 | 2.00 |
  | 4 bits, rotated in blocks of 256, scales fitted | 4377 (99.57%) | 83.0 | 2.06 |

  Eight bits lose nothing and need no rotation: they are what `--int8` takes. At four bits the rotation takes the positions that agree to 99.6% and Acc@1 back from 81.5 to 83.0 (the scales fitted besides add nothing to either), but not to single precision's 87.0: an item's conversion has some 22 positions and changes with any of them, and half the items that change are lost, so that Acc@1 within a point asks for some 99.9% of them, which only eight bits give. Four bits wait for a quantization calibrated on data.

  Timed with `kiral-tools bench model`, 50 items converted one after another on a pinned core, each prompt fed at once and each step of decoding after it, on the i7-1360P:

  | Two runs each | Prompts (37 tokens), p50 | p99 | Steps, p50 | p99 |
  | --- | ---: | ---: | ---: | ---: |
  | Single precision, P-core (CPU 0, 4.2–4.3 GHz) | 170–178 ms | 533–536 ms | 14.5 ms | 19.3–21.2 ms |
  | Integers, portable, P-core (4.3 GHz) | 349–362 ms | 1017–1019 ms | 13.0–13.1 ms | 16.9–17.2 ms |
  | Integers, AVX-VNNI, P-core (4.2–4.8 GHz) | 21.3–22.9 ms | 67.3–67.6 ms | 4.5–4.7 ms | 6.1–6.9 ms |
  | Single precision, E-core (CPU 8, 3.70 GHz) | 246–250 ms | 684–696 ms | 17.8–18.0 ms | 21.4–21.7 ms |
  | Integers, portable, E-core | 472–474 ms | 1361–1362 ms | 20.4 ms | 22.6–22.8 ms |
  | Integers, AVX-VNNI, E-core | 74.6–76.7 ms | 224–234 ms | 9.2–9.3 ms | 10.9–13.4 ms |

  With AVX-VNNI's `vpdpbusd`, the form the processor runs here, a prompt takes about an eighth of single precision's time on the P-core and a third on the E-core, and a step of decoding about a third and a half: a step reads every weight, 340 MB in single precision and a quarter of that in bytes, whereas a prompt's tokens are bound by the products, each some 0.5 ms more on the P-core. The portable form, for processors without VNNI, takes twice as long as single precision over a prompt and about as long over a step.

- KIRAL's model file ([docs/MODEL-FORMAT.md](docs/MODEL-FORMAT.md)): `kiral-tools model convert` writes zenz in integers into one file -- its configuration, its tokenizer, and its weights quantized and packed in the layout the kernels read, each tensor on a line of the caches -- with the license of its weights, their notices and a SHA-256 of itself, which `kiral-tools model check` checks. It is loaded by mapping it (`kiral-sys`): the weights are read where they lie, an embedding's row converted to single precision when it is looked up, and only the normalizations' and the biases' 0.5 MB are copied. zenz-v3.1-small's is 111.9 MB, against 180.9 MB of safetensors in bfloat16, and AJIMEE-Bench converts from it exactly as from the safetensors quantized as they are read: Acc@1 87.0, no conversion changed, 4395 of the 4396 positions agreeing with single precision. Timed with `kiral-tools bench load`, ten loads, each followed by the first item's prompt fed twice, on the i7-1360P:

  | Two runs each | Loads, p50 | First prompts after a load, p50 | Second prompts, p50 |
  | --- | ---: | ---: | ---: |
  | The model file, P-core (CPU 0, 4.5–4.6 GHz) | 0.56–0.57 ms | 42.8–43.1 ms | 40.7–40.9 ms |
  | The safetensors read and quantized, P-core (4.8 GHz) | 450–452 ms | 41.6–41.9 ms | 41.3–41.4 ms |
  | The model file, E-core (CPU 8, 3.70 GHz) | 1.31 ms | 154.6–155.0 ms | 151.4–152.1 ms |
  | The safetensors read and quantized, E-core | 621–627 ms | 152.2–153.1 ms | 150.4–151.5 ms |

  The prompt, of 71 tokens, reads every weight. The first after a load, which takes the mapped file's pages from the system's cache of the file as it touches them, is 2.2 to 3.2 ms longer than the second, against 0.3 to 1.8 ms after reading the safetensors. Converting `bench model`'s items, the mapped file is as fast as weights quantized into memory, within what the runs differ by: steps of decoding at 5.3–5.4 ms at the median on the P-core against 5.6–5.9 ms, and 10.0–11.1 ms on the E-core against 10.2–11.9 ms, the P-core's clock between 4.1 and 4.9 GHz at the median of the runs.

- Mozc, measured as it is used: `kiral-tools eval --mozc` types each reading in romaji into a Mozc of the run's own -- a profile made for the run and removed after, so that the user's own Mozc learns nothing -- a key at a time through `mozc_emacs_helper`, converts it with Space and cancels it, checking that what was typed is the reading. Mozc 2.29 has Acc@1 47.5 and CER 7.91 (43.0 / 10.29 with left context, which it does not take, 52.0 / 5.54 without). `kiral-tools replay --mozc --cpu p0 --cpu e0` times each key and each conversion, the run pinned to a performance core and then to an efficiency core (`kiral-sys`, `kiral-cpu`), with the clock read beside; on an i7-1360P, the times of answers through Mozc's IPC:

  | Mozc 2.29 | Keys, p50 | p99 | Conversions, p50 | p99 |
  | --- | ---: | ---: | ---: | ---: |
  | P-core (CPU 0, 4.76 GHz at the median) | 0.83 ms | 3.79 ms | 1.36 ms | 7.64 ms |
  | E-core (CPU 8, 3.70 GHz at the median) | 1.25 ms | 5.71 ms | 2.14 ms | 11.11 ms |

- KIRAL's dictionary (`kiral-dict`) and lattice (`kiral-lattice`): `kiral-tools dict build` makes a dictionary file of Mozc's open-source dictionary and the nouns of [SudachiDict](https://github.com/WorksApplications/SudachiDict) that Mozc's lacks -- their words, keyed by their readings in a double-array trie, and the costs of the connections between their classes -- which is read where it lies ([docs/DICT-FORMAT.md](docs/DICT-FORMAT.md)). SudachiDict's nouns take the class of Mozc's that their part of speech corresponds to, and the cost under which nine in ten of Mozc's words of that class lie. A reading's lattice holds every word whose reading is a stretch of it, and Viterbi finds its cheapest path. A number of more than a digit read in kana (よんじゅうご) is a word of the lattice too, made of the numerals the dictionary has for its words (四, 十, 五) and costing what they and their connections cost, less one a connection, so that it is read whole where they would read it; it is written in Arabic numerals or in kanji (45 or 四十五), and from ten thousand on with its units in kanji (100万). The cheapest path writes a number, a digit read in kana among them, in kanji unless a suffix follows it, as Mozc shows it (いち alone as 一, いちねん as 1年); the search within a model's constraint takes either. The dictionary alone, with no context, against AJIMEE-Bench's 200 items, and the oracle, the items that have an accepted conversion that is a path of their lattice:

  | The dictionary alone | Entries | File | Acc@1 | CER | Oracle |
  | --- | ---: | ---: | ---: | ---: | ---: |
  | Mozc's | 1,284,987 | 78.8 MB | 53.0 | 6.55 | 196 (98.0%) |
  | Mozc's, and SudachiDict's core nouns | 2,083,623 | 137.8 MB | 53.5 | 6.46 | 198 (99.0%) |
  | Mozc's, and all SudachiDict's nouns | 2,985,608 | 233.3 MB | 54.0 | 6.24 | 198 (99.0%) |

  With SudachiDict's core lists, which `--sudachi` reads unless told otherwise: 50.0 / 8.28 with left context, 57.0 / 4.64 without; 20 items right that Mozc 2.29 has wrong and 8 wrong that it has right (McNemar's p 0.036). Typed a kana at a time, each prefix converted from its lattice up, on the i7-1360P:

  | Mozc's, and SudachiDict's core nouns, two runs each | Kana, p50 | p99 | Conversions, p50 | p99 |
  | --- | ---: | ---: | ---: | ---: |
  | P-core (CPU 0, 4.50 GHz at the median) | 0.040 ms | 0.274–0.279 ms | 0.049–0.050 ms | 0.285–0.306 ms |
  | E-core (CPU 8, 3.70 GHz at the median) | 0.064–0.065 ms | 0.374–0.387 ms | 0.082–0.086 ms | 0.412–0.444 ms |

  The numbers cost a kana a fifth more than the lattice of words alone did, run in turn with it (0.034 ms at the median and 0.232–0.248 ms at the 99th percentile on the P-core, 0.053–0.054 and 0.328–0.330 ms on the E-core), and change none of the dictionary's conversions alone: they come into play where a model wants them.

- KIRAL's engine (`kiral-engine`): the dictionary's conversion is a draft that zenz verifies, as Zenzai does. The model is fed the prompt and the draft's tokens at once -- the keys and values of what the sequence shares with the one fed before kept -- and the draft stands where each of its tokens, and the end of text after them, is the model's likeliest. At the first that is not, what the model accepted and the token it wants instead become a constraint on the text, and the lattice is searched again for the cheapest path that keeps it (`Lattice::best_within`), round after round, until the model accepts the draft or wants a text no path has. The conversion is a path of the lattice whatever happens. zenz-v3.1-small in integers, from its model file, and the dictionary of Mozc's and SudachiDict's core nouns, against AJIMEE-Bench's 200 items, each converted as an explicit conversion with no deadline:

  | Rounds at most | Acc@1 | CER | Rounds an item |
  | ---: | ---: | ---: | ---: |
  | 1 | 73.0 | 3.53 | 1.00 |
  | 2 | 82.5 | 2.12 | 1.51 |
  | 3 | 84.5 | 1.81 | 1.74 |
  | 4 | 85.0 | 1.64 | 1.82 |
  | 8 | 86.0 | 1.68 | 1.90 |
  | 16 | 86.5 | 1.53 | 1.92 |

  No item takes more than 13 rounds, the one that writes 50cm three times. With them all, the model accepts the draft of 193 items, and each of them is zenz's own conversion; the 7 others end where the model wants a text the dictionary cannot write, and one of those zenz alone converts right (87.0): 夜駆け, which the dictionary lacks. No item is right that zenz alone has wrong. Before the lattice made numbers of more than a digit, five more items ended so -- 45, 1991, 50, 10, 2005 and 26 -- and the engine converted 84.0 with 8 rounds and with any more.

  Typed live (`kiral-tools replay`), a kana every 100 ms: the input thread drafts each key from its lattice, within the constraint of the last refinement, and posts the reading to a worker, which refines it by a round and gives it back unless a later key has made it stale -- it asks before each block of the model -- and Space waits up to 30 ms for a refinement of up to 8 rounds. 50 items, two runs each, on the i7-1360P on AC power (the energy preference `performance`), and on battery (`balance_power`) before the lattice made numbers, the worker on a core of its own of the same kind:

  | Two runs each | Kana, p50 | p99 | Refinements, p50 | p99 | Came before the next key | Space refined in time | Acc@1 of Space |
  | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
  | On AC, P-cores (typing on CPU 0, the worker on CPU 2 at 2.37–2.48 GHz at the median) | 0.138–0.145 ms | 0.45–0.47 ms | 12.5–12.8 ms | 39.5–41.6 ms | 1435 of 1435 | 50 of 50 | 86.0 |
  | On AC, E-cores (CPU 8, the worker on CPU 9 at 3.69–3.70 GHz) | 0.20–0.21 ms | 0.69–0.72 ms | 36.6–36.7 ms | 96.6–96.7 ms | 1332–1333 of 1435 | 42 of 50 | 82.0 |
  | On battery, P-cores (the worker at 1.80 GHz) | 0.45–0.47 ms | 1.27–1.50 ms | 42.9–43.4 ms | 95.5–96.6 ms | 1384–1387 of 1435 | 45 of 50 | 82.0 |
  | On battery, E-cores (the worker at 2.20 GHz) | 0.41–0.42 ms | 1.15–1.18 ms | 60.3–60.4 ms | 98.6–98.8 ms | 734–752 of 1435 | 24 of 50 | 64.0–66.0 |

  A key's work on the input thread stays within 2 ms at the 99th percentile while the model works, and within 0.5 ms on a P-core on AC power. A key's refinement feeds the model the reading's new kana, the tag after it and the whole draft again, as zenz's prompt puts the reading before the conversion: on a P-core on AC power it comes after 12.5 to 12.8 ms at the median, within the 15 ms that is the aim, every one before the next key, against 15.7 to 16.3 ms before GELU and the softmax took KIRAL's own exponential (run in turn with it), and on battery, whose cores keep a lower clock between keys, after about three times that. The same 50 items converted with no deadline are right for 43, and Space shows 43 of them on a P-core on AC power. Back to back, without the time between keys (`kiral-tools bench keys`), a key's refinement feeds the model 19.8 tokens on average and takes 12.2 to 12.3 ms at the median on a P-core (CPU 0, 4.1–4.2 GHz) and 38.0 to 40.7 ms on an E-core (CPU 8, 3.70 GHz). Its tokens are bound by the products, not by reading the weights: drafts fed together would each add their own tokens to every key's, and a draft fed in parts would add passes, so that one draft is fed, whole.

- The kernels in single precision (`kiral-kernels`): linear layers on weights packed in panels, layer normalization, GELU, softmax and attention over the cache, in plain Rust that the compiler turns into vector code. Each is tested against double precision, its error held to the bound that the analysis of its roundings gives, at lengths that are not multiples of the vectors' width. The exponential that GELU and the softmax take is computed there too -- the argument less its nearest multiple of ln 2, Cephes' polynomial for what is left, the power of two put in the exponent's bits -- within 1.42 u of double precision on every float of its range (u = 2^-24), so that their loops become vector code as well: taken from the system's library a value at a time, GELU's hyperbolic tangent was a sixth of a key's refinement on a P-core. GELU is computed as `x / (1 + exp(-2z))`, which `x (1 + tanh z) / 2` equals. The inputs of the products in integers are quantized by comparisons and by adding 2^23, which rounds them, in a third of the time that rounding by `round_ties_even` and a saturating conversion took, to the same bytes. `kiral-tools bench kernels` times each.
- The products in integers in assembly for AVX-VNNI (`crates/kiral-kernels/asm/x86_64/qgemm_avx_vnni.S`), the first form in assembly: tiles of one to six rows by sixteen columns, `vpdpbusd` adding four products of bytes to a lane, of three rows or fewer in two sets of sums so that twice the chains run side by side. Tested at every number of rows against double precision, as the portable form is, and run wherever the processor has AVX-VNNI and the environment allows it. For the output layer, each row keeps only its largest sum as the tiles come (`quantized::likeliest`), the logits of 6000 words never written out: what verification takes of them. With 16 to 40 rows, 179–205 GMAC/s on a P-core of the i7-1360P against the portable form's 8–10, and 46–48 on an E-core against 6.3–6.7 (`kiral-tools bench kernels`).
- The products for AVX2 without VNNI (`qgemm_avx2.S`), for the processors before AVX-VNNI and AVX-512's VNNI (Haswell to Comet Lake, Zen to Zen 3): `vpmaddubsw` sums two products of an input's byte by a weight's in a word, which two of 255 by 127 would pass, so that the inputs are quantized to seven bits for this form, the weights kept at eight -- one model file for every form, mapped as it lies -- and `vpmaddwd` by ones sums the words into the lanes `vpdpbusd` would. With 16 or 40 rows, 102–111 GMAC/s on a P-core of the i7-1360P and 23–26 on an E-core, 0.55 of AVX-VNNI's, against the portable form's 8–9 and 6.4–6.7; on a processor it is for, the AMD EPYC 7763 (Zen 3) of a CI runner, 79–84 with 4 to 40 rows against 6.6–6.8 (unpinned). The inputs of seven bits cost zenz-v3.1-small one item of AJIMEE-Bench (Acc@1 86.5, CER 1.60, 4395 of the 4396 positions agreeing with single precision, as with eight bits) and the engine the same item (86.0 with as many rounds as it takes). Held to this form on the same machine (`KIRAL_ISA=avx2`), a key's refinement typed live comes 17.5 to 18.1 ms after the key at the median on a P-core, every one before the next, against 12.5 to 12.8 with AVX-VNNI.
- The products for AVX-512 with VNNI (`qgemm_avx512_vnni.S`), Ice Lake's and Sapphire Rapids' server cores and after, Zen 4 and 5: a panel's quad is one vector of 512 bits, so that a row takes one `vpdpbusd` a quad for the sixteen columns, the sums of a few rows kept in two or four sets. Tested under Intel's SDE as Ice Lake and Sapphire Rapids, and on the CI's runners that have AVX-512. No processor here runs it; on the AMD EPYC 9V45 (Zen 5) of a CI runner, unpinned, 256–328 GMAC/s with 4 to 40 rows, against AVX-VNNI's 190–215 there and the portable form's 10–11 (`kiral-tools bench kernels`).
- The products for AArch64's dot products (`asm/aarch64/qgemm_dotprod.S`), Apple M1 and after: `sdot` adds to a lane a column's four signed bytes times four of a row's, the row's bytes made signed by flipping their highest bit and the zero point moved by 128 to match. Built and tested by the CI's runners of macOS: on the Apple M1 (virtual) of one, unpinned, 96–125 GMAC/s with 4 to 40 rows, against the portable form's 3.3–5.1.
- The levels of the instruction set (`kiral-cpu`), as std's detection finds them, and the choice among them that `KIRAL_ISA=<level>` and `KIRAL_KERNELS=portable` hold back.
- KIRAL's model file, safetensors and JSON read by hand, and SHA-256 written out (`kiral-model`), the parsers taking any bytes: what is not a file of theirs is an error, never a panic.

No crate is fetched from crates.io: the formats are parsed by hand, and Cargo.lock names the workspace's own crates only.

## Building and testing

The latest stable Rust, and LLVM 23: Clang drives the link, and LLD links, on every system (`.cargo/config.toml`); Clang assembles the kernels written in assembly, through their preprocessor and `asm/common.h` for ELF, COFF and Mach-O, and `llvm-ar` archives them (`crates/kiral-kernels/build.rs`; `KIRAL_CLANG` and `KIRAL_AR` name others than those on `PATH`). The builds for x86-64 are for x86-64-v2, so that the input method runs wherever its system does; the builds for macOS are for macOS 26 and later on Apple silicon.

```bash
cargo build --release --locked
```

```bash
bash ci/rust.sh
```

`ci/rust.sh` checks the layout and clippy (every warning an error) and runs the tests in the dev and release profiles. `bash ci/targets.sh test` builds the workspace for x86-64-v3 and for x86-64-v4 with vectors of 512 bits, and tests each where the processor has it; `bash ci/emulated.sh <sde64>` runs the tests under [Intel SDE](https://www.intel.com/content/www/us/en/developer/articles/tool/software-development-emulator.html) as processors from Nehalem to Arrow Lake.

## Measuring

```bash
python3 scripts/fetch_datasets.py fetch ajimee zenz-v3.1-small
```

```bash
cargo run --release --locked -p kiral-tools -- eval --model data/zenz-v3.1-small
```

```bash
cargo run --release --locked -p kiral-tools -- eval --model data/zenz-v3.1-small --int8
```

```bash
cargo run --release --locked -p kiral-tools -- agree --model data/zenz-v3.1-small --weight-bits 4 --rotate 256
```

```bash
cargo run --release --locked -p kiral-tools -- bench kernels --cpu p0 --cpu e0
```

```bash
cargo run --release --locked -p kiral-tools -- bench model --model data/zenz-v3.1-small --int8 --cpu p0 --cpu e0
```

```bash
cargo run --release --locked -p kiral-tools -- model convert --model data/zenz-v3.1-small --output data/models/zenz-v3.1-small-int8.kmdl --source "zenz-v3.1-small by Keita Miwa, fine-tuned from ku-nlp's gpt2-small-japanese-char: https://huggingface.co/Miwa-Keita/zenz-v3.1-small, revision 344612e331bf3ba752a9bfef6633c93c6b933ebb"
```

```bash
cargo run --release --locked -p kiral-tools -- eval --model data/models/zenz-v3.1-small-int8.kmdl
```

```bash
cargo run --release --locked -p kiral-tools -- bench load --model data/models/zenz-v3.1-small-int8.kmdl --cpu p0 --cpu e0
```

```bash
python3 scripts/fetch_datasets.py fetch mozc-dictionary sudachi-dictionary
```

```bash
cargo run --release --locked -p kiral-tools -- dict build --mozc data/mozc-dictionary --sudachi data/sudachi-dictionary --output data/dictionaries/mozc-sudachi.kdic
```

```bash
cargo run --release --locked -p kiral-tools -- eval --dict data/dictionaries/mozc-sudachi.kdic
```

```bash
cargo run --release --locked -p kiral-tools -- replay --dict data/dictionaries/mozc-sudachi.kdic --cpu p0 --cpu e0
```

```bash
cargo run --release --locked -p kiral-tools -- eval --model data/models/zenz-v3.1-small-int8.kmdl --dict data/dictionaries/mozc-sudachi.kdic
```

```bash
cargo run --release --locked -p kiral-tools -- replay --model data/models/zenz-v3.1-small-int8.kmdl --dict data/dictionaries/mozc-sudachi.kdic --limit 50 --cpu p0:p2 --cpu e0:e1
```

```bash
cargo run --release --locked -p kiral-tools -- bench keys --model data/models/zenz-v3.1-small-int8.kmdl --dict data/dictionaries/mozc-sudachi.kdic --limit 50 --cpu p0 --cpu e0
```

```bash
cargo run --release --locked -p kiral-tools -- eval --mozc
```

```bash
cargo run --release --locked -p kiral-tools -- replay --mozc --cpu p0 --cpu e0
```

The script fetches AJIMEE-Bench, zenz-v3.1-small, Mozc's dictionary and SudachiDict from fixed revisions into `data/`, which is not committed, and checks them against the SHA-256 that `scripts/datasets.sha256` records; `list` shows where each comes from and its terms. `eval` with both `--model` and `--dict` converts with KIRAL's engine, up to `--rounds` rounds (8), and `replay` with both types into it, the worker pinned to the processor after the colon -- on the i7-1360P, `p0` is CPU 0 and `p2` CPU 2, another core (`p1` is CPU 0's other thread). `bench keys` times the engine's refinement of each key back to back, without the time between keys, its clock steady. `model convert` writes zenz in integers to `data/models/`; `--source` names where its weights come from, for the file's notices, and `--model` takes the file wherever it takes zenz's directory (`agree` takes it as `--quantized`). `dict build` without `--sudachi` builds Mozc's dictionary alone, and `--lists small_lex.csv,core_lex.csv,notcore_lex.csv` all of SudachiDict's lists. The results of each run go to `bench/results/`, and `bench/baselines.toml` keeps the numbers that runs are compared with. Mozc is measured on Linux, with the package `emacs-mozc-bin` for its helper.

## License

KIRAL's code is under the GNU General Public License, version 3 or any later version (`LICENSES/GPL-3.0-or-later.txt`); every file names its copyright holder and license in the form of the [REUSE](https://reuse.software/) specification, which `reuse lint` checks.

Models and data are not part of the repository: they are separate files under their sources' licenses, fetched by whoever measures. zenz-v3.1-small is under CC BY-SA 4.0 (Keita Miwa, fine-tuned from ku-nlp's gpt2-small-japanese-char, CC BY-SA 4.0), and AJIMEE-Bench's data under CC BY-SA 3.0, from Kyoto University's Japanese Wikipedia Typo Dataset v2. Mozc's open-source dictionary is under IPAdic's terms, with Google's additions under Mozc's BSD-3-Clause license and the Okinawa dictionary's words in the public domain. SudachiDict is under the Apache License 2.0 (Works Applications), with what it holds of UniDic under UniDic's BSD-3-Clause license and its words from NEologd credited to their sources. A dictionary file built from them goes under the same terms, whose notices it holds; a model file made from zenz-v3.1-small goes under CC BY-SA 4.0 too, with the notices it holds: where its weights come from, and what was changed.
