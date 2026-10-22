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

  In integers (`--int8`): the weights of the linear layers and of the output layer quantized to bytes as the model is read, a scale for each output's group of 64 inputs, and their inputs quantized as they come, a scale and a zero point for each token's group (`kiral_kernels::quantized`, in the layout of stable-diffusion.assembly's `bits=8`), the products of a group summed exactly in 32 bits: Acc@1 87.0 and CER 1.58, the same items right as in single precision, one conversion of the 200 changed and wrong either way. `kiral-tools agree` feeds each item's prompt and single precision's conversion of it to both and compares the likeliest next token at each position of the conversion, what checking a draft depends on: 4395 of the 4396 positions agree.

  Fewer bits take more care. A scheme may rotate each block of a layer's inputs, as ConvRot does for the diffusion transformers, by a regular Hadamard matrix -- each of its rows and columns sums to the square root of its order, so that a row's large numbers spread rather than add up -- the weights when they are packed and the inputs online; fit each group's scale to its least squared error; and the output layer keeps eight bits. The positions of the conversions that agree with single precision, of 4396:

  | Weights | Agree | Acc@1 | CER |
  | --- | ---: | ---: | ---: |
  | 8 bits | 4395 (99.98%) | 87.0 | 1.58 |
  | 8 bits, rotated in blocks of 256 | 4395 (99.98%) | 86.5 | 1.67 |
  | 4 bits | 4366 (99.32%) | 80.5 | 2.56 |
  | 4 bits, scales fitted | 4362 (99.23%) | | |
  | 4 bits, rotated in blocks of 64, scales fitted | 4367 (99.34%) | | |
  | 4 bits, rotated in blocks of 256 | 4378 (99.59%) | 82.0 | 2.07 |
  | 4 bits, rotated in blocks of 256, scales fitted | 4376 (99.55%) | 83.0 | 2.02 |

  Eight bits lose nothing and need no rotation: they are what `--int8` takes. At four bits the rotation takes the positions that agree to 99.6% and Acc@1 back from 80.5 to 82.0 (83.0 with the scales fitted, which add nothing to the positions), but not to single precision's 87.0: an item's conversion has some 22 positions and changes with any of them, and half the items that change are lost, so that Acc@1 within a point asks for some 99.9% of them, which only eight bits give. Four bits wait for a quantization calibrated on data.

  Timed with `kiral-tools bench model`, 50 items converted one after another on a pinned core, each prompt fed at once and each step of decoding after it, on the i7-1360P:

  | Two runs each | Prompts (37 tokens), p50 | p99 | Steps, p50 | p99 |
  | --- | ---: | ---: | ---: | ---: |
  | Single precision, P-core (CPU 0, 4.3–4.4 GHz) | 175–176 ms | 474–492 ms | 15.9–16.2 ms | 20.0–20.5 ms |
  | Integers, P-core | 351–368 ms | 1020–1042 ms | 13.5–14.3 ms | 17.1–17.7 ms |
  | Single precision, E-core (CPU 8, 3.70 GHz) | 246–254 ms | 688–700 ms | 20.9–21.1 ms | 26.5–27.2 ms |
  | Integers, E-core | 486–488 ms | 1399–1402 ms | 21.3–23.2 ms | 24.2–25.4 ms |

  In plain Rust the products in integers take twice as long as those in single precision over a prompt, and about as long over a step of decoding, which reads every weight, 340 MB in single precision and a quarter of that in bytes: the forms in assembly, AVX-VNNI's `vpdpbusd` first, come next, with these as their measure.

- Mozc, measured as it is used: `kiral-tools eval --mozc` types each reading in romaji into a Mozc of the run's own -- a profile made for the run and removed after, so that the user's own Mozc learns nothing -- a key at a time through `mozc_emacs_helper`, converts it with Space and cancels it, checking that what was typed is the reading. Mozc 2.29 has Acc@1 47.5 and CER 7.91 (43.0 / 10.29 with left context, which it does not take, 52.0 / 5.54 without). `kiral-tools replay --mozc --cpu p0 --cpu e0` times each key and each conversion, the run pinned to a performance core and then to an efficiency core (`kiral-sys`, `kiral-cpu`), with the clock read beside; on an i7-1360P, the times of answers through Mozc's IPC:

  | Mozc 2.29 | Keys, p50 | p99 | Conversions, p50 | p99 |
  | --- | ---: | ---: | ---: | ---: |
  | P-core (CPU 0, 4.76 GHz at the median) | 0.83 ms | 3.79 ms | 1.36 ms | 7.64 ms |
  | E-core (CPU 8, 3.70 GHz at the median) | 1.25 ms | 5.71 ms | 2.14 ms | 11.11 ms |

- KIRAL's dictionary (`kiral-dict`) and lattice (`kiral-lattice`): `kiral-tools dict build` makes a dictionary file of Mozc's open-source dictionary and the nouns of [SudachiDict](https://github.com/WorksApplications/SudachiDict) that Mozc's lacks -- their words, keyed by their readings in a double-array trie, and the costs of the connections between their classes -- which is read where it lies ([docs/DICT-FORMAT.md](docs/DICT-FORMAT.md)). SudachiDict's nouns take the class of Mozc's that their part of speech corresponds to, and the cost under which nine in ten of Mozc's words of that class lie. A reading's lattice holds every word whose reading is a stretch of it, and Viterbi finds its cheapest path; a digit read in kana is written in kanji, as Mozc shows it, unless a suffix follows it (いち alone as 一, いちねん as 1年). The dictionary alone, with no context, against AJIMEE-Bench's 200 items, and the oracle, the items that have an accepted conversion that is a path of their lattice:

  | The dictionary alone | Entries | File | Acc@1 | CER | Oracle |
  | --- | ---: | ---: | ---: | ---: | ---: |
  | Mozc's | 1,284,987 | 78.8 MB | 53.0 | 6.55 | 196 (98.0%) |
  | Mozc's, and SudachiDict's core nouns | 2,083,623 | 137.8 MB | 53.5 | 6.46 | 198 (99.0%) |
  | Mozc's, and all SudachiDict's nouns | 2,985,608 | 233.3 MB | 54.0 | 6.24 | 198 (99.0%) |

  With SudachiDict's core lists, which `--sudachi` reads unless told otherwise: 50.0 / 8.28 with left context, 57.0 / 4.64 without; 20 items right that Mozc 2.29 has wrong and 8 wrong that it has right (McNemar's p 0.036). Typed a kana at a time, each prefix converted from its lattice up, on the i7-1360P:

  | Mozc's, and SudachiDict's core nouns | Kana, p50 | p99 | Conversions, p50 | p99 |
  | --- | ---: | ---: | ---: | ---: |
  | P-core (CPU 0, 4.87 GHz at the median) | 0.032 ms | 0.235 ms | 0.040 ms | 0.242 ms |
  | E-core (CPU 8, 3.69 GHz at the median) | 0.055 ms | 0.330 ms | 0.071 ms | 0.358 ms |

- The kernels in single precision (`kiral-kernels`): linear layers on weights packed in panels, layer normalization, GELU, softmax and attention over the cache, in plain Rust that the compiler turns into vector code. Each is tested against double precision, its error held to the bound that the analysis of its roundings gives, at lengths that are not multiples of the vectors' width.
- The levels of the instruction set (`kiral-cpu`), as std's detection finds them, and the choice among them that `KIRAL_ISA=<level>` and `KIRAL_KERNELS=portable` hold back.
- safetensors and JSON read by hand (`kiral-model`), taking any bytes: what is not a file of theirs is an error, never a panic.

No crate is fetched from crates.io: the formats are parsed by hand, and Cargo.lock names the workspace's own crates only.

## Building and testing

The latest stable Rust, and LLVM 23: Clang drives the link, and LLD links, on every system (`.cargo/config.toml`). The builds for x86-64 are for x86-64-v2, so that the input method runs wherever its system does; the builds for macOS are for macOS 26 and later on Apple silicon.

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
cargo run --release --locked -p kiral-tools -- bench model --model data/zenz-v3.1-small --int8 --cpu p0 --cpu e0
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
cargo run --release --locked -p kiral-tools -- eval --mozc
```

```bash
cargo run --release --locked -p kiral-tools -- replay --mozc --cpu p0 --cpu e0
```

The script fetches AJIMEE-Bench, zenz-v3.1-small, Mozc's dictionary and SudachiDict from fixed revisions into `data/`, which is not committed, and checks them against the SHA-256 that `scripts/datasets.sha256` records; `list` shows where each comes from and its terms. `dict build` without `--sudachi` builds Mozc's dictionary alone, and `--lists small_lex.csv,core_lex.csv,notcore_lex.csv` all of SudachiDict's lists. The results of each run go to `bench/results/`, and `bench/baselines.toml` keeps the numbers that runs are compared with. Mozc is measured on Linux, with the package `emacs-mozc-bin` for its helper.

## License

KIRAL's code is under the GNU General Public License, version 3 or any later version (`LICENSES/GPL-3.0-or-later.txt`); every file names its copyright holder and license in the form of the [REUSE](https://reuse.software/) specification, which `reuse lint` checks.

Models and data are not part of the repository: they are separate files under their sources' licenses, fetched by whoever measures. zenz-v3.1-small is under CC BY-SA 4.0 (Keita Miwa, fine-tuned from ku-nlp's gpt2-small-japanese-char, CC BY-SA 4.0), and AJIMEE-Bench's data under CC BY-SA 3.0, from Kyoto University's Japanese Wikipedia Typo Dataset v2. Mozc's open-source dictionary is under IPAdic's terms, with Google's additions under Mozc's BSD-3-Clause license and the Okinawa dictionary's words in the public domain. SudachiDict is under the Apache License 2.0 (Works Applications), with what it holds of UniDic under UniDic's BSD-3-Clause license and its words from NEologd credited to their sources. A dictionary file built from them goes under the same terms, whose notices it holds.
