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
- [zenz](https://huggingface.co/Miwa-Keita/zenz-v3.1-small), the GPT-2 of [Zenzai](https://github.com/azooKey/AzooKeyKanaKanjiConverter) for kana-kanji conversion, computed by KIRAL's own code (`kiral-nn`): its tokenizer, GPT-2's byte-level BPE; the network in single precision with a cache of keys and values; greedy decoding after zenz-v3's prompt. zenz-v3.1-small alone, with no dictionary:

  | | Items | Acc@1 | CER |
  | --- | ---: | ---: | ---: |
  | All | 200 | 87.0 | 1.58 |
  | With left context | 100 | 83.0 | 2.47 |
  | Without | 100 | 91.0 | 0.69 |

  Every one of the 200 conversions is the same as that of the network written in double precision with NumPy, apart from KIRAL's code.

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

The script fetches AJIMEE-Bench and zenz-v3.1-small from fixed revisions into `data/`, which is not committed, and checks them against the SHA-256 that `scripts/datasets.sha256` records; `list` shows where each comes from and its terms. The results of each run go to `bench/results/`, and `bench/baselines.toml` keeps the numbers that runs are compared with.

## License

KIRAL's code is under the GNU General Public License, version 3 or any later version (`LICENSES/GPL-3.0-or-later.txt`); every file names its copyright holder and license in the form of the [REUSE](https://reuse.software/) specification, which `reuse lint` checks.

Models and data are not part of the repository: they are separate files under their sources' licenses, fetched by whoever measures. zenz-v3.1-small is under CC BY-SA 4.0 (Keita Miwa, fine-tuned from ku-nlp's gpt2-small-japanese-char, CC BY-SA 4.0), and AJIMEE-Bench's data under CC BY-SA 3.0, from Kyoto University's Japanese Wikipedia Typo Dataset v2.
