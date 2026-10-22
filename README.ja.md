<!--
SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
SPDX-License-Identifier: GPL-3.0-or-later

このファイルは README.md の日本語版で，KIRAL の一部です．SPDX の行は，著作権者とライセンスを REUSE 仕様の機械可読な形式で示しています：GNU General Public License バージョン 3 またはそれ以降のバージョン（LICENSES/GPL-3.0-or-later.txt）．
-->

# KIRAL
開発中の日本語入力システムです．かな漢字変換の候補を辞書のラティスから作り，その中からの選択を小さなニューラルネットワークが行います．ネットワークは独自のカーネルで CPU 上で整数演算により計算します．対象は Fcitx 5，Windows（TSF），macOS（InputMethodKit）です．

[English](README.md) | 日本語

## 目指すもの

- **キー入力はネットワークを待たない．** 辞書による変換を即座に表示し，別スレッドで計算したネットワークの変換が届いたらそれで置き換えます．
- **ニューラル変換の精度**を，ネットワークの重みと入力を 8 ビット整数にしたまま実現します．精度は [AJIMEE-Bench](https://github.com/azooKey/AJIMEE-Bench) で測ります．
- **軽さ**：エンジンは各システムで 1 プロセス，モデルと辞書はファイルからマップし，入力していない間は何も起きていません．GPU は使いません．
- **移植性**：どのカーネルにも，どのプロセッサでも正しく動くシンプルな Rust の形があり，それが基準です．アセンブリは同じ計算の速い形にすぎず，プロセッサが持つ命令セットのレベルごとに用意し，実行時に選びます．

## 現在あるもの

今後のすべての段階を測るための土台です．

- `kiral-tools eval` は AJIMEE-Bench の 200 の読みを変換し，全体と左文脈の有無別に Acc@1 と文字誤り率（CER）を求め，前回の実行と項目ごとに比べて McNemar の正確検定の p を示し，次回のために変換結果を保存します．
- [Zenzai](https://github.com/azooKey/AzooKeyKanaKanjiConverter) のかな漢字変換用 GPT-2 である [zenz](https://huggingface.co/Miwa-Keita/zenz-v3.1-small) を，KIRAL 自身のコード（`kiral-nn`）で計算します．トークナイザ（GPT-2 のバイトレベル BPE），キーと値のキャッシュを持つ単精度のネットワーク，zenz-v3 のプロンプトに続く貪欲デコードです．辞書を使わない zenz-v3.1-small 単体の結果：

  | | 項目数 | Acc@1 | CER |
  | --- | ---: | ---: | ---: |
  | 全体 | 200 | 87.0 | 1.58 |
  | 左文脈あり | 100 | 83.0 | 2.47 |
  | 左文脈なし | 100 | 91.0 | 0.69 |

  200 項目の変換は，KIRAL のコードとは別に NumPy で倍精度で書いた同じネットワークの変換とすべて一致します．

- Mozc は，使われるとおりに測ります．`kiral-tools eval --mozc` は，実行のためだけの Mozc（実行ごとに作って後で消すプロファイル．利用者自身の Mozc には何も学習させません）に，`mozc_emacs_helper` を通して読みをローマ字で 1 キーずつ打ち込み，打ったものが読みと一致することを確かめてから Space で変換し，取り消します．Mozc 2.29 は Acc@1 47.5，CER 7.91（左文脈あり 43.0 / 10.29．Mozc は左文脈を受け取りません．左文脈なし 52.0 / 5.54）です．`kiral-tools replay --mozc --cpu p0 --cpu e0` は，実行を P コアに，次に E コアに固定し（`kiral-sys`，`kiral-cpu`），クロックを読みながら，各キーと各変換の時間を測ります．i7-1360P での，Mozc の IPC を通した応答の時間：

  | Mozc 2.29 | キー p50 | p99 | 変換 p50 | p99 |
  | --- | ---: | ---: | ---: | ---: |
  | P コア（CPU 0，クロックの中央値 4.76 GHz） | 0.83 ms | 3.79 ms | 1.36 ms | 7.64 ms |
  | E コア（CPU 8，クロックの中央値 3.70 GHz） | 1.25 ms | 5.71 ms | 2.14 ms | 11.11 ms |

- KIRAL の辞書（`kiral-dict`）とラティス（`kiral-lattice`）：`kiral-tools dict build` は，Mozc のオープンソース辞書と，Mozc の辞書にない [SudachiDict](https://github.com/WorksApplications/SudachiDict) の名詞から辞書ファイルを作ります．語を読みのダブル配列トライで引けるようにし，品詞の連接コストの表を持たせたもので，ファイルのままの場所で読みます（[docs/DICT-FORMAT.md](docs/DICT-FORMAT.md)）．SudachiDict の名詞には，品詞が対応する Mozc の品詞と，Mozc のその品詞の語の 9 割がそれ以下に収まるコストを与えます．読みのラティスは読みのどこかに当たるすべての語を持ち，Viterbi で最もコストの低い経路を求めます．かなで読む数字は，Mozc が表示するとおり，後ろに接尾辞が続かなければ漢数字で書きます（いちだけなら一，いちねんなら 1年）．文脈を使わない辞書だけの変換の，AJIMEE-Bench の 200 項目に対する結果と，許容される変換のどれかがラティス上の経路になっている項目の数（オラクル）：

  | 辞書だけの変換 | 項目数 | ファイル | Acc@1 | CER | オラクル |
  | --- | ---: | ---: | ---: | ---: | ---: |
  | Mozc の辞書 | 1,284,987 | 78.8 MB | 53.0 | 6.55 | 196（98.0%） |
  | Mozc の辞書と SudachiDict の core の名詞 | 2,083,623 | 137.8 MB | 53.5 | 6.46 | 198（99.0%） |
  | Mozc の辞書と SudachiDict のすべての名詞 | 2,985,608 | 233.3 MB | 54.0 | 6.24 | 198（99.0%） |

  `--sudachi` が標準で読む SudachiDict の core のリストを加えたとき，左文脈あり 50.0 / 8.28，なし 57.0 / 4.64 です．Mozc 2.29 が誤る項目のうち 20 を正し，正しい項目のうち 8 を誤ります（McNemar の p 0.036）．1 かなずつ打ち，そのたびに読みの先頭からラティスを作り直して変換したときの時間（i7-1360P）：

  | Mozc の辞書と SudachiDict の core の名詞 | かな p50 | p99 | 変換 p50 | p99 |
  | --- | ---: | ---: | ---: | ---: |
  | P コア（CPU 0，クロックの中央値 4.87 GHz） | 0.032 ms | 0.235 ms | 0.040 ms | 0.242 ms |
  | E コア（CPU 8，クロックの中央値 3.69 GHz） | 0.055 ms | 0.330 ms | 0.071 ms | 0.358 ms |

- 単精度のカーネル（`kiral-kernels`）：パネルにパックした重みによる線形層，LayerNorm，GELU，softmax，キャッシュに対する attention を，コンパイラがベクトル化するシンプルな Rust で書いています．どれも倍精度と比べてテストし，誤差を丸めの解析から導いた上界に収めています．長さはベクトルの幅の倍数でないものを使います．
- 命令セットのレベル（`kiral-cpu`）：std の検出によるレベルと，その中からの選択です．`KIRAL_ISA=<レベル>` と `KIRAL_KERNELS=portable` で選択を抑えられます．
- 手書きの safetensors と JSON の読み取り（`kiral-model`）：どんなバイト列も受け付け，読めないものはパニックではなくエラーにします．

crates.io からは何も取得しません．形式はすべて手で解析し，Cargo.lock にはワークスペース自身のクレートしか載りません．

## ビルドとテスト

最新の stable の Rust と LLVM 23 を使います．どのシステムでも，リンクは Clang が駆動し LLD が行います（`.cargo/config.toml`）．x86-64 向けのビルドは，システムが動くところならどこでも入力システムが動くよう x86-64-v2 向けです．macOS 向けは Apple シリコンの macOS 26 以降向けです．

```bash
cargo build --release --locked
```

```bash
bash ci/rust.sh
```

`ci/rust.sh` はレイアウトと clippy（警告はすべてエラー）を検査し，dev と release の両プロファイルでテストを実行します．`bash ci/targets.sh test` は x86-64-v3 向けと，512 ビット幅のベクトルの x86-64-v4 向けにワークスペースをビルドし，プロセッサが対応していればそれぞれテストします．`bash ci/emulated.sh <sde64>` は，[Intel SDE](https://www.intel.com/content/www/us/en/developer/articles/tool/software-development-emulator.html) の下で Nehalem から Arrow Lake までのプロセッサとしてテストを実行します．

## 計測

```bash
python3 scripts/fetch_datasets.py fetch ajimee zenz-v3.1-small
```

```bash
cargo run --release --locked -p kiral-tools -- eval --model data/zenz-v3.1-small
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

このスクリプトは AJIMEE-Bench，zenz-v3.1-small，Mozc の辞書，SudachiDict を固定のリビジョンから `data/`（コミットしない）に取得し，`scripts/datasets.sha256` に記録した SHA-256 と照合します．`list` で取得元と利用条件を表示します．`dict build` は，`--sudachi` なしでは Mozc の辞書だけで，`--lists small_lex.csv,core_lex.csv,notcore_lex.csv` では SudachiDict のすべてのリストを加えて作ります．各実行の結果は `bench/results/` に保存し，比較の基準となる数値は `bench/baselines.toml` にあります．Mozc の計測は Linux で，ヘルパーのためにパッケージ `emacs-mozc-bin` を入れて行います．

## ライセンス

KIRAL のコードは GNU General Public License バージョン 3 またはそれ以降のバージョン（`LICENSES/GPL-3.0-or-later.txt`）です．各ファイルは [REUSE](https://reuse.software/) 仕様の形式で著作権者とライセンスを示し，`reuse lint` で検査しています．

モデルとデータはリポジトリに含めません．それぞれの提供元のライセンスに従う別のファイルで，計測する人が取得します．zenz-v3.1-small は CC BY-SA 4.0（Keita Miwa 氏，ku-nlp の gpt2-small-japanese-char（CC BY-SA 4.0）からのファインチューニング），AJIMEE-Bench のデータは京都大学の日本語 Wikipedia 入力誤りデータセット (v2) に由来する CC BY-SA 3.0 です．Mozc のオープンソース辞書は IPAdic の条件に従い，Google の追加分は Mozc の BSD-3-Clause，沖縄辞書の語はパブリックドメインです．SudachiDict は Apache License 2.0（Works Applications）で，含まれる UniDic の部分は UniDic の BSD-3-Clause に従い，NEologd に由来する語はその出典を示しています．これらから作った辞書ファイルも同じ条件に従い，その表示をファイルの中に持ちます．
