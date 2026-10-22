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
- [Zenzai](https://github.com/azooKey/AzooKeyKanaKanjiConverter) のかな漢字変換用 GPT-2 である [zenz](https://huggingface.co/Miwa-Keita/zenz-v3.1-small) を，KIRAL 自身のコード（`kiral-nn`）で計算します．トークナイザ（GPT-2 のバイトレベル BPE），キーと値のキャッシュを持つ単精度と整数のネットワーク，zenz-v3 のプロンプトに続く貪欲デコードです．辞書を使わない zenz-v3.1-small 単体の結果：

  | | 項目数 | Acc@1 | CER |
  | --- | ---: | ---: | ---: |
  | 全体 | 200 | 87.0 | 1.58 |
  | 左文脈あり | 100 | 83.0 | 2.47 |
  | 左文脈なし | 100 | 91.0 | 0.69 |

  200 項目の変換は，KIRAL のコードとは別に NumPy で倍精度で書いた同じネットワークの変換とすべて一致します．

  整数（`--int8`）では，線形層と出力層の重みをモデルを読むときに（あるいは一度だけモデルファイルへ．下記）バイトへ量子化し（出力ごと・入力 64 個のグループごとにスケール），入力は来るたびに量子化します（トークンごと・グループごとにスケールとゼロ点．`kiral_kernels::quantized`，並びは stable-diffusion.assembly の `bits=8` と同じ）．グループの積は 32 ビットで正確に足し合わせます．Acc@1 87.0，CER 1.58 で，正解の項目は単精度と同じです（変わった変換は 200 のうち 1 つで，どちらでも誤り）．`kiral-tools agree` は各項目のプロンプトと単精度による変換を両方に与え，変換の各位置で次に最も確からしいトークンを比べます（ドラフトの検証が依存するものです）．4396 位置のうち 4395 が一致します．

  ビット数を減らすには工夫が要ります．方式は，ConvRot が拡散トランスフォーマーで行うように，層の入力の各ブロックを正則アダマール行列（各行・各列の和が次数の平方根になるので，行の大きな値が足し合わさらずに散らばる）で回転し（重みは詰めるときに，入力はその場で），各グループのスケールを二乗誤差が最小のものに当てはめられます．出力層は 8 ビットのまま残します．変換の位置で単精度と一致する数（4396 のうち）：

  | 重み | 一致 | Acc@1 | CER |
  | --- | ---: | ---: | ---: |
  | 8 ビット | 4395（99.98%） | 87.0 | 1.58 |
  | 8 ビット，256 ずつ回転 | 4395（99.98%） | 86.5 | 1.67 |
  | 4 ビット | 4366（99.32%） | 80.5 | 2.56 |
  | 4 ビット，スケールを当てはめ | 4362（99.23%） | | |
  | 4 ビット，64 ずつ回転，スケールを当てはめ | 4367（99.34%） | | |
  | 4 ビット，256 ずつ回転 | 4378（99.59%） | 82.0 | 2.07 |
  | 4 ビット，256 ずつ回転，スケールを当てはめ | 4376（99.55%） | 83.0 | 2.02 |

  8 ビットは何も失わず，回転も要りません．`--int8` はこれを使います．4 ビットでは回転によって一致する位置が 99.6% になり，Acc@1 も 80.5 から 82.0（スケールを当てはめると 83.0．位置の一致は増えません）に戻りますが，単精度の 87.0 には届きません．1 項目の変換には 22 ほどの位置があり，そのどれが違っても変わるので，変わった項目の半分は誤りになるので，Acc@1 を 1 ポイント以内に保つには位置の 99.9% ほどが要り，それを満たすのは 8 ビットだけです．4 ビットは，データで較正する量子化を待ちます．

  `kiral-tools bench model` による時間（50 項目を固定したコアで 1 つずつ変換し，プロンプトは一度に，その後のデコードは 1 ステップずつ．i7-1360P）：

  | 2 回ずつ | プロンプト（37 トークン）p50 | p99 | 1 ステップ p50 | p99 |
  | --- | ---: | ---: | ---: | ---: |
  | 単精度，P コア（CPU 0，4.3〜4.4 GHz） | 174〜182 ms | 482〜499 ms | 15.4〜15.9 ms | 19.8〜20.1 ms |
  | 整数，可搬な形，P コア（4.3 GHz） | 356〜375 ms | 1010〜1045 ms | 13.4〜14.4 ms | 16.2〜17.4 ms |
  | 整数，AVX-VNNI，P コア（5.0 GHz） | 27.3〜27.4 ms | 80.8〜82.0 ms | 5.4〜5.5 ms | 6.3〜6.8 ms |
  | 単精度，E コア（CPU 8，3.70 GHz） | 262〜266 ms | 739〜748 ms | 20.3〜20.9 ms | 23.7〜26.0 ms |
  | 整数，可搬な形，E コア | 487 ms | 1400〜1405 ms | 22.2〜23.3 ms | 24.9〜25.6 ms |
  | 整数，AVX-VNNI，E コア | 88.5〜88.6 ms | 265〜266 ms | 10.7〜11.3 ms | 12.2〜12.8 ms |

  このプロセッサが使う AVX-VNNI の `vpdpbusd` では，プロンプトは単精度の P コアで約 6 分の 1，E コアで約 3 分の 1 の時間で，デコードの 1 ステップは約 3 分の 1 と 2 分の 1 です（1 ステップはすべての重みを読みます．単精度で 340 MB，バイトではその 4 分の 1 です）．VNNI を持たないプロセッサ向けの可搬な形は，プロンプトでは単精度の 2 倍，1 ステップでは同じくらいの時間がかかります．

- KIRAL のモデルファイル（[docs/MODEL-FORMAT.md](docs/MODEL-FORMAT.md)）：`kiral-tools model convert` は整数の zenz を 1 つのファイルに書き出します．設定，トークナイザ，カーネルが読む並びに量子化して詰めた重み（各テンソルはキャッシュのラインの境界から始まります）に加えて，重みのライセンスとその表示，ファイル自身の SHA-256 を持ち，`kiral-tools model check` がそれを検査します．読み込みはファイルのマップです（`kiral-sys`）．重みはファイルのままの場所で読み，埋め込みの行は引くときに単精度に変換し，コピーするのは正規化とバイアスの 0.5 MB だけです．zenz-v3.1-small のファイルは 111.9 MB（bfloat16 の safetensors は 180.9 MB）で，AJIMEE-Bench の変換は safetensors を読むときに量子化したものと完全に同じです（Acc@1 87.0，変わった変換なし，単精度との一致は 4396 位置のうち 4395）．`kiral-tools bench load` による時間（10 回の読み込み，それぞれの後に最初の項目のプロンプトを 2 回与える．i7-1360P）：

  | 2 回ずつ | 読み込み p50 | 読み込み後の最初のプロンプト p50 | 2 回目のプロンプト p50 |
  | --- | ---: | ---: | ---: |
  | モデルファイル，P コア（CPU 0，4.6 GHz） | 0.46〜0.49 ms | 55.6〜55.9 ms | 54.9〜55.7 ms |
  | safetensors を読んで量子化，P コア（4.4〜4.9 GHz） | 467〜535 ms | 55.6〜59.0 ms | 55.5〜58.6 ms |
  | モデルファイル，E コア（CPU 8，3.70 GHz） | 1.23〜1.25 ms | 178.9〜180.0 ms | 178.1〜178.9 ms |
  | safetensors を読んで量子化，E コア | 637〜646 ms | 179.3〜180.9 ms | 178.8〜179.4 ms |

  71 トークンのプロンプトはすべての重みを読みます．読み込み後の最初のプロンプトは，マップしたファイルのページを触れるたびにシステムのファイルキャッシュから取り込みますが，2 回目との差は 0.2〜1.2 ms で，safetensors を読んだ後の差（0.1〜1.5 ms）と変わりません．`bench model` の項目の変換でも，マップしたファイルはメモリに量子化した重みと同じ速さで，差は実行ごとのばらつきの範囲です（デコードの 1 ステップの中央値は P コアで 5.3〜5.4 ms と 5.6〜5.9 ms，E コアで 10.0〜11.1 ms と 10.2〜11.9 ms．P コアのクロックは実行の中央値で 4.1〜4.9 GHz）．

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

- KIRAL のエンジン（`kiral-engine`）：辞書による変換を下書きとし，Zenzai と同じく zenz が検証します．プロンプトと下書きのトークンを一度にモデルへ与え（前に与えた列と共通する部分のキーと値は保持します），下書きの各トークンとその後の文末がどれもモデルの最も確からしいトークンなら，下書きをそのまま採ります．そうでない最初の位置では，モデルが受け入れた部分と代わりに望むトークンをテキストの制約とし，それを保つ最もコストの低い経路をラティスから探し直します（`Lattice::best_within`）．これを，モデルが下書きを受け入れるか，どの経路にもないテキストを望むまで繰り返します．変換は何があってもラティス上の経路です．モデルファイルから読んだ整数の zenz-v3.1-small と，Mozc と SudachiDict の core の名詞による辞書で，AJIMEE-Bench の 200 項目を締め切りのない明示的な変換として変換した結果：

  | 往復の上限 | Acc@1 | CER | 1 項目あたりの往復 |
  | ---: | ---: | ---: | ---: |
  | 1 | 72.0 | 3.52 | 1.00 |
  | 2 | 81.5 | 2.32 | 1.49 |
  | 3 | 83.5 | 1.96 | 1.69 |
  | 4 | 83.5 | 1.90 | 1.75 |
  | 8 | 84.0 | 1.91 | 1.79 |

  7 往復を超える項目はありません．186 項目ではモデルが下書きを受け入れ，そのどれもが zenz 自身の変換です．残る 14 項目は，辞書が書けないテキストをモデルが望むところで終わり，そのうち 6 項目は zenz 単体（87.0）なら正しく変換します．5 つは複数桁の数（45，1991，50，10，2005 と 26）で，辞書のかなで読む数字は 1 桁だけだからです．もう 1 つは辞書にない「夜駆け」です．zenz 単体が誤る項目を正すことはありません．

  打ちながらの変換（`kiral-tools replay`，100 ms ごとに 1 かな）：入力スレッドは各キーで，直前の精緻化が残した制約の中でラティスから下書きを作り，読みをワーカーに渡します．ワーカーはそれを 1 往復で精緻化して返します（後のキーで古くなったものは，モデルのブロックの合間に確かめて捨てます）．Space は最大 8 往復の精緻化を 30 ms まで待ちます．50 項目を 2 回ずつ，AC 電源（エネルギーの優先設定 `performance`）とバッテリー駆動（`balance_power`）の i7-1360P で，ワーカーを同じ種類の別のコアに置いて測りました：

  | 2 回ずつ | かな p50 | p99 | 精緻化 p50 | p99 | 次のキーまでに届いた数 | 締め切りまでに精緻化された Space | Space の Acc@1 |
  | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
  | AC，P コア（入力は CPU 0，ワーカーは CPU 2．クロックの中央値 2.69〜3.20 GHz） | 0.139 ms | 0.44〜0.47 ms | 15.1〜16.4 ms | 51.9〜53.0 ms | 1435 のうち 1435 | 50 のうち 50 | 84.0 |
  | AC，E コア（CPU 8，ワーカーは CPU 9．3.69〜3.70 GHz） | 0.18〜0.19 ms | 0.69〜0.72 ms | 39.5〜39.8 ms | 97.7〜98.2 ms | 1435 のうち 1236〜1243 | 50 のうち 38 | 78.0 |
  | バッテリー，P コア（ワーカーは 1.80 GHz） | 0.45〜0.47 ms | 1.27〜1.50 ms | 42.9〜43.4 ms | 95.5〜96.6 ms | 1435 のうち 1384〜1387 | 50 のうち 45 | 82.0 |
  | バッテリー，E コア（ワーカーは 2.20 GHz） | 0.41〜0.42 ms | 1.15〜1.18 ms | 60.3〜60.4 ms | 98.6〜98.8 ms | 1435 のうち 734〜752 | 50 のうち 24 | 64.0〜66.0 |

  モデルが動いている間も，入力スレッドでのキーの処理は 99 パーセンタイルで 2 ms に収まり，AC 電源の P コアでは 0.5 ms に収まります．zenz のプロンプトは読みを変換の前に置くので，キーの精緻化は読みの新しいかな，その後のタグ，下書き全体をモデルに与え直します．AC 電源の P コアでは中央値 15.1〜16.4 ms で届き（目標の 15 ms とほぼ同じで，すべて次のキーより前に届きます），キーの合間のクロックが低いバッテリー駆動ではその約 3 倍かかります．同じ 50 項目を締め切りなしで変換すると 42 項目が正解で，AC 電源の P コアで Space が示す数と同じです．

- 単精度のカーネル（`kiral-kernels`）：パネルにパックした重みによる線形層，LayerNorm，GELU，softmax，キャッシュに対する attention を，コンパイラがベクトル化するシンプルな Rust で書いています．どれも倍精度と比べてテストし，誤差を丸めの解析から導いた上界に収めています．長さはベクトルの幅の倍数でないものを使います．
- AVX-VNNI のアセンブリによる整数の積（`crates/kiral-kernels/asm/x86_64/qgemm_avx_vnni.S`）．最初のアセンブリの形です．1〜6 行×16 列のタイルで，`vpdpbusd` がバイトの積 4 つを 1 レーンに足します．3 行以下では和を 2 組に分け，2 倍の連鎖を並べて走らせます．可搬な形と同じく行数ごとに倍精度と比べてテストし，プロセッサが AVX-VNNI を持ち環境が許すところで使います．16〜40 行で，i7-1360P の P コアで 179〜205 GMAC/s（可搬な形は 8〜10），E コアで 46〜48（可搬な形は 6.3〜6.7）です（`kiral-tools bench kernels`）．
- 命令セットのレベル（`kiral-cpu`）：std の検出によるレベルと，その中からの選択です．`KIRAL_ISA=<レベル>` と `KIRAL_KERNELS=portable` で選択を抑えられます．
- 手書きの KIRAL のモデルファイル，safetensors，JSON の読み取りと SHA-256（`kiral-model`）：読み取りはどんなバイト列も受け付け，読めないものはパニックではなくエラーにします．

crates.io からは何も取得しません．形式はすべて手で解析し，Cargo.lock にはワークスペース自身のクレートしか載りません．

## ビルドとテスト

最新の stable の Rust と LLVM 23 を使います．どのシステムでも，リンクは Clang が駆動し LLD が行います（`.cargo/config.toml`）．アセンブリで書いたカーネルは，Clang がプリプロセッサと ELF・COFF・Mach-O 用の `asm/common.h` を通して組み立て，`llvm-ar` が束ねます（`crates/kiral-kernels/build.rs`．PATH 上のもの以外は `KIRAL_CLANG` と `KIRAL_AR` で指定します）．x86-64 向けのビルドは，システムが動くところならどこでも入力システムが動くよう x86-64-v2 向けです．macOS 向けは Apple シリコンの macOS 26 以降向けです．

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
cargo run --release --locked -p kiral-tools -- eval --mozc
```

```bash
cargo run --release --locked -p kiral-tools -- replay --mozc --cpu p0 --cpu e0
```

このスクリプトは AJIMEE-Bench，zenz-v3.1-small，Mozc の辞書，SudachiDict を固定のリビジョンから `data/`（コミットしない）に取得し，`scripts/datasets.sha256` に記録した SHA-256 と照合します．`list` で取得元と利用条件を表示します．`eval` は `--model` と `--dict` を両方与えると KIRAL のエンジンで変換し（往復の上限は `--rounds`，標準で 8），`replay` は両方を与えるとエンジンに打ち込みます．ワーカーはコロンの後のプロセッサに固定します（i7-1360P では `p0` が CPU 0，`p2` が別のコアの CPU 2 で，`p1` は CPU 0 のもう一つのスレッドです）．`model convert` は整数の zenz を `data/models/` に書き出します．`--source` は重みの出所で，ファイルの表示に入ります．`--model` は zenz のディレクトリを受け取るところならどこでもこのファイルを受け取ります（`agree` では `--quantized`）．`dict build` は，`--sudachi` なしでは Mozc の辞書だけで，`--lists small_lex.csv,core_lex.csv,notcore_lex.csv` では SudachiDict のすべてのリストを加えて作ります．各実行の結果は `bench/results/` に保存し，比較の基準となる数値は `bench/baselines.toml` にあります．Mozc の計測は Linux で，ヘルパーのためにパッケージ `emacs-mozc-bin` を入れて行います．

## ライセンス

KIRAL のコードは GNU General Public License バージョン 3 またはそれ以降のバージョン（`LICENSES/GPL-3.0-or-later.txt`）です．各ファイルは [REUSE](https://reuse.software/) 仕様の形式で著作権者とライセンスを示し，`reuse lint` で検査しています．

モデルとデータはリポジトリに含めません．それぞれの提供元のライセンスに従う別のファイルで，計測する人が取得します．zenz-v3.1-small は CC BY-SA 4.0（Keita Miwa 氏，ku-nlp の gpt2-small-japanese-char（CC BY-SA 4.0）からのファインチューニング），AJIMEE-Bench のデータは京都大学の日本語 Wikipedia 入力誤りデータセット (v2) に由来する CC BY-SA 3.0 です．Mozc のオープンソース辞書は IPAdic の条件に従い，Google の追加分は Mozc の BSD-3-Clause，沖縄辞書の語はパブリックドメインです．SudachiDict は Apache License 2.0（Works Applications）で，含まれる UniDic の部分は UniDic の BSD-3-Clause に従い，NEologd に由来する語はその出典を示しています．これらから作った辞書ファイルも同じ条件に従い，その表示をファイルの中に持ちます．zenz-v3.1-small から作ったモデルファイルも CC BY-SA 4.0 に従い，重みの出所と変更点の表示をファイルの中に持ちます．
