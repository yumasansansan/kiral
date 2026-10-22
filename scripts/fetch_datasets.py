#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
#
# This file is part of KIRAL. The SPDX lines name its copyright holder and its
# license, in the machine-readable form of the REUSE specification: the GNU
# General Public License, version 3 or any later version
# (LICENSES/GPL-3.0-or-later.txt).
"""Fetch what KIRAL is measured with, into data/: the benchmark and the models.

    python scripts/fetch_datasets.py list
    python scripts/fetch_datasets.py fetch ajimee [zenz-v3.1-small] [--record]

None of it is this project's. Each set comes under terms of its own, which `list`
shows with where they are written, and none of it may be committed (data/ is
ignored): models and data are files apart from the code, under their sources'
licenses. The engine itself never fetches anything; this script is for measuring.

Every file is taken from a fixed revision of its repository and checked against
the SHA-256 that scripts/datasets.sha256 records. Hugging Face publishes the
SHA-256 of the large files it keeps, and the script checks those against what it
publishes as well; for the others, which no source gives a digest of, the
recorded digests are those of a first fetch, which `fetch --record` writes, to be
committed.
"""

import argparse
import dataclasses
import hashlib
import os
import sys
import urllib.request
from pathlib import Path
from typing import Final

ROOT: Final = Path(__file__).resolve().parent.parent
DIGESTS: Final = ROOT / "scripts" / "datasets.sha256"
USER_AGENT: Final = "kiral-fetch-datasets"
CHUNK: Final = 1 << 20

AJIMEE_REVISION: Final = "401666cd56d1a570c2021798b64b6da4396bfd45"
ZENZ_V31_SMALL_REVISION: Final = "344612e331bf3ba752a9bfef6633c93c6b933ebb"


@dataclasses.dataclass(frozen=True)
class File:
    url: str
    name: str  # under data/<set>/
    published: str = ""  # the SHA-256 the source publishes, where it does


@dataclasses.dataclass(frozen=True)
class Dataset:
    name: str
    title: str
    source: str
    terms: str
    files: list[File]


def github(repository: str, revision: str, path: str, name: str) -> File:
    return File(f"https://raw.githubusercontent.com/{repository}/{revision}/{path}", name)


def hugging_face(repository: str, revision: str, path: str, published: str = "") -> File:
    return File(f"https://huggingface.co/{repository}/resolve/{revision}/{path}", path, published)


AJIMEE: Final = Dataset(
    name="ajimee",
    title="AJIMEE-Bench, JWTD_v2/v1: 200 readings in katakana and their accepted conversions, 100 with left context",
    source=f"https://github.com/azooKey/AJIMEE-Bench/tree/{AJIMEE_REVISION}",
    terms=(
        "The data is built from the Japanese Wikipedia Typo Dataset v2 of Kyoto University and follows its license, "
        "CC BY-SA 3.0, as the README fetched with it says; its evaluation code, which is not fetched, is CC0 1.0."
    ),
    files=[
        github("azooKey/AJIMEE-Bench", AJIMEE_REVISION, "JWTD_v2/v1/evaluation_items.json", "evaluation_items.json"),
        github("azooKey/AJIMEE-Bench", AJIMEE_REVISION, "README.md", "README.md"),
    ],
)

ZENZ_V31_SMALL: Final = Dataset(
    name="zenz-v3.1-small",
    title="zenz-v3.1-small: a GPT-2 of 91 M parameters for kana-kanji conversion, bfloat16 safetensors",
    source=f"https://huggingface.co/Miwa-Keita/zenz-v3.1-small/tree/{ZENZ_V31_SMALL_REVISION}",
    terms="CC BY-SA 4.0, as the model card says. What is made from it, an int8 model file among them, is too.",
    files=[
        hugging_face(
            "Miwa-Keita/zenz-v3.1-small",
            ZENZ_V31_SMALL_REVISION,
            "model.safetensors",
            "e167291de8e276ec30c5753a58fa5de39222a3829ff3ff433d347499f17b5e8f",
        ),
        *(
            hugging_face("Miwa-Keita/zenz-v3.1-small", ZENZ_V31_SMALL_REVISION, path)
            for path in (
                "config.json",
                "generation_config.json",
                "merges.txt",
                "special_tokens_map.json",
                "tokenizer_config.json",
                "vocab.json",
                "README.md",
            )
        ),
    ],
)

DATASETS: Final = {dataset.name: dataset for dataset in (AJIMEE, ZENZ_V31_SMALL)}


def read_digests() -> dict[str, str]:
    digests: dict[str, str] = {}
    if DIGESTS.exists():
        for line in DIGESTS.read_text(encoding="utf-8").splitlines():
            if line.strip():
                digest, name = line.split(maxsplit=1)
                digests[name.strip()] = digest
    return digests


def write_digests(digests: dict[str, str]) -> None:
    lines = [f"{digest}  {name}" for name, digest in sorted(digests.items())]
    DIGESTS.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, target: Path) -> None:
    """Writes what url answers into target, through a partial file that only a whole answer replaces it with."""
    if not url.startswith("https://"):
        message = f"not fetched over anything but HTTPS: {url}"
        raise ValueError(message)
    partial = target.with_name(target.name + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})  # noqa: S310 -- HTTPS, checked above
    with urllib.request.urlopen(request, timeout=60) as response, partial.open("wb") as stream:  # noqa: S310 -- HTTPS
        expected = response.headers.get("Content-Length")
        written = 0
        while chunk := response.read(CHUNK):
            stream.write(chunk)
            written += len(chunk)
    if expected is not None and written != int(expected):
        partial.unlink()
        message = f"{url} answered {written} bytes of {expected}"
        raise OSError(message)
    partial.replace(target)


def fetch(dataset: Dataset, digests: dict[str, str], *, record: bool) -> int:
    directory = ROOT / "data" / dataset.name
    directory.mkdir(parents=True, exist_ok=True)
    failures = 0
    for file in dataset.files:
        key = f"{dataset.name}/{file.name}"
        target = directory / file.name
        if key not in digests and not record:
            print(
                f"error: scripts/datasets.sha256 records no digest for {key}. If this is the first fetch, "
                "fetch with --record, and commit the digests it writes."
            )
            return 1
        if not (target.exists() and sha256_of(target) == digests.get(key, file.published or None)):
            print(f"fetching {file.url}")
            download(file.url, target)
        digest = sha256_of(target)
        if file.published and digest != file.published:
            print(f"error: {key} has the SHA-256 {digest}, not the {file.published} its source publishes")
            failures += 1
        elif key not in digests:
            digests[key] = digest
            print(f"recorded {key}")
        elif digest != digests[key]:
            print(f"error: {key} has the SHA-256 {digest}, not the {digests[key]} recorded")
            failures += 1
    if not failures:
        print(f"{dataset.name}: {len(dataset.files)} files in {directory.relative_to(ROOT).as_posix()}")
    return failures


def show(dataset: Dataset) -> None:
    print(f"{dataset.name}: {dataset.title}")
    print(f"  source: {dataset.source}")
    print(f"  terms:  {dataset.terms}")
    print()


def main() -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list", help="the sets, where they come from, and their terms")
    get = commands.add_parser("fetch", help="fetch sets into data/")
    get.add_argument("names", nargs="+", choices=list(DATASETS))
    get.add_argument("--record", action="store_true", help="record the digests of files that have none")
    options = parser.parse_args()

    if options.command == "list":
        for dataset in DATASETS.values():
            show(dataset)
        return 0
    digests = read_digests()
    failures = 0
    for name in options.names:
        show(DATASETS[name])
        failures += fetch(DATASETS[name], digests, record=options.record)
    if options.record:
        write_digests(digests)
        print(f"the digests are in {os.path.relpath(DIGESTS, ROOT)}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
