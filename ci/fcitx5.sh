#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
#
# This file is part of KIRAL. The SPDX lines name its copyright holder and its
# license, in the machine-readable form of the REUSE specification: the GNU
# General Public License, version 3 or any later version
# (LICENSES/GPL-3.0-or-later.txt).
#
#   ci/fcitx5.sh
#
# Builds KIRAL's addon for Fcitx 5 (frontends/fcitx5) with Clang, every warning
# an error, KIRAL's C ABI built by cargo beside it, and runs its tests in a
# Fcitx 5 of their own, on Fcitx's test frontend. Fcitx 5's development files
# and test modules come from the system's packages: on Ubuntu,
# libfcitx5core-dev, libfcitx5config-dev, libfcitx5utils-dev and
# fcitx5-modules-dev. KIRAL_FCITX5_BUILD names the build's directory
# [build/fcitx5].
set -euo pipefail

cd "$(dirname "$0")/.."
build=${KIRAL_FCITX5_BUILD:-build/fcitx5}
cmake -S frontends/fcitx5 -B "$build" -G Ninja -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_BUILD_TYPE=Release
cmake --build "$build"
ctest --test-dir "$build" --output-on-failure
