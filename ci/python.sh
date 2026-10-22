#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
#
# This file is part of KIRAL. The SPDX lines name its copyright holder and its
# license, in the machine-readable form of the REUSE specification: the GNU
# General Public License, version 3 or any later version
# (LICENSES/GPL-3.0-or-later.txt).
#
#   ci/python.sh
#
# Checks the Python of the repository, which is scripts/ alone, with ruff
# (ruff.toml): its rules, and its layout. RUFF names the program where it is
# not ruff on PATH.
set -euo pipefail

ruff=${RUFF:-ruff}
"$ruff" --version
"$ruff" check scripts
"$ruff" format --check scripts
