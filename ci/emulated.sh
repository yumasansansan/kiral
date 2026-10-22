#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
#
# This file is part of KIRAL. The SPDX lines name its copyright holder and its
# license, in the machine-readable form of the REUSE specification: the GNU
# General Public License, version 3 or any later version
# (LICENSES/GPL-3.0-or-later.txt).
#
#   ci/emulated.sh <sde64> [model...]
#
# Runs the workspace's tests under Intel's Software Development Emulator, as
# processors the machine running it may not be: SDE reports through CPUID what
# the processor it emulates has, and stops at any instruction that processor
# would not have run. Cargo runs each test program under it (the target's
# runner). On Linux, for x86-64. Only the test programs are checked, which hold
# all of KIRAL and Rust's standard library: the system's own libraries may be
# built for more than an old model has -- Ubuntu's amd64v3 variant builds glibc
# for x86-64-v3 -- which is the system's matter, not KIRAL's; a model the
# system's own programs cannot run as is left out, and said to be.
#
# Each model runs the build that is shipped, for x86-64-v2, with
# KIRAL_EXPECT_LEVELS naming the levels of the kernels the model has, which
# kiral-cpu's test of detection compares with what std's detection finds there;
# a run that SDE stops shows that the build took an instruction above its
# baseline. The models that have what x86-64-v3 and v4 need run those builds as
# well (ci/targets.sh), whose portable kernels are vector code of 256 and 512
# bits.
#
# The models, by default: Nehalem (x86-64-v2 and nothing more), Haswell (AVX2),
# Skylake-SP (AVX-512 without VNNI), Ice Lake-SP (AVX512-VNNI without the VEX
# form), Alder Lake (the VEX form without AVX-512), Sapphire Rapids (both) and
# Arrow Lake (the VEX form, and AVX-VNNI-INT8).
set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "usage: ci/emulated.sh <sde64> [model...]" >&2
    exit 2
fi
sde=$(realpath "$1")
shift
models=("$@")
if [[ ${#models[@]} -eq 0 ]]; then
    models=(nhm hsw skx icx adl spr arl)
fi
triple=x86_64-unknown-linux-gnu

# Each model: the levels it has, and the builds above x86-64-v2 it runs.
levels_of() {
    case "$1" in
    nhm) echo "v2" ;;
    hsw | skx) echo "v2,avx2" ;;
    icx) echo "v2,avx2,avx512-vnni" ;;
    adl | arl) echo "v2,avx2,avx-vnni" ;;
    spr) echo "v2,avx2,avx-vnni,avx512-vnni" ;;
    *) return 1 ;;
    esac
}
builds_of() {
    case "$1" in
    nhm) echo "" ;;
    hsw | adl | arl) echo "v3" ;;
    skx | icx | spr) echo "v3 v4" ;;
    *) return 1 ;;
    esac
}
flags_of() {
    case "$1" in
    v2) echo '[]' ;;
    v3) echo '["-C", "target-cpu=x86-64-v3"]' ;;
    v4) echo '["-C", "target-cpu=x86-64-v4", "-C", "target-feature=-prefer-256-bit"]' ;;
    esac
}

status=0
for model in "${models[@]}"; do
    levels=$(levels_of "$model") || {
        echo "error: no levels known for the model $model" >&2
        exit 2
    }
    if ! refusal=$("$sde" "-$model" -chip_check_exe_only 1 -- /bin/true 2>&1); then
        echo "== $model: left out, the system's own programs do not run as it: ${refusal%%$'\n'*}"
        continue
    fi
    for build in v2 $(builds_of "$model"); do
        arguments=(--release --locked --workspace --lib --bins --tests)
        if [[ $build == v2 ]]; then
            target_dir=target
        else
            arguments+=(--target "$triple")
            target_dir=target/targets
        fi
        echo "== $model, the build for x86-64-$build"
        if ! KIRAL_EXPECT_LEVELS=$levels cargo test "${arguments[@]}" --target-dir "$target_dir" \
            --config "target.$triple.rustflags=$(flags_of "$build")" \
            --config "target.$triple.runner=[\"$sde\", \"-$model\", \"-chip_check_exe_only\", \"1\", \"--\"]" \
            -- --test-threads 4; then
            echo "error: the tests failed on $model in the build for x86-64-$build" >&2
            status=1
        fi
    done
done
exit "$status"
