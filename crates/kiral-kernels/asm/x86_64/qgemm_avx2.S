// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later
//
// This file is part of KIRAL. The SPDX lines name its copyright holder and its
// license, in the machine-readable form of the REUSE specification: the GNU
// General Public License, version 3 or any later version
// (LICENSES/GPL-3.0-or-later.txt).
//
//     void kiral_qgemm_avx2_R(size_t groups, const uint8_t *a,
//                             size_t a_stride, const float *a_floats,
//                             const uint8_t *b, float *c);
//         R = 1 to 6, System V's convention on every system
//
// The products of a linear layer in integers (src/quantized.rs) for AVX2
// without VNNI -- Haswell to Comet Lake and Skylake's server cores, Zen to
// Zen 3 -- on the inputs of seven bits that the quantization gives this form: vpmaddubsw multiplies each
// unsigned byte of an input by the signed byte of a weight and sums the pairs
// into words, which saturate past 32767 and so never do here, two products of
// 127 by 127 at most making 32258; vpmaddwd by ones sums the pairs of words
// into the lanes of 32 bits that vpdpbusd would have summed the quad into; and
// vpaddd adds them to the sums. The sums are exactly those of the other forms,
// and so are the outputs, as the scaling after each group is
// qgemm_avx_vnni.S's:
//
//     c[i][j] += sa[i] sb[j] sum(qa qb) - oa[i] ob[j]   for each group.
//
// This is stable-diffusion.assembly's quant_maddubs.S, by the same author,
// made for KIRAL's rows and tiles: three instructions for each of AVX-VNNI's
// one, of which the two multiplications take the multipliers of vector
// integers -- twice the products of an FMA's eight floats where a core issues
// two of them a cycle (Skylake's cores and those after, Zen 3), as many where
// it issues one. A quad loads the sixteen columns' four bytes in two vectors
// (ymm12, ymm13) and, for each row, broadcasts the row's four bytes (ymm14) for
// the two products (ymm15). The sums of up to six rows take ymm0 to ymm11,
// row i's in ymm(2i) and ymm(2i + 1); a sum waits only on its vpaddd, a cycle
// a quad, so that one set of sums is enough for any number of rows, the long
// multiplications of a quad running beside those of the next. The ones that
// vpmaddwd multiplies by are read from the constants, as no register is left.
// After a group the sums are scaled and added to c, a row at a time: sa sb in
// ymm12 and ymm13 (sa in ymm14), the offsets' product subtracted by an FMA with
// oa in ymm15.
//
// General registers, of those the caller saves: rdi the groups left, rsi row
// 0's bytes of the group, rdx a_stride, rcx row 0's floats of the group, r8
// the panel's group, r9 c, r10 the floats' stride (a_stride / 8), and, of four
// rows or more, rax row 3's bytes and r11 row 3's floats. The kernels touch no
// other memory than their operands' and the ones, and rsp not at all.
//
// Measured with kiral-tools bench kernels on zenz's five layers, their weights
// in the caches, two runs: on a P-core of the i7-1360P at 4.1 to 4.3 GHz, 102
// to 111 GMAC/s with 16 or 40 rows -- 0.55 of AVX-VNNI's 180 to 199, as its
// two multiplications a quad are AVX-VNNI's one -- and 44 to 84 with one, the
// portable form 8 to 9 with any; on an E-core at 3.7 GHz, 23 to 26 with any
// number of rows, against AVX-VNNI's 44 to 48 and the portable form's 6.4 to
// 6.7. The inputs of seven bits cost zenz-v3.1-small an item of AJIMEE-Bench's
// 200, 86.5 against 87.0, every one of 4396 positions of its conversions but
// one agreeing with single precision, as with eight.
//
// Assembled by Clang, in Intel's syntax.

#include "common.h"

        .intel_syntax noprefix

        CONSTANTS
        .p2align 5
.Lones:
        .rept 16
        .short 1
        .endr

        .text

// Row \base's quad \q broadcast, its products with the quad's sixteen columns
// added to ymm\lo (columns 0 to 7) and ymm\hi (8 to 15).
.macro DOT lo, hi, base, q
        vpbroadcastd ymm14, dword ptr [\base + 4*\q]
        vpmaddubsw ymm15, ymm14, ymm12
        vpmaddwd ymm15, ymm15, [rip + .Lones]
        vpaddd ymm\lo, ymm\lo, ymm15
        vpmaddubsw ymm15, ymm14, ymm13
        vpmaddwd ymm15, ymm15, [rip + .Lones]
        vpaddd ymm\hi, ymm\hi, ymm15
.endm

// A quad of \rows rows: row i's sums in ymm(2i) and ymm(2i + 1).
.macro QUAD q, rows
        vmovdqu ymm12, [r8 + 64*\q]
        vmovdqu ymm13, [r8 + 64*\q + 32]
        DOT 0, 1, rsi, \q
.if \rows > 1
        DOT 2, 3, rsi+rdx, \q
.endif
.if \rows > 2
        DOT 4, 5, rsi+2*rdx, \q
.endif
.if \rows > 3
        DOT 6, 7, rax, \q
.endif
.if \rows > 4
        DOT 8, 9, rax+rdx, \q
.endif
.if \rows > 5
        DOT 10, 11, rax+2*rdx, \q
.endif
.endm

// Zeros in the sums that a tile of \rows rows takes.
.macro ZERO rows
        vpxor xmm0, xmm0, xmm0
        vpxor xmm1, xmm1, xmm1
.if \rows > 1
        vpxor xmm2, xmm2, xmm2
        vpxor xmm3, xmm3, xmm3
.endif
.if \rows > 2
        vpxor xmm4, xmm4, xmm4
        vpxor xmm5, xmm5, xmm5
.endif
.if \rows > 3
        vpxor xmm6, xmm6, xmm6
        vpxor xmm7, xmm7, xmm7
.endif
.if \rows > 4
        vpxor xmm8, xmm8, xmm8
        vpxor xmm9, xmm9, xmm9
.endif
.if \rows > 5
        vpxor xmm10, xmm10, xmm10
        vpxor xmm11, xmm11, xmm11
.endif
.endm

// Row \row's sums in ymm\lo and ymm\hi scaled and added to its row of c, its
// scale and offset at \floats: c += sa sb sum - oa ob.
.macro SCALE lo, hi, floats, row
        vbroadcastss ymm14, dword ptr [\floats]
        vmulps ymm12, ymm14, [r8 + 1024]
        vmulps ymm13, ymm14, [r8 + 1056]
        vcvtdq2ps ymm\lo, ymm\lo
        vcvtdq2ps ymm\hi, ymm\hi
        vfmadd213ps ymm\lo, ymm12, [r9 + 64*\row]
        vfmadd213ps ymm\hi, ymm13, [r9 + 64*\row + 32]
        vbroadcastss ymm15, dword ptr [\floats + 4]
        vfnmadd231ps ymm\lo, ymm15, [r8 + 1088]
        vfnmadd231ps ymm\hi, ymm15, [r8 + 1120]
        vmovups [r9 + 64*\row], ymm\lo
        vmovups [r9 + 64*\row + 32], ymm\hi
.endm

// The kernel of \rows rows.
.macro QGEMM rows
FUNCTION(kiral_qgemm_avx2_\rows)
        mov r10, rdx
        shr r10, 3
.if \rows > 3
        lea rax, [rdx + 2*rdx]
        add rax, rsi
        lea r11, [r10 + 2*r10]
        add r11, rcx
.endif
        .p2align 4
1:
        // A group: its sums from zero, the sixteen quads, the sums into c.
        ZERO \rows
        .irp q, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
        QUAD \q, \rows
        .endr
        SCALE 0, 1, rcx, 0
.if \rows > 1
        SCALE 2, 3, rcx+r10, 1
.endif
.if \rows > 2
        SCALE 4, 5, rcx+2*r10, 2
.endif
.if \rows > 3
        SCALE 6, 7, r11, 3
.endif
.if \rows > 4
        SCALE 8, 9, r11+r10, 4
.endif
.if \rows > 5
        SCALE 10, 11, r11+2*r10, 5
.endif
.if \rows > 3
        add rax, 64
        add r11, 8
.endif
        add rsi, 64
        add rcx, 8
        add r8, 1152
        dec rdi
        jnz 1b
        vzeroupper
        ret
END_FUNCTION(kiral_qgemm_avx2_\rows)
.endm

        QGEMM 1
        QGEMM 2
        QGEMM 3
        QGEMM 4
        QGEMM 5
        QGEMM 6

END_FILE
