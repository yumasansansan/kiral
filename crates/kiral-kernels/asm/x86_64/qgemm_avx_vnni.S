// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later
//
// This file is part of KIRAL. The SPDX lines name its copyright holder and its
// license, in the machine-readable form of the REUSE specification: the GNU
// General Public License, version 3 or any later version
// (LICENSES/GPL-3.0-or-later.txt).
//
//     void kiral_qgemm_avxvnni_R(size_t groups, const uint8_t *a,
//                                size_t a_stride, const float *a_floats,
//                                const uint8_t *b, float *c);
//         R = 1 to 6, System V's convention on every system
//
// The products of a linear layer in integers (src/quantized.rs) for AVX-VNNI,
// the VEX form of vpdpbusd that Alder Lake's cores and those after have
// without AVX-512: each adds to a lane of 32 bits the four products of a quad
// of bytes, the inputs' unsigned and the weights' signed. A call adds to a tile
// of R rows by one panel's sixteen columns, c, its products with `groups`
// groups of 64 of the depth:
//
//     c[i][j] += sa[i] sb[j] sum(qa qb) - oa[i] ob[j]   for each group,
//
// as the portable form adds them, the sums exact, the rest in single
// precision by FMAs. The inputs are the quantized rows' bytes, row i's group
// g at a + i a_stride + 64 g (a_stride = 64 groups), and their floats, row i's
// group g's scale and offset at a_floats + i a_stride / 8 + 8 g; the weights
// are the panel, a group of 1152 bytes after another: sixteen quads of the
// sixteen columns' four bytes, then the columns' sixteen scales and sixteen
// offsets. c is R rows of sixteen floats, 64 bytes apart.
//
// This is stable-diffusion.assembly's quant_avx2.S, by the same author, made
// for KIRAL's rows: a quad loads the sixteen columns' four bytes in two
// vectors (ymm12, ymm13) and, for each row, broadcasts the row's four bytes
// (ymm14) for two vpdpbusd. Of six rows the twelve sums of a group fill ymm0
// to ymm11; of three rows or fewer, which leave registers over, the sums take
// two sets, the even quads' and the odd quads', added at the group's end, so
// that twice the chains of vpdpbusd run side by side -- one row's two chains
// alone would wait on vpdpbusd's latency at every quad. After the group the
// sums are scaled and added to c, a row at a time: sa sb in ymm12 and ymm13
// (sa in ymm14), the offsets' product subtracted by an FMA with oa in ymm15.
//
// General registers, of those the caller saves: rdi the groups left, rsi row
// 0's bytes of the group, rdx a_stride, rcx row 0's floats of the group, r8
// the panel's group, r9 c, r10 the floats' stride (a_stride / 8), and, of
// four rows or more, rax row 3's bytes and r11 row 3's floats. The kernels
// touch no other memory than their operands', and rsp not at all.
//
// Measured with kiral-tools bench kernels on zenz's five layers, their weights
// in the caches, two runs: on a P-core of the i7-1360P at 4.1 to 4.2 GHz, 179
// to 205 GMAC/s with 16 or 40 rows -- some 49 products a cycle of the 64 that
// two vpdpbusd a cycle make -- 114 to 186 with 4 rows and 43 to 108 with one,
// the portable form 8 to 10 with any; on an E-core at 3.7 GHz, 46 to 48 with
// 4 rows or more and 32 to 45 with one, against 6.3 to 6.7.
//
// Assembled by Clang, in Intel's syntax.

#include "common.h"

        .intel_syntax noprefix
        .text

// Row \base's quad \q broadcast, its products with the quad's sixteen columns
// added to ymm\lo (columns 0 to 7) and ymm\hi (8 to 15).
.macro DOT lo, hi, base, q
        vpbroadcastd ymm14, dword ptr [\base + 4*\q]
        {vex} vpdpbusd ymm\lo, ymm14, ymm12
        {vex} vpdpbusd ymm\hi, ymm14, ymm13
.endm

// The quad's sixteen columns.
.macro WEIGHTS q
        vmovdqu ymm12, [r8 + 64*\q]
        vmovdqu ymm13, [r8 + 64*\q + 32]
.endm

// A quad of three rows or fewer: row i's sums of the even quads in ymm(4i)
// and ymm(4i + 1), of the odd quads in ymm(4i + 2) and ymm(4i + 3).
.macro QUAD_TWO_SETS q, rows
        WEIGHTS \q
.if \q % 2
        DOT 2, 3, rsi, \q
.if \rows > 1
        DOT 6, 7, rsi+rdx, \q
.endif
.if \rows > 2
        DOT 10, 11, rsi+2*rdx, \q
.endif
.else
        DOT 0, 1, rsi, \q
.if \rows > 1
        DOT 4, 5, rsi+rdx, \q
.endif
.if \rows > 2
        DOT 8, 9, rsi+2*rdx, \q
.endif
.endif
.endm

// A quad of four rows or more: row i's sums in ymm(2i) and ymm(2i + 1).
.macro QUAD_ONE_SET q, rows
        WEIGHTS \q
        DOT 0, 1, rsi, \q
        DOT 2, 3, rsi+rdx, \q
        DOT 4, 5, rsi+2*rdx, \q
        DOT 6, 7, rax, \q
.if \rows > 4
        DOT 8, 9, rax+rdx, \q
.endif
.if \rows > 5
        DOT 10, 11, rax+2*rdx, \q
.endif
.endm

// Zeros in the sums that a tile of \rows rows takes: 4 \rows of them, or 12.
.macro ZERO rows
        vpxor xmm0, xmm0, xmm0
        vpxor xmm1, xmm1, xmm1
        vpxor xmm2, xmm2, xmm2
        vpxor xmm3, xmm3, xmm3
.if \rows > 1
        vpxor xmm4, xmm4, xmm4
        vpxor xmm5, xmm5, xmm5
        vpxor xmm6, xmm6, xmm6
        vpxor xmm7, xmm7, xmm7
.endif
.if \rows > 2
        vpxor xmm8, xmm8, xmm8
        vpxor xmm9, xmm9, xmm9
        vpxor xmm10, xmm10, xmm10
        vpxor xmm11, xmm11, xmm11
.endif
.endm

// The odd quads' sums added to the even quads', of three rows or fewer.
.macro MERGE rows
        vpaddd ymm0, ymm0, ymm2
        vpaddd ymm1, ymm1, ymm3
.if \rows > 1
        vpaddd ymm4, ymm4, ymm6
        vpaddd ymm5, ymm5, ymm7
.endif
.if \rows > 2
        vpaddd ymm8, ymm8, ymm10
        vpaddd ymm9, ymm9, ymm11
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
FUNCTION(kiral_qgemm_avxvnni_\rows)
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
.if \rows > 3
        ZERO 3
        .irp q, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
        QUAD_ONE_SET \q, \rows
        .endr
        SCALE 0, 1, rcx, 0
        SCALE 2, 3, rcx+r10, 1
        SCALE 4, 5, rcx+2*r10, 2
        SCALE 6, 7, r11, 3
.if \rows > 4
        SCALE 8, 9, r11+r10, 4
.endif
.if \rows > 5
        SCALE 10, 11, r11+2*r10, 5
.endif
        add rax, 64
        add r11, 8
.else
        ZERO \rows
        .irp q, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
        QUAD_TWO_SETS \q, \rows
        .endr
        MERGE \rows
        SCALE 0, 1, rcx, 0
.if \rows > 1
        SCALE 4, 5, rcx+r10, 1
.endif
.if \rows > 2
        SCALE 8, 9, rcx+2*r10, 2
.endif
.endif
        add rsi, 64
        add rcx, 8
        add r8, 1152
        dec rdi
        jnz 1b
        vzeroupper
        ret
END_FUNCTION(kiral_qgemm_avxvnni_\rows)
.endm

        QGEMM 1
        QGEMM 2
        QGEMM 3
        QGEMM 4
        QGEMM 5
        QGEMM 6

END_FILE
