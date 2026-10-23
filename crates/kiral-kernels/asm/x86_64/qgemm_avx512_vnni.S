// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later
//
// This file is part of KIRAL. The SPDX lines name its copyright holder and its
// license, in the machine-readable form of the REUSE specification: the GNU
// General Public License, version 3 or any later version
// (LICENSES/GPL-3.0-or-later.txt).
//
//     void kiral_qgemm_avx512vnni_R(size_t groups, const uint8_t *a,
//                                   size_t a_stride, const float *a_floats,
//                                   const uint8_t *b, float *c);
//         R = 1 to 6, System V's convention on every system
//
// The products of a linear layer in integers (src/quantized.rs) for AVX-512
// with VNNI -- Ice Lake and Sapphire Rapids and the server cores after them,
// Zen 4 and 5 -- as qgemm_avx_vnni.S computes them, at 512 bits: a panel's
// quad, sixteen columns of four bytes, is one vector (zmm12), so that a row's
// four bytes broadcast (zmm13) take one vpdpbusd for the sixteen columns where
// AVX-VNNI takes two, and a row's sums one register:
//
//     c[i][j] += sa[i] sb[j] sum(qa qb) - oa[i] ob[j]   for each group.
//
// vpdpbusd waits on its own sum, some four or five cycles, and a core issues
// one or two of them a cycle; so that enough of them run side by side, the
// sums are kept in sets, a quad's in the set of its number: of four rows or
// more two sets, the even quads' and the odd quads', and of three rows or
// fewer four, a quad's in the set of its number modulo four -- eight to twelve
// sums in all, two to four of a single row -- added together at the group's
// end. Row i's sums of set s are in zmm(6s + i) of two sets, zmm(3s + i) of
// four. After the group the sums are scaled and added to c, a row at a time:
// the sixteen scales sb in zmm28 and offsets ob in zmm29, the row's scale and
// offset broadcast from memory by the instructions that take them.
//
// General registers, of those the caller saves: rdi the groups left, rsi row
// 0's bytes of the group, rdx a_stride, rcx row 0's floats of the group, r8
// the panel's group, r9 c, r10 the floats' stride (a_stride / 8), and, of four
// rows or more, rax row 3's bytes and r11 row 3's floats. The kernels touch no
// other memory than their operands', and rsp not at all.
//
// Measured by the CI with kiral-tools bench kernels on zenz's five layers,
// unpinned, its clock not read, on the AMD EPYC 9V45 (Zen 5) of a runner of
// Windows (2026-10-23): 256 to 328 GMAC/s with 4 to 40 rows and 98 to 126 with
// one, against AVX-VNNI's 190 to 215 and 95 to 126 on the same runner and the
// portable form's 10 to 11. The i7-1360P this was written on has no AVX-512;
// Intel's SDE checks the kernel's results as Ice Lake and Sapphire Rapids
// (ci/emulated.sh).
//
// Assembled by Clang, in Intel's syntax.

#include "common.h"

        .intel_syntax noprefix
        .text

// Row \base's quad \q broadcast, its products with the quad's sixteen columns
// added to zmm\sum.
.macro DOT sum, base, q
        vpbroadcastd zmm13, dword ptr [\base + 4*\q]
        vpdpbusd zmm\sum, zmm13, zmm12
.endm

// A quad's products for \rows rows into the sums \s0 to \s5 of its set.
.macro ROWS q, rows, s0, s1, s2, s3, s4, s5
        DOT \s0, rsi, \q
.if \rows > 1
        DOT \s1, rsi+rdx, \q
.endif
.if \rows > 2
        DOT \s2, rsi+2*rdx, \q
.endif
.if \rows > 3
        DOT \s3, rax, \q
.endif
.if \rows > 4
        DOT \s4, rax+rdx, \q
.endif
.if \rows > 5
        DOT \s5, rax+2*rdx, \q
.endif
.endm

// Quad \q of \rows rows, into the set of its number.
.macro QUAD q, rows
        vmovdqu64 zmm12, [r8 + 64*\q]
.if \rows > 3
.if \q % 2
        ROWS \q, \rows, 6, 7, 8, 9, 10, 11
.else
        ROWS \q, \rows, 0, 1, 2, 3, 4, 5
.endif
.else
.if \q % 4 == 0
        ROWS \q, \rows, 0, 1, 2, 0, 0, 0
.elseif \q % 4 == 1
        ROWS \q, \rows, 3, 4, 5, 0, 0, 0
.elseif \q % 4 == 2
        ROWS \q, \rows, 6, 7, 8, 0, 0, 0
.else
        ROWS \q, \rows, 9, 10, 11, 0, 0, 0
.endif
.endif
.endm

// Zeros in the sums of every set that a tile of \rows rows takes.
.macro ZERO rows
        vpxord xmm0, xmm0, xmm0
.if \rows > 1
        vpxord xmm1, xmm1, xmm1
.endif
.if \rows > 2
        vpxord xmm2, xmm2, xmm2
.endif
.if \rows > 3
        vpxord xmm3, xmm3, xmm3
.endif
.if \rows > 4
        vpxord xmm4, xmm4, xmm4
.endif
.if \rows > 5
        vpxord xmm5, xmm5, xmm5
.endif
.if \rows > 3
        vpxord xmm6, xmm6, xmm6
        vpxord xmm7, xmm7, xmm7
        vpxord xmm8, xmm8, xmm8
.if \rows > 4
        vpxord xmm10, xmm10, xmm10
.endif
.if \rows > 5
        vpxord xmm11, xmm11, xmm11
.endif
        vpxord xmm9, xmm9, xmm9
.else
        vpxord xmm3, xmm3, xmm3
        vpxord xmm6, xmm6, xmm6
        vpxord xmm9, xmm9, xmm9
.if \rows > 1
        vpxord xmm4, xmm4, xmm4
        vpxord xmm7, xmm7, xmm7
        vpxord xmm10, xmm10, xmm10
.endif
.if \rows > 2
        vpxord xmm5, xmm5, xmm5
        vpxord xmm8, xmm8, xmm8
        vpxord xmm11, xmm11, xmm11
.endif
.endif
.endm

// The sets of row \r added into its sums of set 0, zmm\r: of four rows or
// more the odd quads' zmm(r + 6), of three or fewer the other three sets'.
.macro MERGE_ROW rows, r, s1, s2, s3
.if \rows > 3
        vpaddd zmm\r, zmm\r, zmm\s1
.else
        vpaddd zmm\r, zmm\r, zmm\s1
        vpaddd zmm\s2, zmm\s2, zmm\s3
        vpaddd zmm\r, zmm\r, zmm\s2
.endif
.endm

.macro MERGE rows
.if \rows > 3
        MERGE_ROW \rows, 0, 6, 0, 0
        MERGE_ROW \rows, 1, 7, 0, 0
        MERGE_ROW \rows, 2, 8, 0, 0
        MERGE_ROW \rows, 3, 9, 0, 0
.if \rows > 4
        MERGE_ROW \rows, 4, 10, 0, 0
.endif
.if \rows > 5
        MERGE_ROW \rows, 5, 11, 0, 0
.endif
.else
        MERGE_ROW \rows, 0, 3, 6, 9
.if \rows > 1
        MERGE_ROW \rows, 1, 4, 7, 10
.endif
.if \rows > 2
        MERGE_ROW \rows, 2, 5, 8, 11
.endif
.endif
.endm

// Row \row's sums in zmm\sum scaled and added to its row of c, its scale and
// offset at \floats: c += sa sb sum - oa ob.
.macro SCALE sum, floats, row
        vmulps zmm14, zmm28, dword ptr [\floats]{1to16}
        vcvtdq2ps zmm\sum, zmm\sum
        vfmadd213ps zmm\sum, zmm14, [r9 + 64*\row]
        vfnmadd231ps zmm\sum, zmm29, dword ptr [\floats + 4]{1to16}
        vmovups [r9 + 64*\row], zmm\sum
.endm

// The kernel of \rows rows.
.macro QGEMM rows
FUNCTION(kiral_qgemm_avx512vnni_\rows)
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
        // A group: its sums from zero, the sixteen quads, the sets added, the
        // sums into c.
        ZERO \rows
        .irp q, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
        QUAD \q, \rows
        .endr
        MERGE \rows
        vmovups zmm28, [r8 + 1024]
        vmovups zmm29, [r8 + 1088]
        SCALE 0, rcx, 0
.if \rows > 1
        SCALE 1, rcx+r10, 1
.endif
.if \rows > 2
        SCALE 2, rcx+2*r10, 2
.endif
.if \rows > 3
        SCALE 3, r11, 3
.endif
.if \rows > 4
        SCALE 4, r11+r10, 4
.endif
.if \rows > 5
        SCALE 5, r11+2*r10, 5
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
END_FUNCTION(kiral_qgemm_avx512vnni_\rows)
.endm

        QGEMM 1
        QGEMM 2
        QGEMM 3
        QGEMM 4
        QGEMM 5
        QGEMM 6

END_FILE
