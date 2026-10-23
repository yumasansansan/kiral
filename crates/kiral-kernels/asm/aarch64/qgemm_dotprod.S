// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later
//
// This file is part of KIRAL. The SPDX lines name its copyright holder and its
// license, in the machine-readable form of the REUSE specification: the GNU
// General Public License, version 3 or any later version
// (LICENSES/GPL-3.0-or-later.txt).
//
//     void kiral_qgemm_dotprod_R(size_t groups, const uint8_t *a,
//                                size_t a_stride, const float *a_floats,
//                                const uint8_t *b, float *c);
//         R = 1 to 4, AAPCS64
//
// The products of a linear layer in integers (src/quantized.rs) for AArch64's
// FEAT_DotProd -- Apple M1 and after, most Arm cores since 2018 -- whose sdot
// adds to each lane of 32 bits the four products of a column's four signed
// bytes by four signed bytes of the row, an element of another register. The
// inputs are unsigned, and so a row's bytes are made signed by flipping their
// highest bit, qa - 128, and the zero point moved by as much, the offset sa z
// becoming sa (z - 128):
//
//     sa sb sum(qa qb) - sa z ob/sb sb  =  sa sb sum((qa - 128) qb) - sa (z - 128) ob,
//
// ob being sb sum(qb), so that a group adds to c
//
//     c[i][j] += sa[i] sb[j] sum((qa - 128) qb) - (oa[i] - 128 sa[i]) ob[j],
//
// the sums exact in 32 bits as the other forms' are, the rest in single
// precision by fused multiply-adds, in another order than theirs.
//
// A quad of a panel, its sixteen columns of four bytes, is four registers (v0
// to v3, columns 4k to 4k + 3 in vk), loaded at once; a row's bytes are loaded
// sixteen at a time, four quads, flipped by the bytes 0x80 in v8 (v4 to v7, row
// i's in v(4 + i)), and each quad's sdot takes the row's element of it. Row i's
// sums of the sixteen columns take four registers; of three or four rows they
// are one set, v16 to v31, row i's from v(16 + 4i); of two rows, two sets, the
// even quads' and the odd quads', v16 to v23 and v24 to v31, row i's from
// v(16 + 4i) and v(24 + 4i); of one row, four, a quad's in the set of its
// number modulo four, from v16, v20, v24 and v28 -- twelve or sixteen sums,
// enough to keep four sdot units busy that each wait on their sum some three
// cycles. After the group the sets are added and the sums scaled and added to
// c, a row at a time: each converted, times sa, then sb added in times it, and
// ob times the moved offset taken away, sb and ob in v0 and v1, a row of c in
// v2, sa and the offset in s4 and s5, and 128 in each lane of v8.
//
// General registers, all of them the caller's to save: x0 the groups left, x1
// row 0's bytes and x2 a_stride, x3 row 0's floats, x4 the panel, read on, x5
// c; x10 to x13 the rows' bytes and x14 to x17 their floats; x9 the low half
// of v8, which the convention has the callee keep, put back before returning.
// x18 is not touched, nor sp.
//
// Not measured here: the i7-1360P this was written on runs no AArch64; the
// CI's runners of macOS run its tests and measure it (kiral-tools bench
// kernels).
//
// Assembled by Clang.

#include "common.h"

        .arch_extension dotprod
        .text

// The products of the quad at element \lane of v\i, the row's bytes, with its
// sixteen columns in v0 to v3, into the sums v\s0 to v\s3.
.macro DOT i, lane, s0, s1, s2, s3
        sdot v\s0\().4s, v0.16b, v\i\().4b[\lane]
        sdot v\s1\().4s, v1.16b, v\i\().4b[\lane]
        sdot v\s2\().4s, v2.16b, v\i\().4b[\lane]
        sdot v\s3\().4s, v3.16b, v\i\().4b[\lane]
.endm

// The quad at element \lane of the rows' sixteen bytes, of \rows rows: its
// columns loaded and its products added to the sums of its set. A quad's
// number modulo four is its element, so that its set is its element, or its
// element modulo two.
.macro QUAD lane, rows
        ld1 {v0.16b, v1.16b, v2.16b, v3.16b}, [x4], #64
.if \rows > 2
        DOT 4, \lane, 16, 17, 18, 19
        DOT 5, \lane, 20, 21, 22, 23
        DOT 6, \lane, 24, 25, 26, 27
.if \rows > 3
        DOT 7, \lane, 28, 29, 30, 31
.endif
.elseif \rows == 2
.if \lane % 2
        DOT 4, \lane, 24, 25, 26, 27
        DOT 5, \lane, 28, 29, 30, 31
.else
        DOT 4, \lane, 16, 17, 18, 19
        DOT 5, \lane, 20, 21, 22, 23
.endif
.else
.if \lane == 0
        DOT 4, 0, 16, 17, 18, 19
.elseif \lane == 1
        DOT 4, 1, 20, 21, 22, 23
.elseif \lane == 2
        DOT 4, 2, 24, 25, 26, 27
.else
        DOT 4, 3, 28, 29, 30, 31
.endif
.endif
.endm

// The next sixteen bytes of each row, its next four quads, flipped signed.
.macro ROWS rows
        ldr q4, [x10], #16
        eor v4.16b, v4.16b, v8.16b
.if \rows > 1
        ldr q5, [x11], #16
        eor v5.16b, v5.16b, v8.16b
.endif
.if \rows > 2
        ldr q6, [x12], #16
        eor v6.16b, v6.16b, v8.16b
.endif
.if \rows > 3
        ldr q7, [x13], #16
        eor v7.16b, v7.16b, v8.16b
.endif
.endm

// Zeros in the sums that a tile of \rows rows takes.
.macro ZERO rows
        movi v16.4s, #0
        movi v17.4s, #0
        movi v18.4s, #0
        movi v19.4s, #0
        movi v20.4s, #0
        movi v21.4s, #0
        movi v22.4s, #0
        movi v23.4s, #0
.if \rows != 3
        movi v24.4s, #0
        movi v25.4s, #0
        movi v26.4s, #0
        movi v27.4s, #0
        movi v28.4s, #0
        movi v29.4s, #0
        movi v30.4s, #0
        movi v31.4s, #0
.else
        movi v24.4s, #0
        movi v25.4s, #0
        movi v26.4s, #0
        movi v27.4s, #0
.endif
.endm

// The sets added into the first: of two rows the odd quads' into the even
// quads', of one row the four sets into the first.
.macro MERGE rows
.if \rows == 2
        add v16.4s, v16.4s, v24.4s
        add v17.4s, v17.4s, v25.4s
        add v18.4s, v18.4s, v26.4s
        add v19.4s, v19.4s, v27.4s
        add v20.4s, v20.4s, v28.4s
        add v21.4s, v21.4s, v29.4s
        add v22.4s, v22.4s, v30.4s
        add v23.4s, v23.4s, v31.4s
.elseif \rows == 1
        add v16.4s, v16.4s, v20.4s
        add v17.4s, v17.4s, v21.4s
        add v18.4s, v18.4s, v22.4s
        add v19.4s, v19.4s, v23.4s
        add v24.4s, v24.4s, v28.4s
        add v25.4s, v25.4s, v29.4s
        add v26.4s, v26.4s, v30.4s
        add v27.4s, v27.4s, v31.4s
        add v16.4s, v16.4s, v24.4s
        add v17.4s, v17.4s, v25.4s
        add v18.4s, v18.4s, v26.4s
        add v19.4s, v19.4s, v27.4s
.endif
.endm

// Columns 4k to 4k + 3 of row \row, its sums in v\s: converted, times sa, sb
// added in times them, the moved offset times ob taken away, into c.
.macro COLUMNS s, row, k
        ldr q0, [x4, #(16*\k)]
        ldr q1, [x4, #(64 + 16*\k)]
        scvtf v\s\().4s, v\s\().4s
        fmul v\s\().4s, v\s\().4s, v4.s[0]
        ldr q2, [x5, #(64*\row + 16*\k)]
        fmla v2.4s, v\s\().4s, v0.4s
        fmls v2.4s, v1.4s, v5.s[0]
        str q2, [x5, #(64*\row + 16*\k)]
.endm

// Row \row, its sums from v\s and its scale and offset at x\floats, scaled
// into c; the offset moved by 128 sa.
.macro SCALE s0, s1, s2, s3, row, floats
        ldp s4, s5, [x\floats]
        fmsub s5, s4, s8, s5
        COLUMNS \s0, \row, 0
        COLUMNS \s1, \row, 1
        COLUMNS \s2, \row, 2
        COLUMNS \s3, \row, 3
.endm

// The kernel of \rows rows.
.macro QGEMM rows
FUNCTION(kiral_qgemm_dotprod_\rows)
        fmov x9, d8
        mov x10, x1
        mov x14, x3
.if \rows > 1
        add x11, x10, x2
        add x15, x14, x2, lsr #3
.endif
.if \rows > 2
        add x12, x11, x2
        add x16, x15, x2, lsr #3
.endif
.if \rows > 3
        add x13, x12, x2
        add x17, x16, x2, lsr #3
.endif
1:
        // A group: its sums from zero, the sixteen quads, four at a time with
        // the rows' next sixteen bytes, the sets added, the sums into c.
        movi v8.16b, #0x80
        ZERO \rows
        .rept 4
        ROWS \rows
        QUAD 0, \rows
        QUAD 1, \rows
        QUAD 2, \rows
        QUAD 3, \rows
        .endr
        MERGE \rows
        // 128.0 in each lane: 0x43000000.
        movi v8.4s, #0x43, lsl #24
        SCALE 16, 17, 18, 19, 0, 14
.if \rows > 1
        SCALE 20, 21, 22, 23, 1, 15
.endif
.if \rows > 2
        SCALE 24, 25, 26, 27, 2, 16
.endif
.if \rows > 3
        SCALE 28, 29, 30, 31, 3, 17
.endif
        add x4, x4, #128
        add x14, x14, #8
.if \rows > 1
        add x15, x15, #8
.endif
.if \rows > 2
        add x16, x16, #8
.endif
.if \rows > 3
        add x17, x17, #8
.endif
        subs x0, x0, #1
        b.ne 1b
        fmov d8, x9
        ret
END_FUNCTION(kiral_qgemm_dotprod_\rows)
.endm

        QGEMM 1
        QGEMM 2
        QGEMM 3
        QGEMM 4

END_FILE
