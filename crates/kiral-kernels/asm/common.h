// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later
//
// This file is part of KIRAL. The SPDX lines name its copyright holder and its
// license, in the machine-readable form of the REUSE specification: the GNU
// General Public License, version 3 or any later version
// (LICENSES/GPL-3.0-or-later.txt).
//
// What the kernels in assembly need to be one file for every system: Clang's
// preprocessor runs on each .S before its assembler does, and these macros
// write what the object formats want differently.
//
//     FUNCTION(name)      a global function, aligned to 32 bytes: the name with
//                         the leading underscore of Mach-O where it is one, its
//                         type on ELF and COFF, and endbr64 where the build
//                         marks indirect branches (-fcf-protection);
//     END_FUNCTION(name)  its size, on ELF;
//     END_FILE            the stack not executable, on ELF, which otherwise
//                         takes an object without the note for one that needs
//                         an executable stack;
//     CONSTANTS           the section of constants that the kernels read:
//                         .rodata on ELF, __TEXT,__const on Mach-O, .rdata on
//                         COFF.
//
// Nothing else in the kernels may be of one format alone.

#ifndef KIRAL_ASM_COMMON_H
#define KIRAL_ASM_COMMON_H

#if defined(__x86_64__) && defined(__CET__) && (__CET__ & 1)
#define KIRAL_BRANCH_TARGET endbr64
#else
#define KIRAL_BRANCH_TARGET
#endif

#if defined(__ELF__)
#define FUNCTION(name) \
    .globl name;       \
    .type name, @function; \
    .p2align 5;        \
    name:              \
    KIRAL_BRANCH_TARGET
#define END_FUNCTION(name) .size name, . - name
#define END_FILE .section .note.GNU-stack, "", @progbits
#define CONSTANTS .section .rodata
#elif defined(__APPLE__)
#define FUNCTION(name) \
    .globl _##name;    \
    .p2align 5;        \
    _##name:           \
    KIRAL_BRANCH_TARGET
#define END_FUNCTION(name)
#define END_FILE
#define CONSTANTS .section __TEXT, __const
#elif defined(_WIN32)
#define FUNCTION(name) \
    .globl name;       \
    .def name;         \
    .scl 2;            \
    .type 32;          \
    .endef;            \
    .p2align 5;        \
    name:              \
    KIRAL_BRANCH_TARGET
#define END_FUNCTION(name)
#define END_FILE
#define CONSTANTS .section .rdata, "dr"
#else
#error "an object format the kernels are not written for"
#endif

#endif
