// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later
//
// This file is part of KIRAL. The SPDX lines name its copyright holder and its
// license, in the machine-readable form of the REUSE specification: the GNU
// General Public License, version 3 or any later version
// (LICENSES/GPL-3.0-or-later.txt).

//! The processors a thread may run on.
//!
//! On Linux, `sched_setaffinity` and `sched_getaffinity` of the calling thread,
//! with masks of 1024 processors, glibc's `cpu_set_t`; a process the thread
//! starts afterwards starts with its mask. On Windows, `SetThreadAffinityMask` of
//! the calling thread, within its group of 64 processors. macOS lets no thread
//! choose its processors (a thread's quality of service is all it takes), and
//! neither do the other systems here.

use std::io;

/// The words of a mask: 1024 processors, as glibc's `cpu_set_t` holds.
#[cfg(target_os = "linux")]
const WORDS: usize = 16;

#[cfg(target_os = "linux")]
mod linux {
    unsafe extern "C" {
        /// Sets the processors that the thread `pid` (0, the calling one) may run
        /// on, from a mask of `size` bytes; 0, or -1 with `errno` set.
        pub fn sched_setaffinity(pid: i32, size: usize, mask: *const u64) -> i32;
        /// Writes the mask of the processors the thread `pid` (0, the calling one)
        /// may run on into `size` bytes; 0, or -1 with `errno` set.
        pub fn sched_getaffinity(pid: i32, size: usize, mask: *mut u64) -> i32;
    }
}

#[cfg(windows)]
mod windows {
    use std::ffi::c_void;

    #[link(name = "kernel32")]
    unsafe extern "system" {
        /// The pseudo-handle of the calling thread, which needs no closing.
        pub safe fn GetCurrentThread() -> *mut c_void;
        /// Sets the processors of the thread's group that it may run on; the
        /// mask it had, or 0 with the last error set.
        pub fn SetThreadAffinityMask(thread: *mut c_void, mask: usize) -> usize;
    }
}

/// Lets the calling thread run on the processor `cpu` alone, as the system
/// numbers its processors.
///
/// # Errors
///
/// Where the system refuses -- a processor it does not have, or one the
/// process may not use -- or lets no thread choose its processors.
pub fn pin_current_thread(cpu: usize) -> io::Result<()> {
    #[cfg(target_os = "linux")]
    {
        if cpu >= WORDS * 64 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("no processor {cpu} in a mask"),
            ));
        }
        let mut mask = [0u64; WORDS];
        mask[cpu / 64] |= 1 << (cpu % 64);
        // SAFETY: `mask` is the `size_of_val(&mask)` bytes the call is told it reads,
        // and lives across the call; pid 0 is the calling thread.
        let result = unsafe { linux::sched_setaffinity(0, size_of_val(&mask), mask.as_ptr()) };
        if result == 0 {
            Ok(())
        } else {
            Err(io::Error::last_os_error())
        }
    }
    #[cfg(windows)]
    {
        let bit = u32::try_from(cpu)
            .ok()
            .and_then(|cpu| 1usize.checked_shl(cpu))
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, format!("no processor {cpu} in a group")))?;
        // SAFETY: the handle is the calling thread's pseudo-handle, valid for the
        // thread's whole life; the mask is a value.
        let previous = unsafe { windows::SetThreadAffinityMask(windows::GetCurrentThread(), bit) };
        if previous == 0 {
            Err(io::Error::last_os_error())
        } else {
            Ok(())
        }
    }
    #[cfg(not(any(target_os = "linux", windows)))]
    {
        Err(io::Error::new(
            io::ErrorKind::Unsupported,
            format!("this system lets no thread choose its processors (asked for {cpu})"),
        ))
    }
}

/// The processors the calling thread may run on, in their order.
///
/// # Errors
///
/// Where the system refuses, or does not say (on any system but Linux).
pub fn current_thread_processors() -> io::Result<Vec<usize>> {
    #[cfg(target_os = "linux")]
    {
        let mut mask = [0u64; WORDS];
        // SAFETY: `mask` is the `size_of_val(&mask)` bytes the call is told it may
        // write, and lives across the call; pid 0 is the calling thread.
        let result = unsafe { linux::sched_getaffinity(0, size_of_val(&mask), mask.as_mut_ptr()) };
        if result != 0 {
            return Err(io::Error::last_os_error());
        }
        Ok((0..WORDS * 64)
            .filter(|&cpu| mask[cpu / 64] & (1 << (cpu % 64)) != 0)
            .collect())
    }
    #[cfg(not(target_os = "linux"))]
    {
        Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "the processors of a thread are read on Linux only",
        ))
    }
}
