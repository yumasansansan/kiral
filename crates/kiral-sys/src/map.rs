// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later
//
// This file is part of KIRAL. The SPDX lines name its copyright holder and its
// license, in the machine-readable form of the REUSE specification: the GNU
// General Public License, version 3 or any later version
// (LICENSES/GPL-3.0-or-later.txt).

//! Files mapped into memory, read-only.
//!
//! On Linux and macOS, `mmap` of the whole file, read-only and private; on
//! Windows, a read-only mapping of the file (`CreateFileMappingW`) and a view of
//! the whole of it (`MapViewOfFile`). The pages are the file's: the system reads
//! each from the disk when it is first touched, shares it with any other process
//! that maps the file, and may drop it again when memory is short rather than
//! write it out, so that a model costs no reading before it is used and no
//! memory of the process's own.
//!
//! A mapped file must not change while it is mapped: a byte written by another
//! process would change under the slice the map gives out, and a file cut short
//! would end the process at its next touch of a page past the new end
//! (`SIGBUS`). Files are replaced, then, never rewritten: a package installs a
//! model or a dictionary by writing a new file and renaming it over the old,
//! which leaves the mapped file as it was. Windows refuses to cut short a file
//! that is mapped.

use std::fs::File;
use std::io;
use std::path::Path;
use std::ptr::NonNull;

#[cfg(any(target_os = "linux", target_os = "macos"))]
mod unix {
    use std::ffi::c_void;

    /// Pages that may be read.
    pub const PROT_READ: i32 = 1;
    /// A mapping whose writes, were there any, would be the process's own.
    pub const MAP_PRIVATE: i32 = 2;

    unsafe extern "C" {
        /// Maps `length` bytes of the open file `descriptor` from `offset`, at an
        /// address the system chooses where `address` is null; the address of
        /// the mapping, or `MAP_FAILED` (all bits set) with `errno` set. `off_t`
        /// is of 64 bits on 64-bit Linux and on macOS.
        pub fn mmap(
            address: *mut c_void,
            length: usize,
            protection: i32,
            flags: i32,
            descriptor: i32,
            offset: i64,
        ) -> *mut c_void;
        /// Unmaps the pages of `length` bytes from `address`; 0, or -1 with
        /// `errno` set.
        pub fn munmap(address: *mut c_void, length: usize) -> i32;
    }
}

#[cfg(windows)]
mod windows {
    use std::ffi::c_void;

    /// A mapping whose pages may be read.
    pub const PAGE_READONLY: u32 = 0x02;
    /// A view whose pages may be read.
    pub const FILE_MAP_READ: u32 = 0x04;

    #[link(name = "kernel32")]
    unsafe extern "system" {
        /// A mapping of the open file `file`, unnamed, of the file's size where
        /// both sizes are 0; its handle, or null with the last error set.
        pub fn CreateFileMappingW(
            file: *mut c_void,
            attributes: *const c_void,
            protection: u32,
            size_high: u32,
            size_low: u32,
            name: *const u16,
        ) -> *mut c_void;
        /// A view of `bytes` bytes of a mapping from an offset; its address, or
        /// null with the last error set. The view keeps the mapping.
        pub fn MapViewOfFile(
            mapping: *mut c_void,
            access: u32,
            offset_high: u32,
            offset_low: u32,
            bytes: usize,
        ) -> *mut c_void;
        /// Unmaps a view; nonzero, or 0 with the last error set.
        pub fn UnmapViewOfFile(address: *const c_void) -> i32;
        /// Closes a handle; nonzero, or 0 with the last error set.
        pub fn CloseHandle(handle: *mut c_void) -> i32;
    }
}

/// A file mapped into memory, read-only, for as long as the map lives.
#[derive(Debug)]
pub struct Map {
    /// The first byte of the mapping; dangling for an empty file, of which no
    /// mapping is made.
    address: NonNull<u8>,
    /// The bytes of the file.
    length: usize,
}

// SAFETY: the mapping is the process's, not the thread's that made it; the map
// gives out shared slices of it alone, and unmaps it once, when it is dropped.
unsafe impl Send for Map {}

// SAFETY: shared slices of memory that nothing writes may be read from any
// thread at once.
unsafe impl Sync for Map {}

impl Map {
    /// Maps the whole file at `path`, read-only; an empty file is no bytes, of
    /// no mapping.
    ///
    /// # Errors
    ///
    /// Where the file cannot be opened, is larger than the address space, or
    /// the system refuses to map it -- or maps no files, on a system other than
    /// Linux, macOS and Windows.
    pub fn open(path: &Path) -> io::Result<Self> {
        let file = File::open(path)?;
        let length = usize::try_from(file.metadata()?.len())
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "a file larger than the address space"))?;
        if length == 0 {
            return Ok(Self {
                address: NonNull::dangling(),
                length,
            });
        }
        Ok(Self {
            address: map(&file, length)?,
            length,
        })
    }

    /// The file's bytes.
    #[must_use]
    pub fn bytes(&self) -> &[u8] {
        // SAFETY: `address` is the start of a mapping of `length` readable bytes,
        // or dangling and well aligned for bytes with `length` 0; the mapping
        // lives until `self` is dropped, which the slice borrows, and nothing
        // writes it (the module's note says why a mapped file must not change).
        unsafe { std::slice::from_raw_parts(self.address.as_ptr(), self.length) }
    }
}

impl Drop for Map {
    fn drop(&mut self) {
        if self.length == 0 {
            return;
        }
        #[cfg(any(target_os = "linux", target_os = "macos"))]
        {
            // SAFETY: the address and length are those of the mapping `mmap` made,
            // unmapped once, here; no slice of it outlives the map.
            let result = unsafe { unix::munmap(self.address.as_ptr().cast(), self.length) };
            debug_assert_eq!(result, 0, "munmap: {}", io::Error::last_os_error());
        }
        #[cfg(windows)]
        {
            // SAFETY: the address is that of the view `MapViewOfFile` made,
            // unmapped once, here; no slice of it outlives the map.
            let result = unsafe { windows::UnmapViewOfFile(self.address.as_ptr().cast_const().cast()) };
            debug_assert_ne!(result, 0, "UnmapViewOfFile: {}", io::Error::last_os_error());
        }
    }
}

/// A read-only mapping of the first `length` bytes of an open file, which has
/// them.
#[cfg(any(target_os = "linux", target_os = "macos"))]
fn map(file: &File, length: usize) -> io::Result<NonNull<u8>> {
    use std::os::fd::AsRawFd;
    // SAFETY: a new mapping at an address the system chooses, which touches no
    // memory the process has; the descriptor is open across the call, and the
    // mapping outlives it, closing a file leaving its mappings.
    let address = unsafe {
        unix::mmap(
            std::ptr::null_mut(),
            length,
            unix::PROT_READ,
            unix::MAP_PRIVATE,
            file.as_raw_fd(),
            0,
        )
    };
    if address.addr() == usize::MAX {
        return Err(io::Error::last_os_error());
    }
    NonNull::new(address.cast()).ok_or_else(|| io::Error::other("a mapping at address 0"))
}

/// A read-only view of the first `length` bytes of an open file, which has
/// them.
#[cfg(windows)]
fn map(file: &File, length: usize) -> io::Result<NonNull<u8>> {
    use std::os::windows::io::AsRawHandle;
    // SAFETY: a new mapping, unnamed, of a file whose handle is open across the
    // call; it touches no memory the process has.
    let mapping = unsafe {
        windows::CreateFileMappingW(
            file.as_raw_handle(),
            std::ptr::null(),
            windows::PAGE_READONLY,
            0,
            0,
            std::ptr::null(),
        )
    };
    if mapping.is_null() {
        return Err(io::Error::last_os_error());
    }
    // SAFETY: a new view of the mapping just made, of the file's length; it
    // touches no memory the process has.
    let address = unsafe { windows::MapViewOfFile(mapping, windows::FILE_MAP_READ, 0, 0, length) };
    let error = io::Error::last_os_error();
    // SAFETY: the mapping's handle, closed once, here; the view keeps the
    // mapping.
    let closed = unsafe { windows::CloseHandle(mapping) };
    debug_assert_ne!(closed, 0, "CloseHandle: {}", io::Error::last_os_error());
    NonNull::new(address.cast()).ok_or(error)
}

/// No mapping, on a system whose functions for it are not declared here.
#[cfg(not(any(target_os = "linux", target_os = "macos", windows)))]
fn map(_file: &File, length: usize) -> io::Result<NonNull<u8>> {
    Err(io::Error::new(
        io::ErrorKind::Unsupported,
        format!("files are mapped on Linux, macOS and Windows only (asked for {length} bytes)"),
    ))
}
