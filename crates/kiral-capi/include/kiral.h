// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later
//
// This file is part of KIRAL. The SPDX lines name its copyright holder and its
// license, in the machine-readable form of the REUSE specification: the GNU
// General Public License, version 3 or any later version
// (LICENSES/GPL-3.0-or-later.txt).
//
// KIRAL's C ABI, for the frontends whose input method framework is reached
// from C or C++ -- Fcitx 5's addon: an engine of a dictionary file and a model
// file, and sessions of it, one for each input context, that keys are typed
// into. The functions are written in Rust (crates/kiral-capi) and held by its
// static library, libkiral_capi.a. C23 or C++.
//
// Threads. An engine and its sessions are used from one thread at a time,
// the frontend's. The wake-up a session is given is called on the engine's
// worker thread, and is to do no more than have the frontend's thread call
// kiral_session_poll (Fcitx 5's EventDispatcher::schedule): setting another,
// or freeing the session, waits for a call under way, and once it returns the
// wake-up replaced is not called. Close an engine once its sessions are freed:
// the sessions left show the dictionary's drafts alone.
//
// Strings. A string given is UTF-8 and its length in bytes, a path a
// NUL-terminated string. A string returned is UTF-8, its length in bytes given,
// and NUL-terminated besides; it lives until the next call that changes the
// session: kiral_session_key, _poll, _reset or _free.

#ifndef KIRAL_H
#define KIRAL_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// An engine: a dictionary, a model, and the worker that refines with it.
typedef struct kiral_engine kiral_engine;

// A session of an engine: what an input context types, shows and commits.
typedef struct kiral_session kiral_session;

// The keys that type no character. A key that types one is the character's
// Unicode scalar value, as the keyboard's layout and Shift make it.
enum : uint32_t {
    KIRAL_KEY_SPACE = 0x110000,
    KIRAL_KEY_ENTER,
    KIRAL_KEY_BACKSPACE,
    KIRAL_KEY_DELETE,
    KIRAL_KEY_ESCAPE,
    KIRAL_KEY_TAB,
    KIRAL_KEY_LEFT,
    KIRAL_KEY_RIGHT,
    KIRAL_KEY_UP,
    KIRAL_KEY_DOWN,
    KIRAL_KEY_HOME,
    KIRAL_KEY_END,
    KIRAL_KEY_PAGE_UP,
    KIRAL_KEY_PAGE_DOWN,
    // F1 to F24: KIRAL_KEY_F1 + n - 1 for Fn.
    KIRAL_KEY_F1 = 0x110100,
};

// The modifiers held with a key: a set of these bits. A key held with Control,
// Alt or Super is the application's.
enum : uint32_t {
    KIRAL_SHIFT = 1u << 0,
    KIRAL_CONTROL = 1u << 1,
    KIRAL_ALT = 1u << 2,
    KIRAL_SUPER = 1u << 3,
};

// Opens an engine of a dictionary file and of a model file, both mapped where
// they lie. Without a model -- model_path NULL -- the dictionary's drafts are
// all its sessions show. NULL where it does not open, the reason written to
// error, NUL-terminated within error_size bytes, where error is not NULL.
[[nodiscard]] kiral_engine *kiral_engine_open(const char *dictionary_path, const char *model_path, char *error,
                                              size_t error_size);

// Closes an engine: its worker stopped, and its files unmapped once its
// sessions are freed. NULL does nothing.
void kiral_engine_close(kiral_engine *engine);

// A new session of an engine, nothing typed; NULL for a NULL engine.
[[nodiscard]] kiral_session *kiral_session_new(kiral_engine *engine);

// Frees a session. NULL does nothing.
void kiral_session_free(kiral_session *session);

// The wake-up the engine's worker calls with data, on its own thread, once it
// has given back a refinement of the session's; NULL for none. Once this
// returns, the wake-up it replaced is not called.
void kiral_session_set_notify(kiral_session *session, void (*notify)(void *data), void *data);

// The text before the cursor, where the frontend knows it: the left context of
// the next conversion, in place of the text the session last committed.
void kiral_session_set_context(kiral_session *session, const char *text, size_t length);

// Types a key, with the modifiers held: whether the session took it, and the
// frontend is not to pass it on. What it shows after it is the preedit, the
// commit and the candidates.
bool kiral_session_key(kiral_session *session, uint32_t key, uint32_t modifiers);

// Takes the refinements that have come, without waiting: whether what the
// session shows changed.
bool kiral_session_poll(kiral_session *session);

// Deletes what is typed and the left context: the input context lost its
// focus, or its text changed under it.
void kiral_session_reset(kiral_session *session);

// What the session shows where the text is typed, and its length.
const char *kiral_session_preedit(const kiral_session *session, size_t *length);

// Where the cursor is in the preedit, in bytes.
size_t kiral_session_preedit_cursor(const kiral_session *session);

// The part of the preedit converted and selected, to be highlighted, from
// start to end in bytes: whether there is one.
bool kiral_session_preedit_highlight(const kiral_session *session, size_t *start, size_t *end);

// The text the last key committed, and its length: empty where it committed
// none.
const char *kiral_session_commit(const kiral_session *session, size_t *length);

// The number of the conversion's candidates: 0 where there is no conversion.
size_t kiral_session_candidates(const kiral_session *session);

// A candidate, and its length; NULL past the last.
const char *kiral_session_candidate(const kiral_session *session, size_t index, size_t *length);

// The candidate the preedit shows.
size_t kiral_session_candidate_selected(const kiral_session *session);

// Whether the list of candidates is to be shown.
bool kiral_session_candidates_shown(const kiral_session *session);

#ifdef __cplusplus
}
#endif

#endif
