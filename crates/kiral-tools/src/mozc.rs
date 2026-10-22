// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later
//
// This file is part of KIRAL. The SPDX lines name its copyright holder and its
// license, in the machine-readable form of the REUSE specification: the GNU
// General Public License, version 3 or any later version
// (LICENSES/GPL-3.0-or-later.txt).

//! Mozc, measured as an input method: keys typed into it one at a time
//! through `mozc_emacs_helper`, whose server it talks to over Mozc's IPC.
//!
//! The run has a Mozc of its own: a profile in a directory made for it and
//! removed after, and the server started here, before the helper, with that
//! profile -- the helper would otherwise start one that outlives it, or talk to
//! the user's own and learn from what the run types. Nothing is committed:
//! a conversion is read from the preedit after Space, then cancelled, so that
//! Mozc learns nothing between the items and each is converted as the first.
//! Linux only, where Mozc's helper is packaged.

use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::time::{Duration, Instant};

use crate::sexp::{self, Value};

/// Where Mozc's programs are.
#[derive(Debug, Clone)]
pub struct Programs {
    /// `mozc_emacs_helper`.
    pub helper: PathBuf,
    /// `mozc_server`.
    pub server: PathBuf,
}

impl Default for Programs {
    fn default() -> Self {
        Self {
            helper: PathBuf::from("mozc_emacs_helper"),
            server: PathBuf::from("/usr/lib/mozc/mozc_server"),
        }
    }
}

/// A Mozc of the run's own, and a session of it.
#[derive(Debug)]
pub struct Mozc {
    server: Child,
    helper: Child,
    input: Option<ChildStdin>,
    output: BufReader<ChildStdout>,
    profile: PathBuf,
    event: u64,
    session: i64,
    version: String,
}

/// The environment Mozc's programs run in: a home of the run's own, whose
/// configuration directory Mozc's profile goes in.
fn environment(profile: &Path) -> Vec<(&'static str, PathBuf)> {
    vec![
        ("PATH", PathBuf::from("/usr/bin:/bin")),
        ("HOME", profile.to_path_buf()),
        ("XDG_CONFIG_HOME", profile.join(".config")),
        ("XDG_DATA_HOME", profile.join(".local/share")),
        ("XDG_CACHE_HOME", profile.join(".cache")),
    ]
}

impl Mozc {
    /// Starts a Mozc of its own: its server, then the helper, and a session.
    ///
    /// # Errors
    ///
    /// Where a program cannot be started, the server does not come up within
    /// ten seconds, or the helper does not answer as it should.
    pub fn start(programs: &Programs) -> Result<Self, String> {
        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |since| since.as_nanos());
        let profile = std::env::temp_dir().join(format!("kiral-mozc-{}-{stamp}", std::process::id()));
        // The server makes its directory in the configuration directory, but not
        // the configuration directory itself.
        std::fs::create_dir_all(profile.join(".config")).map_err(|error| format!("{}: {error}", profile.display()))?;
        let environment = environment(&profile);
        let mut server = Command::new(&programs.server)
            .env_clear()
            .envs(environment.iter().map(|(name, value)| (name, value)))
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .map_err(|error| format!("{}: {error}", programs.server.display()))?;
        let socket = profile.join(".config/mozc/.session.ipc");
        let started = Instant::now();
        while !socket.exists() {
            if let Ok(Some(status)) = server.try_wait() {
                let _ = std::fs::remove_dir_all(&profile);
                return Err(format!(
                    "{} ended before it took sessions: {status}",
                    programs.server.display()
                ));
            }
            if started.elapsed() > Duration::from_secs(10) {
                let _ = server.kill();
                let _ = server.wait();
                let _ = std::fs::remove_dir_all(&profile);
                return Err(format!(
                    "{} took no sessions within ten seconds",
                    programs.server.display()
                ));
            }
            std::thread::sleep(Duration::from_millis(5));
        }
        let mut helper = match Command::new(&programs.helper)
            .env_clear()
            .envs(environment.iter().map(|(name, value)| (name, value)))
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
        {
            Ok(helper) => helper,
            Err(error) => {
                let _ = server.kill();
                let _ = server.wait();
                let _ = std::fs::remove_dir_all(&profile);
                return Err(format!(
                    "{}: {error} (the package emacs-mozc-bin has it)",
                    programs.helper.display()
                ));
            }
        };
        let input = helper.stdin.take();
        let output = BufReader::new(helper.stdout.take().expect("the helper's piped output"));
        let mut mozc = Self {
            server,
            helper,
            input,
            output,
            profile,
            event: 0,
            session: 0,
            version: String::new(),
        };
        let greeting = mozc.line()?;
        mozc.version = greeting
            .get("version")
            .and_then(|version| version.as_str().map(str::to_owned))
            .ok_or_else(|| "the helper's greeting names no version".to_owned())?;
        let created = mozc.request("CreateSession")?;
        mozc.session = created
            .get("emacs-session-id")
            .and_then(|id| id.as_integer())
            .ok_or_else(|| "the helper gave no session".to_owned())?;
        Ok(mozc)
    }

    /// The version of Mozc, as its helper gives it.
    pub fn version(&self) -> &str {
        &self.version
    }

    /// The helper's next line, read as an S-expression.
    fn line(&mut self) -> Result<Value, String> {
        let mut line = String::new();
        let read = self
            .output
            .read_line(&mut line)
            .map_err(|error| format!("the helper's answer: {error}"))?;
        if read == 0 {
            return Err("the helper ended".to_owned());
        }
        sexp::parse(&line).map_err(|error| format!("the helper's answer: {error}"))
    }

    /// Sends a command, and returns the helper's answer to it.
    fn request(&mut self, command: &str) -> Result<Value, String> {
        self.event += 1;
        let input = self.input.as_mut().ok_or("the helper's input is closed")?;
        writeln!(input, "({} {command})", self.event)
            .and_then(|()| input.flush())
            .map_err(|error| format!("the helper's input: {error}"))?;
        let answer = self.line()?;
        if let Some(error) = answer.get("error") {
            let kind = error.as_symbol().unwrap_or("an error");
            let message = answer
                .get("message")
                .and_then(|message| message.as_str().map(str::to_owned));
            return Err(format!("Mozc: {kind}: {}", message.unwrap_or_default()));
        }
        let event = answer.get("emacs-event-id").and_then(|id| id.as_integer());
        if event != i64::try_from(self.event).ok() {
            return Err(format!("an answer to another command than {}", self.event));
        }
        Ok(answer)
    }

    /// Sends a key -- a character's code, or a key's name such as `space` --
    /// and returns the output of its answer.
    fn key(&mut self, key: &str) -> Result<Value, String> {
        let answer = self.request(&format!("SendKey {} {key}", self.session))?;
        Ok(answer.get("output").unwrap_or(Value::List(Vec::new(), None)))
    }

    /// The text of an output's preedit: its segments one after another; empty
    /// where there is none.
    fn preedit_of(output: &Value) -> String {
        output
            .get("preedit")
            .and_then(|preedit| preedit.get("segment"))
            .map(|segments| {
                segments
                    .items()
                    .iter()
                    .filter_map(|segment| segment.get("value").and_then(|value| value.as_str().map(str::to_owned)))
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Types keys, one at a time, pushing the time each took to answer, and
    /// returns the preedit after the last.
    ///
    /// # Errors
    ///
    /// Where the helper does not answer as it should.
    pub fn type_keys(&mut self, keys: &str, times: &mut Vec<Duration>) -> Result<String, String> {
        let mut preedit = String::new();
        for key in keys.bytes() {
            let started = Instant::now();
            let output = self.key(&key.to_string())?;
            times.push(started.elapsed());
            preedit = Self::preedit_of(&output);
        }
        Ok(preedit)
    }

    /// Converts what is typed (Space), and returns the conversion -- its
    /// segments' candidates, one after another -- and the time it took.
    ///
    /// # Errors
    ///
    /// Where the helper does not answer as it should.
    pub fn convert(&mut self) -> Result<(String, Duration), String> {
        let started = Instant::now();
        let output = self.key("space")?;
        let time = started.elapsed();
        Ok((Self::preedit_of(&output), time))
    }

    /// Cancels the conversion and what was typed, committing nothing.
    ///
    /// # Errors
    ///
    /// Where the helper does not answer as it should, or a preedit is left.
    pub fn cancel(&mut self) -> Result<(), String> {
        for _ in 0..3 {
            if Self::preedit_of(&self.key("escape")?).is_empty() {
                return Ok(());
            }
        }
        Err("a preedit that three escapes did not cancel".to_owned())
    }
}

impl Drop for Mozc {
    fn drop(&mut self) {
        let _ = self.request(&format!("DeleteSession {}", self.session));
        drop(self.input.take());
        let _ = self.helper.wait();
        let _ = self.server.kill();
        let _ = self.server.wait();
        let _ = std::fs::remove_dir_all(&self.profile);
    }
}
