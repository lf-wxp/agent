use std::{
  fmt::Write as _,
  path::Path,
  process::Stdio,
  sync::{Mutex, PoisonError},
  time::Duration,
};

use tokio::{
  io::{AsyncRead, AsyncReadExt},
  process::Command,
};

use crate::{
  config,
  error::ResultExt,
  tools::shell::{DEFAULT_TIMEOUT_SECS, MAX_TIMEOUT_SECS, RunCommandArgs},
};

/// Bytes of stdout kept; the rest is cut from the middle (see [`Capped`]).
const MAX_STDOUT_BYTES: usize = 32 * 1024;

/// Bytes of stderr kept.
const MAX_STDERR_BYTES: usize = 16 * 1024;

/// How long to keep draining the output pipes after the command has exited (or been
/// killed). They normally hit end-of-file at once; the grace only matters for a process
/// that escaped the group and still holds a pipe open, which must not hold the tool call
/// hostage until the timeout.
const DRAIN_GRACE: Duration = Duration::from_secs(2);

/// Runs `command` in a shell and reports exit code, stdout and stderr.
///
/// A non-zero exit is *not* an error: the output is the useful part (a failing test run,
/// a compiler error), so it is returned as-is with the exit code on top. Only failing to
/// start the command, or it outliving its timeout, is an error — and that error still
/// carries whatever the command had printed by then.
///
/// # Output
///
/// Read as it arrives into a bounded buffer ([`Capped`]) instead of collected whole:
/// `yes` or `cat /dev/urandom | base64` would otherwise fill memory for the full
/// timeout before anything got to truncate it.
///
/// # Process lifetime
///
/// On Unix the command runs in its own process group, and *every* way out of this
/// function — normal exit, timeout, or the future simply being dropped because the run
/// was cancelled — kills the whole group ([`ProcessGroup`]). Killing only the shell would
/// leave `server` running after `sh -c 'server & sleep 999'`, and a background process
/// that inherited the output pipes would also keep them open, so the call would never
/// see end-of-file. The consequence is deliberate: whatever a command leaves running in
/// the background does not outlive it.
pub async fn run(args_json: &str) -> crate::Result<String> {
  let RunCommandArgs {
    command,
    workdir,
    timeout_secs,
  } = serde_json::from_str::<RunCommandArgs>(args_json)
    .wrap_err(crate::Error::tool, "invalid run_command arguments")?;
  crate::error::ensure!(
    !command.trim().is_empty(),
    tool,
    "command must not be empty"
  );

  let secs = timeout_secs
    .unwrap_or(DEFAULT_TIMEOUT_SECS)
    .clamp(1, MAX_TIMEOUT_SECS);

  let mut child = build(&command, workdir.as_deref())?
    .spawn()
    .wrap_err_with(crate::Error::tool, || {
      format!("failed to start `{command}`")
    })?;
  // Created before anything can fail or be cancelled below, so no path skips the sweep.
  let group = ProcessGroup::new(child.id());
  let (Some(stdout), Some(stderr)) = (child.stdout.take(), child.stderr.take()) else {
    crate::error::bail!(tool, "failed to capture the output of `{command}`");
  };

  // Outside the future below so that, if the timeout drops it, what was captured so far
  // is still there to report.
  let out = Mutex::new(Capped::new(MAX_STDOUT_BYTES));
  let err = Mutex::new(Capped::new(MAX_STDERR_BYTES));

  let drive = async {
    // Pumped for as long as the command runs, not only after it exits: a command that
    // writes more than a pipe buffer would otherwise block on its own output.
    let pumps = async {
      tokio::join!(pump(stdout, &out), pump(stderr, &err));
    };
    tokio::pin!(pumps);

    let exited = tokio::select! {
      status = child.wait() => Some(status),
      () = &mut pumps => None,
    };
    let (status, pumps_done) = match exited {
      Some(status) => (status, false),
      // Both pipes closed first, e.g. a command that closes its stdio and carries on.
      None => (child.wait().await, true),
    };

    // The shell is gone; sweep anything it left behind. This is also what releases a
    // background process's grip on the pipes, so the drain below can finish.
    group.kill();
    if !pumps_done {
      let _ = tokio::time::timeout(DRAIN_GRACE, &mut pumps).await;
    }
    status
  };

  match tokio::time::timeout(Duration::from_secs(secs), drive).await {
    Ok(status) => {
      let status = status.wrap_err(crate::Error::tool, "failed to wait for the command")?;
      Ok(render(status.code(), &finish(out), &finish(err)))
    }
    Err(_) => {
      group.kill();
      let partial = render_streams(&finish(out), &finish(err));
      let mut message = format!("command timed out after {secs}s and was killed: `{command}`");
      if !partial.is_empty() {
        let _ = write!(message, "\n[output before the timeout]\n{partial}");
      }
      Err(crate::Error::tool(message))
    }
  }
}

/// The fully configured, not yet spawned command: closed stdin, piped output, own
/// process group, credentials removed from the environment.
fn build(command: &str, workdir: Option<&str>) -> crate::Result<Command> {
  let mut cmd = shell(command);
  if let Some(dir) = workdir.filter(|dir| !dir.is_empty()) {
    crate::error::ensure!(
      Path::new(dir).is_dir(),
      tool,
      "workdir `{dir}` is not an existing directory"
    );
    cmd.current_dir(dir);
  }
  strip_secrets(&mut cmd);
  cmd
    .stdin(Stdio::null())
    .stdout(Stdio::piped())
    .stderr(Stdio::piped())
    .kill_on_drop(true);
  #[cfg(unix)]
  cmd.process_group(0);
  Ok(cmd)
}

/// Removes this process's own credentials from the command's environment.
///
/// Not a full `env_clear()` like the stdio MCP servers get (see
/// `crate::tools::mcp::config`): a shell command legitimately needs the user's toolchain
/// environment (`CARGO_HOME`, `VIRTUAL_ENV`, ...). But nothing a model runs needs the
/// agent's *own* API keys, and `echo $OPENAI_API_KEY` is the obvious first thing a
/// prompt-injected page would ask for.
///
/// Two layers: the keys this crate reads ([`config::SECRET_ENV_VARS`], one list shared
/// with the code that reads them), and anything in the environment whose *name* looks
/// like a credential ([`config::is_secret_env_name`]) — a key loaded from `.env` for some
/// other purpose is just as readable by `env`. The price is that a command which needs
/// such a variable (`gh` with `GITHUB_TOKEN`) has to be given it some other way.
///
/// **This is not a security boundary.** A command can still read `.env` from disk, and
/// whatever else the user's account can read; stripping the environment only removes the
/// zero-effort leak. What actually gates a command is the human approval in front of it.
fn strip_secrets(cmd: &mut Command) {
  for var in config::SECRET_ENV_VARS {
    cmd.env_remove(var);
  }
  for (name, _) in std::env::vars_os() {
    if name.to_str().is_some_and(config::is_secret_env_name) {
      cmd.env_remove(name);
    }
  }
}

#[cfg(unix)]
fn shell(command: &str) -> Command {
  let mut cmd = Command::new("sh");
  cmd.arg("-c").arg(command);
  cmd
}

#[cfg(windows)]
fn shell(command: &str) -> Command {
  let mut cmd = Command::new("cmd");
  // Passed through as-is: `.arg()` would wrap a command containing spaces in quotes
  // using the C runtime's rules, which `cmd.exe` does not parse the same way.
  cmd.arg("/C").raw_arg(command);
  cmd
}

/// The command's process group, killed when this is dropped.
///
/// A guard rather than a call at the end because the interesting exits are the ones that
/// do not reach the end: the timeout, and — what cancellation relies on — the future
/// being dropped mid-await. `kill_on_drop` on its own reaches only the shell.
///
/// On Windows there is no process group to signal and this does nothing; the shell
/// itself is still killed by `kill_on_drop`.
struct ProcessGroup {
  #[cfg(unix)]
  pgid: Option<i32>,
  /// Set by the first kill, so the guard's own drop after a normal finish does not send
  /// a second signal to a group id the system may by then have handed to something else.
  #[cfg(unix)]
  killed: std::sync::atomic::AtomicBool,
}

impl ProcessGroup {
  #[cfg(unix)]
  fn new(pid: Option<u32>) -> Self {
    Self {
      // `process_group(0)` made the child its own group leader, so its pid is the group
      // id. Anything at or below 1 is refused: `kill(-1, ..)` would signal every process
      // the user owns.
      pgid: pid
        .and_then(|pid| i32::try_from(pid).ok())
        .filter(|pgid| *pgid > 1),
      killed: std::sync::atomic::AtomicBool::new(false),
    }
  }

  #[cfg(windows)]
  fn new(_pid: Option<u32>) -> Self {
    Self {}
  }

  /// `SIGKILL` to every process still in the group, once. A group that is already empty
  /// is not an error.
  ///
  /// There is a window, after the shell has been reaped and before this runs, in which
  /// its pid could in principle be reused; the call follows the reap directly, so the
  /// window is microseconds wide, and while any member of the group survives its id
  /// cannot be reused at all — which is exactly the case this exists for.
  #[cfg(unix)]
  fn kill(&self) {
    use std::sync::atomic::Ordering;

    let Some(pgid) = self.pgid else {
      return;
    };
    if self.killed.swap(true, Ordering::SeqCst) {
      return;
    }
    // SAFETY: `kill(2)` takes two integers and no pointers. A negative pid addresses
    // the process group `-pid`, and `pgid > 1` was checked in `new`.
    unsafe {
      libc::kill(-pgid, libc::SIGKILL);
    }
  }

  #[cfg(windows)]
  fn kill(&self) {}
}

impl Drop for ProcessGroup {
  fn drop(&mut self) {
    self.kill();
  }
}

/// Copies `reader` into `sink` until end-of-file or a read error.
///
/// The lock is held only for the copy itself, never across an `.await`.
async fn pump(mut reader: impl AsyncRead + Unpin, sink: &Mutex<Capped>) {
  let mut chunk = [0u8; 8 * 1024];
  loop {
    match reader.read(&mut chunk).await {
      Ok(0) | Err(_) => break,
      Ok(read) => sink
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .push(&chunk[..read]),
    }
  }
}

fn finish(sink: Mutex<Capped>) -> Capped {
  sink.into_inner().unwrap_or_else(PoisonError::into_inner)
}

/// The first and last bytes of a stream, with a count of what lies between.
///
/// Both ends matter for command output: the head says what ran, the tail carries the
/// error or the summary line — so a head-only cut would drop exactly the part the model
/// needs most. Memory is bounded by the cap however much the command prints.
///
/// Counts bytes, not characters, because the cut happens before the text is decoded; a
/// multi-byte character split by it decodes to U+FFFD rather than panicking.
#[derive(Debug)]
struct Capped {
  head: Vec<u8>,
  head_cap: usize,
  tail: Vec<u8>,
  tail_cap: usize,
  omitted: u64,
}

impl Capped {
  fn new(max: usize) -> Self {
    let head_cap = max / 2;
    Self {
      head: Vec::new(),
      head_cap,
      tail: Vec::new(),
      tail_cap: max - head_cap,
      omitted: 0,
    }
  }

  fn push(&mut self, mut bytes: &[u8]) {
    if self.head.len() < self.head_cap {
      let take = (self.head_cap - self.head.len()).min(bytes.len());
      self.head.extend_from_slice(&bytes[..take]);
      bytes = &bytes[take..];
    }
    self.tail.extend_from_slice(bytes);
    if self.tail.len() > self.tail_cap {
      let excess = self.tail.len() - self.tail_cap;
      self.tail.drain(..excess);
      self.omitted += excess as u64;
    }
  }

  fn text(&self) -> String {
    let head = String::from_utf8_lossy(&self.head);
    let tail = String::from_utf8_lossy(&self.tail);
    if self.omitted == 0 {
      format!("{head}{tail}")
    } else {
      format!("{head}\n... [{} bytes omitted] ...\n{tail}", self.omitted)
    }
  }
}

/// The `[stdout]` / `[stderr]` sections, empty when neither stream printed anything.
fn render_streams(stdout: &Capped, stderr: &Capped) -> String {
  let mut text = String::new();
  for (label, stream) in [("stdout", stdout), ("stderr", stderr)] {
    let body = stream.text();
    if !body.trim().is_empty() {
      let _ = writeln!(text, "[{label}]\n{}", body.trim_end());
    }
  }
  text
}

fn render(code: Option<i32>, stdout: &Capped, stderr: &Capped) -> String {
  let mut text = match code {
    Some(code) => format!("exit code: {code}\n"),
    None => "exit code: none (terminated by a signal)\n".to_owned(),
  };
  let streams = render_streams(stdout, stderr);
  if streams.is_empty() {
    text.push_str("(no output)\n");
  } else {
    text.push_str(&streams);
  }
  text
}

#[cfg(all(test, unix))]
mod tests {
  use std::time::Instant;

  use serde_json::json;

  use super::*;
  use crate::tools::testing::TempDir;

  async fn run_json(value: serde_json::Value) -> crate::Result<String> {
    run(&value.to_string()).await
  }

  /// Whether a process with this pid still exists (signal 0 only probes).
  fn is_alive(pid: i32) -> bool {
    std::process::Command::new("kill")
      .args(["-0", &pid.to_string()])
      .stdout(Stdio::null())
      .stderr(Stdio::null())
      .status()
      .is_ok_and(|status| status.success())
  }

  async fn wait_until_dead(pid: i32) -> bool {
    for _ in 0..40 {
      if !is_alive(pid) {
        return true;
      }
      tokio::time::sleep(Duration::from_millis(50)).await;
    }
    false
  }

  #[tokio::test]
  async fn reports_exit_code_and_stdout() {
    let output = run_json(json!({ "command": "echo hello" })).await.unwrap();
    assert_eq!(output, "exit code: 0\n[stdout]\nhello\n");
  }

  #[tokio::test]
  async fn a_failing_command_is_not_an_error() {
    let output = run_json(json!({ "command": "echo oops >&2; exit 3" }))
      .await
      .unwrap();
    assert!(output.starts_with("exit code: 3"), "got: {output}");
    assert!(output.contains("[stderr]\noops"), "got: {output}");
  }

  #[tokio::test]
  async fn runs_in_the_requested_workdir() {
    let dir = TempDir::new("shell");
    std::fs::write(dir.join("marker.txt"), "").unwrap();

    let output = run_json(json!({ "command": "ls", "workdir": dir.path().to_string_lossy() }))
      .await
      .unwrap();
    assert!(output.contains("marker.txt"), "got: {output}");
  }

  #[tokio::test]
  async fn rejects_a_missing_workdir() {
    let err = run_json(json!({ "command": "ls", "workdir": "/definitely/not/here" }))
      .await
      .unwrap_err();
    assert!(
      err.to_string().contains("not an existing directory"),
      "got: {err}"
    );
  }

  #[tokio::test]
  async fn stdin_is_closed_so_reading_it_does_not_hang() {
    let output = run_json(json!({ "command": "cat", "timeout_secs": 5 }))
      .await
      .unwrap();
    assert_eq!(output, "exit code: 0\n(no output)\n");
  }

  #[tokio::test]
  async fn a_command_outliving_its_timeout_is_killed() {
    let started = Instant::now();
    let err = run_json(json!({ "command": "sleep 30 & sleep 30", "timeout_secs": 1 }))
      .await
      .unwrap_err();
    assert!(err.to_string().contains("timed out after 1s"), "got: {err}");
    assert!(started.elapsed() < Duration::from_secs(10));
  }

  /// The model needs to see how far a command got before it was cut off, not only that
  /// it was.
  #[tokio::test]
  async fn a_timeout_still_reports_the_output_so_far() {
    let err = run_json(json!({ "command": "echo got this far; sleep 30", "timeout_secs": 1 }))
      .await
      .unwrap_err()
      .to_string();
    assert!(err.contains("timed out after 1s"), "got: {err}");
    assert!(err.contains("got this far"), "got: {err}");
  }

  /// A background process inherits the output pipes. Waiting for end-of-file on them
  /// would last as long as the background process does — here, the whole timeout.
  #[tokio::test]
  async fn a_background_process_does_not_hold_the_call_hostage() {
    let started = Instant::now();
    let output = run_json(json!({ "command": "(sleep 30 &); echo started", "timeout_secs": 20 }))
      .await
      .unwrap();
    assert!(output.contains("started"), "got: {output}");
    assert!(
      started.elapsed() < Duration::from_secs(10),
      "waited for the background process"
    );
  }

  /// Reads the pid the command wrote, waiting briefly for it to appear: the file is
  /// created by the background process, so on a loaded machine it may lag the call.
  async fn read_pid(path: &Path) -> i32 {
    for _ in 0..200 {
      if let Ok(text) = std::fs::read_to_string(path)
        && let Ok(pid) = text.trim().parse()
      {
        return pid;
      }
      tokio::time::sleep(Duration::from_millis(25)).await;
    }
    panic!(
      "the background process never wrote its pid to {}",
      path.display()
    );
  }

  /// What a command leaves running in the background does not outlive it.
  #[tokio::test]
  async fn background_processes_are_killed_when_the_command_finishes() {
    let dir = TempDir::new("shell-bg");
    let pid_file = dir.join("pid");
    let command = format!("sleep 60 & echo $! > '{}'", pid_file.display());
    run_json(json!({ "command": command })).await.unwrap();

    let pid = read_pid(&pid_file).await;
    assert!(wait_until_dead(pid).await, "background process {pid} lives");
  }

  /// Cancellation drops the future rather than timing it out; the group must die on
  /// that path too, or a cancelled run leaves its commands running.
  #[tokio::test]
  async fn dropping_the_call_kills_the_whole_process_group() {
    let dir = TempDir::new("shell-drop");
    let pid_file = dir.join("pid");
    let command = format!("sleep 60 & echo $! > '{}'; sleep 60", pid_file.display());

    // Aborted once both processes are demonstrably up, so the test does not depend on
    // how long that takes.
    let call =
      tokio::spawn(
        async move { run_json(json!({ "command": command, "timeout_secs": 300 })).await },
      );
    let pid = read_pid(&pid_file).await;
    call.abort();
    let _ = call.await;

    assert!(wait_until_dead(pid).await, "background process {pid} lives");
  }

  /// A command printing without end must not be buffered whole.
  #[tokio::test]
  async fn runaway_output_is_bounded_and_keeps_both_ends() {
    let output = run_json(json!({
      "command": "echo first; head -c 20000000 /dev/zero | tr '\\0' 'a'; echo last",
      "timeout_secs": 60,
    }))
    .await
    .unwrap();

    assert!(output.len() < 100_000, "kept {} bytes", output.len());
    assert!(output.contains("first"), "lost the head");
    assert!(output.contains("last"), "lost the tail");
    assert!(
      output.contains("bytes omitted"),
      "got a prefix: {output:.200}"
    );
  }

  /// Asserted on the built command rather than by echoing the variable: the test process
  /// may not have the key set at all, which would make an echo-based test pass vacuously.
  #[test]
  fn does_not_leak_the_agents_own_secrets() {
    let cmd = build("true", None).unwrap();
    let removed: Vec<_> = cmd
      .as_std()
      .get_envs()
      .filter(|(_, value)| value.is_none())
      .map(|(key, _)| key.to_string_lossy().into_owned())
      .collect();
    for var in config::SECRET_ENV_VARS {
      assert!(removed.iter().any(|key| key == var), "`{var}` is inherited");
    }
  }

  /// A key this crate never reads (loaded from `.env` for something else) is caught by
  /// its name.
  #[test]
  fn strips_other_credential_looking_variables_too() {
    // SAFETY: a name used by no other test, set before anything reads it.
    unsafe { std::env::set_var("AGENT_TEST_SHELL_OTHER_PROVIDER_API_KEY", "s3cret") };
    unsafe { std::env::set_var("AGENT_TEST_SHELL_HARMLESS", "fine") };

    let cmd = build("true", None).unwrap();
    let removed: Vec<_> = cmd
      .as_std()
      .get_envs()
      .filter(|(_, value)| value.is_none())
      .map(|(key, _)| key.to_string_lossy().into_owned())
      .collect();
    assert!(
      removed
        .iter()
        .any(|key| key == "AGENT_TEST_SHELL_OTHER_PROVIDER_API_KEY")
    );
    assert!(!removed.iter().any(|key| key == "AGENT_TEST_SHELL_HARMLESS"));
  }

  #[tokio::test]
  async fn rejects_an_empty_command() {
    let err = run_json(json!({ "command": "  " })).await.unwrap_err();
    assert!(err.to_string().contains("must not be empty"), "got: {err}");
  }

  #[test]
  fn capped_keeps_both_ends_and_counts_the_middle() {
    let mut capped = Capped::new(4);
    capped.push(b"abcdefghij");
    let text = capped.text();

    assert!(text.starts_with("ab\n"), "got: {text}");
    assert!(text.ends_with("\nij"), "got: {text}");
    assert!(text.contains("[6 bytes omitted]"), "got: {text}");
  }

  #[test]
  fn capped_is_exact_when_the_output_fits() {
    let mut capped = Capped::new(100);
    capped.push(b"hello ");
    capped.push(b"world");
    assert_eq!(capped.text(), "hello world");
  }

  #[test]
  fn capped_gives_the_same_result_however_the_input_is_chunked() {
    let data: Vec<u8> = (0..=255u8).cycle().take(5_000).collect();
    let mut whole = Capped::new(64);
    whole.push(&data);
    let mut pieces = Capped::new(64);
    for piece in data.chunks(7) {
      pieces.push(piece);
    }
    assert_eq!(whole.text(), pieces.text());
  }

  #[test]
  fn capped_survives_a_cut_inside_a_multibyte_character() {
    let mut capped = Capped::new(4);
    capped.push("中".repeat(10).as_bytes());
    assert!(capped.text().contains("bytes omitted"));
  }

  #[test]
  fn the_group_guard_refuses_ids_that_would_signal_everyone() {
    assert!(ProcessGroup::new(Some(0)).pgid.is_none());
    assert!(ProcessGroup::new(Some(1)).pgid.is_none());
    assert!(ProcessGroup::new(None).pgid.is_none());
    assert_eq!(ProcessGroup::new(Some(4242)).pgid, Some(4242));
  }
}
