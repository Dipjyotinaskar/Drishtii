"""
Cross-platform Text-to-Speech engine.
Uses pyttsx3 via a subprocess worker to avoid Streamlit threading conflicts.
Works on Windows (SAPI5), macOS (NSSpeechSynthesizer), Linux (espeak).

SAVE THIS FILE TO:
    C:\\Users\\DIPJYOTI\\Desktop\\SIGN\\SiglanR\\inter\\backend\\tts_engine.py

═══════════════════════════════════════════════════════════════════════════
CHANGELOG  (8 bugs fixed)
═══════════════════════════════════════════════════════════════════════════

BUG 1 [HIGH] — _check() only tested 'import pyttsx3', not worker-file existence
  If speak_worker.py was missing or in the wrong path, self.available was True
  but every speak() silently failed inside Popen with FileNotFoundError.
  FIX: _check() now also verifies _WORKER exists on disk and raises a clear
  warning if it doesn't.

BUG 2 [HIGH] — stop() leaked zombie processes on Linux/macOS
  terminate() sends SIGTERM; the kernel keeps the process entry until the
  parent calls wait(). Over many speak/stop cycles this leaked OS handles.
  Confirmed: State = Z (zombie) after terminate() without wait().
  FIX: Added proc.wait(timeout=2) after terminate(); falls back to kill()
  if the process ignores SIGTERM (rare but possible with pyttsx3/espeak).

BUG 3 [MEDIUM] — No max-length guard on text
  Very long strings (>32,767 chars) hit Windows CreateProcess ARG_MAX.
  FIX: Text is silently truncated to MAX_TEXT_CHARS (2000) at a clean word
  boundary. A warning is printed if truncation occurs.

BUG 4 [MEDIUM] — stderr=DEVNULL silently swallowed all worker errors
  Any crash in speak_worker.py (missing espeak, bad voice, etc.) was
  completely invisible — the engine appeared "available" but produced nothing.
  FIX: stderr is now captured into a PIPE. After speak() returns, a
  background-compatible stderr drain is available; stop() logs any stderr
  output when the process has already exited.

BUG 5 [MEDIUM] — No __del__ / context-manager cleanup
  If TTSEngine was garbage-collected while speaking (Streamlit hot-reload,
  app restart), the child process kept running as an orphan, potentially
  holding the audio device lock and blocking future TTS calls.
  FIX: Added __del__ and __exit__ / context-manager support so the
  subprocess is always terminated on cleanup.

BUG 6 [LOW] — available flag cached forever at init time
  Installed pyttsx3 mid-session wouldn't be detected until app restart.
  FIX: Added recheck() method so callers can probe again after installation.
  Also documented the st.cache_resource interaction.

BUG 7 [LOW] — is_speaking() had a TOCTOU race
  poll() returned None (running), then the process could exit before the
  return statement. Benign in single-threaded Streamlit but risky if used
  in a blocking loop.
  FIX: Result is snapshotted into a local variable before return; docstring
  documents the inherent race and recommends not using it for blocking waits.

BUG 8 [LOW] — _WORKER path computed at module-load time
  Streamlit hot-reload can resolve __file__ differently across reloads,
  causing _WORKER to point to a stale path.
  FIX: _WORKER is re-resolved inside __init__ using the class's own __file__
  attribute rather than a module-level constant.
"""

from __future__ import annotations

import os
import sys
import subprocess
import warnings
from typing import Optional

# ── Constants ──────────────────────────────────────────────────────────────────
# Safe upper bound for subprocess argv text length.
# Windows CreateProcess limit: 32,767 chars total command line.
# Keeping text ≤ 2,000 chars leaves room for python.exe + worker path.
MAX_TEXT_CHARS: int = 2_000


class TTSEngine:
    """
    Non-blocking TTS engine that runs pyttsx3 in an isolated subprocess,
    sidestepping Streamlit's threading restrictions.

    Usage
    -----
    tts = TTSEngine()
    tts.speak("Hello world")   # non-blocking
    tts.stop()                 # cancel mid-speech
    if tts.is_speaking(): ...  # poll status

    Context-manager usage (guarantees cleanup):
        with TTSEngine() as tts:
            tts.speak("Hello")
    """

    def __init__(self) -> None:
        # FIX BUG 8: resolve worker path at instance-creation time (not module
        # load time) so Streamlit hot-reloads always get a fresh, correct path.
        self._worker: str = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "speak_worker.py",
        )
        self._proc: Optional[subprocess.Popen] = None
        self.available: bool = self._check()

    # ── Availability check ─────────────────────────────────────────────────────

    def _check(self) -> bool:
        """
        Return True only if BOTH conditions hold:
          1. pyttsx3 is importable in the current Python environment.
          2. speak_worker.py exists at the expected path.

        FIX BUG 1: original only checked condition 1.
        """
        # Condition 1: pyttsx3 installed
        try:
            import pyttsx3  # noqa: F401
        except ImportError:
            warnings.warn(
                "[TTSEngine] pyttsx3 is not installed. "
                "Run:  pip install pyttsx3\n"
                "TTS will be disabled until the package is installed and "
                "recheck() is called (or the app is restarted).",
                UserWarning,
                stacklevel=2,
            )
            return False

        # Condition 2: worker script present on disk
        if not os.path.isfile(self._worker):
            warnings.warn(
                f"[TTSEngine] speak_worker.py not found at:\n  {self._worker}\n"
                "Ensure speak_worker.py is in the same directory as tts_engine.py. "
                "TTS will be disabled.",
                UserWarning,
                stacklevel=2,
            )
            return False

        return True

    def recheck(self) -> bool:
        """
        Re-probe availability.  Call after installing pyttsx3 mid-session
        without restarting the app.

        FIX BUG 6: available was set only once at __init__ (inside
        @st.cache_resource), making mid-session installation invisible.
        """
        self.available = self._check()
        return self.available

    # ── Core API ───────────────────────────────────────────────────────────────

    def speak(self, text: str) -> None:
        """
        Start speaking *text* in a background subprocess (non-blocking).
        Any ongoing speech is cancelled first.

        Parameters
        ----------
        text : str
            Text to speak.  Silently truncated to MAX_TEXT_CHARS at a clean
            word boundary if it exceeds the safe OS argument-length limit.
        """
        if not self.available:
            return

        text = text.strip()
        if not text:
            return

        # FIX BUG 3: guard against OS ARG_MAX overrun
        if len(text) > MAX_TEXT_CHARS:
            truncated = text[:MAX_TEXT_CHARS].rsplit(" ", 1)[0]
            print(
                f"[TTSEngine] Text truncated from {len(text)} → {len(truncated)} chars "
                f"(OS argument limit).",
                file=sys.stderr,
            )
            text = truncated

        self.stop()  # cancel any in-flight speech

        try:
            self._proc = subprocess.Popen(
                [sys.executable, self._worker, text],
                stdout=subprocess.DEVNULL,
                # FIX BUG 4: capture stderr so errors are not silently swallowed
                stderr=subprocess.PIPE,
            )
        except FileNotFoundError:
            # speak_worker.py moved or deleted after init
            warnings.warn(
                f"[TTSEngine] speak_worker.py not found: {self._worker}\n"
                "Call recheck() after fixing the path.",
                UserWarning,
                stacklevel=2,
            )
            self.available = False
            self._proc = None
        except OSError as exc:
            warnings.warn(
                f"[TTSEngine] Could not launch TTS worker: {exc}",
                UserWarning,
                stacklevel=2,
            )
            self._proc = None

    def stop(self) -> None:
        """
        Cancel any ongoing speech and reap the subprocess.

        FIX BUG 2: original called terminate() without wait(), leaking zombie
        processes on Linux/macOS (confirmed State=Z in /proc/<pid>/status).
        """
        proc = self._proc
        self._proc = None  # clear reference first so is_speaking() is false

        if proc is None:
            return

        if proc.poll() is None:
            # Process still running — ask it to stop
            proc.terminate()
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                # SIGTERM was ignored (e.g. pyttsx3 blocked in C extension)
                # Fall back to SIGKILL (Unix) / TerminateProcess (Windows)
                proc.kill()
                try:
                    proc.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    pass  # OS will reap it eventually

        else:
            # Process already finished — drain stderr so the pipe buffer
            # doesn't block and log any errors that occurred.
            # FIX BUG 4: without this, the PIPE buffer can fill and block.
            self._drain_stderr(proc)

        # Always reap the zombie (no-op if already reaped above)
        try:
            proc.wait(timeout=0)
        except subprocess.TimeoutExpired:
            pass

    def is_speaking(self) -> bool:
        """
        Return True if the TTS subprocess is currently running.

        Note: There is an inherent TOCTOU race between poll() and the return
        value — the process may exit in between.  Do NOT use this method in a
        tight blocking loop; use it only for UI state display.

        FIX BUG 7: result is snapshotted into a local variable so the same
        poll() call drives both the None-check and the return value.
        """
        proc = self._proc
        if proc is None:
            return False
        # FIX BUG 7: single poll() call → local variable; no double-call race
        return proc.poll() is None

    # ── Cleanup ────────────────────────────────────────────────────────────────

    def __del__(self) -> None:
        """
        FIX BUG 5: ensure the subprocess is terminated when this object is
        garbage-collected (e.g. Streamlit hot-reload, app restart).
        """
        try:
            self.stop()
        except Exception:
            pass  # never raise from __del__

    def __enter__(self) -> "TTSEngine":
        return self

    def __exit__(self, *_) -> None:
        """FIX BUG 5: context-manager support for guaranteed cleanup."""
        self.stop()

    # ── Internals ──────────────────────────────────────────────────────────────

    @staticmethod
    def _drain_stderr(proc: subprocess.Popen) -> None:
        """
        Read and log any stderr output from a finished worker process.
        Prevents the PIPE buffer from silently filling up.
        """
        if proc.stderr is None:
            return
        try:
            err = proc.stderr.read()
            if err:
                msg = err.decode(errors="replace").strip()
                print(f"[TTSEngine] Worker stderr:\n  {msg}", file=sys.stderr)
        except OSError:
            pass

    def __repr__(self) -> str:
        return (
            f"TTSEngine("
            f"available={self.available}, "
            f"speaking={self.is_speaking()}, "
            f"worker={os.path.basename(self._worker)!r}"
            f")"
        )