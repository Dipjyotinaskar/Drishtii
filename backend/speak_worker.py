"""
speak_worker.py — Minimal subprocess entry-point for pyttsx3 TTS.
Called as:  python speak_worker.py <text to speak>

Running pyttsx3 in its own process avoids threading conflicts with Streamlit.

FIX:  ModuleNotFoundError: No module named 'pyttsx3'
      Run this once in your project environment:
          pip install pyttsx3

      On Windows, pyttsx3 uses the built-in SAPI5 engine — no extra
      drivers are needed.  On Linux you also need espeak:
          sudo apt-get install espeak espeak-ng
      On macOS, nsss / AVFoundation is used automatically.

USAGE:
    python speak_worker.py Hello world
    python speak_worker.py "Hello world"     # quotes optional

OPTIONAL ENV OVERRIDES:
    TTS_RATE    words-per-minute  (default 150)
    TTS_VOLUME  0.0 – 1.0        (default 1.0)
    TTS_VOICE   index or partial name string (default: system default)

EXIT CODES:
    0  — success
    1  — pyttsx3 not installed   (pip install pyttsx3)
    2  — no text supplied
    3  — TTS engine runtime error
"""

from __future__ import annotations
import os
import sys


# ── Friendly import guard ─────────────────────────────────────────────────────
try:
    import pyttsx3
except ModuleNotFoundError:
    print(
        "[speak_worker] ERROR: pyttsx3 is not installed.\n"
        "  Fix: pip install pyttsx3\n"
        "  Then re-run this script.",
        file=sys.stderr,
    )
    sys.exit(1)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _parse_voice(engine: pyttsx3.Engine, voice_hint: str) -> str | None:
    """
    Return a voice ID matching `voice_hint` (case-insensitive substring match
    on voice name or ID).  Returns None if no match — engine keeps its default.
    """
    if not voice_hint:
        return None
    voices = engine.getProperty("voices")
    voice_hint_lower = voice_hint.lower()
    for v in voices:
        if voice_hint_lower in v.name.lower() or voice_hint_lower in v.id.lower():
            return v.id
    return None


def _env_float(key: str, default: float, lo: float, hi: float) -> float:
    """Read a float from env, clamped to [lo, hi], fallback to default."""
    raw = os.environ.get(key, "")
    try:
        return max(lo, min(hi, float(raw)))
    except (ValueError, TypeError):
        return default


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    # ── 1. Gather text ────────────────────────────────────────────────────────
    if len(sys.argv) < 2:
        print("[speak_worker] No text supplied.", file=sys.stderr)
        sys.exit(2)

    text = " ".join(sys.argv[1:]).strip()
    if not text:
        print("[speak_worker] Empty text — nothing to speak.", file=sys.stderr)
        sys.exit(2)

    # ── 2. Read optional env overrides ───────────────────────────────────────
    rate   = int(_env_float("TTS_RATE",   150.0, 50.0, 400.0))
    volume = _env_float("TTS_VOLUME", 1.0, 0.0, 1.0)
    voice_hint = os.environ.get("TTS_VOICE", "").strip()

    # ── 3. Initialise engine ──────────────────────────────────────────────────
    try:
        engine = pyttsx3.init()
    except RuntimeError as exc:
        # Raised on Linux when espeak/espeak-ng is missing
        print(
            f"[speak_worker] Could not initialise TTS engine: {exc}\n"
            "  On Linux: sudo apt-get install espeak espeak-ng\n"
            "  On macOS: should work out of the box.\n"
            "  On Windows: SAPI5 is built-in; reinstall pyttsx3 if this fails.",
            file=sys.stderr,
        )
        sys.exit(3)
    except Exception as exc:
        print(f"[speak_worker] Unexpected init error: {exc}", file=sys.stderr)
        sys.exit(3)

    # ── 4. Apply properties ───────────────────────────────────────────────────
    try:
        engine.setProperty("rate",   rate)
        engine.setProperty("volume", volume)

        voice_id = _parse_voice(engine, voice_hint)
        if voice_id:
            engine.setProperty("voice", voice_id)

    except Exception as exc:
        # Non-fatal — continue with defaults
        print(f"[speak_worker] Warning — could not set property: {exc}", file=sys.stderr)

    # ── 5. Speak ──────────────────────────────────────────────────────────────
    try:
        engine.say(text)
        engine.runAndWait()
    except RuntimeError as exc:
        # runAndWait can raise if the loop is already running (shouldn't happen
        # in subprocess mode, but guard anyway)
        print(f"[speak_worker] Runtime error during speech: {exc}", file=sys.stderr)
        sys.exit(3)
    except KeyboardInterrupt:
        # Graceful Ctrl-C (e.g. parent process killed the subprocess)
        engine.stop()
        sys.exit(0)
    except Exception as exc:
        print(f"[speak_worker] Unexpected error: {exc}", file=sys.stderr)
        sys.exit(3)


if __name__ == "__main__":
    main()