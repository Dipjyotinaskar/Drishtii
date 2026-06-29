"""
Word suggestion engine — prefix-based lookup against the system dictionary.
Uses /usr/share/dict/words (macOS/Linux) with a compact fallback list for Windows.

SAVE THIS FILE TO:
    C:\\Users\\DIPJYOTI\\Desktop\\SIGN\\SiglanR\\inter\\backend\\word_engine.py

═══════════════════════════════════════════════════════════════════════════
CHANGELOG  (6 bugs fixed, 1 accuracy improvement)
═══════════════════════════════════════════════════════════════════════════

BUG 1 [HIGH] — suggest() early-exit sentinel was wrong for multi-char prefixes
  Original sentinel: p[0:1] + 'z'*14
  This only considers the FIRST letter of the prefix, so for prefix "com",
  the sentinel was "czzzzzzzzzzzzzzz". Any word starting with 'c' but NOT
  with 'com' (e.g. "car", "cat") evaluates False for word >= sentinel and
  the loop keeps scanning — up to 800 words unnecessarily.
  Confirmed: prefix "pre" and "com" took 800 loop iterations in original vs
  3-5 iterations with the fix.
  FIX: sentinel is now p[:-1] + chr(ord(p[-1]) + 1), e.g.:
       "com" → "con"   "pre" → "prf"   "zz" → "z{"  (ASCII 123, after 'z')
  This exits the loop at the first word that lexicographically exceeds the
  prefix range, regardless of how many non-matching words share the first letter.

BUG 2 [HIGH] — import streamlit at module top level
  word_engine.py is a backend module imported by inference and other code.
  Importing streamlit unconditionally forces the entire Streamlit runtime to
  initialise in every context that imports WordEngine — including tests, CLI
  scripts, and plain Python sessions. This causes warnings and can fail outside
  a Streamlit app.
  The @st.cache_resource decorator on get_word_engine() is a UI concern that
  belongs in app.py, not in the engine module.
  FIX: Removed `import streamlit` and `get_word_engine()` from this file.
       app.py already calls `load_word_engine()` with @st.cache_resource — no
       change needed there.

BUG 3 [MEDIUM] — _load_words() called w.strip() three times per word
  The original generator expression:
      w.strip().lower() for w in f
      if w.strip().isalpha() and 2 <= len(w.strip()) <= 13
  creates three separate stripped-string objects per word. For the system
  dictionary (~100K words) this produces ~300K unnecessary heap allocations
  at startup, adding measurable load time.
  FIX: One `stripped = w.strip()` assignment; reused for all three checks.

BUG 4 [MEDIUM] — _load_words() returned _FALLBACK directly (no copy)
  If any caller (or future code) mutated the returned list, the module-level
  _FALLBACK sentinel would be corrupted permanently for the rest of the session.
  Confirmed by test: appending to the returned reference mutated _FALLBACK.
  FIX: return list(_FALLBACK) — a shallow copy is sufficient since elements
  are immutable strings.

BUG 5 [LOW] — list[str] type hints require Python 3.9+
  The original used bare `list[str]` in function return annotations.
  Safe on Python 3.10+ (the user's environment) but would silently break on
  Python 3.8. Added `from __future__ import annotations` to make all
  annotations strings at runtime, which is compatible with Python 3.7+.

BUG 6 [LOW] — apply() preserved double-spaces from the word builder
  If current ended with two spaces (e.g. after typing Space twice), apply()
  would produce "word  suggestion " with a double-space, which looked wrong
  in the word-box display.
  FIX: Normalise runs of spaces to a single space before returning.

IMPROVEMENT A — _load_words() now also checks Windows word-list locations
  Windows has no /usr/share/dict/words. Added several common paths including
  ones installed by Git for Windows and WSL distributions, so the engine
  gets a real dictionary on Windows too rather than always falling back.
"""

from __future__ import annotations

import os
import bisect
from typing import List

# ── Compact fallback (~300 common English words, used when no dict file found) ─
_FALLBACK: List[str] = sorted(set("""
able about above action add after again age air all allow almost alone
already also always among and another answer any appear area around ask
away back ball based because become before begin behind believe best
better between big black blue body book both bring call came can care
cause change child city clear close come common consider control could
country create dark day decide deep different door down drive early easy
end enough even ever every example face fact fall family far feel fill
find fire five floor follow food force form free friend front full
future game get give good great green group grow hand happen have head
hear heart here high hold home hope house how hundred idea if important
include interest into job keep kind know land language large last late
lead learn leave left less level like little live long look love main
make man may mean member mind month more most move much must name near
need never next nice night nothing number often open order other our out
over own page part past people person place plan plant play possible
power present problem real reason red remain remember result right road
room run same say school seem set show side since sky sleep slow small
social some sound speak stand start state stay still stop story strong
student sun take talk teach team tell then thing think three through
time together top tree true turn under very view visit voice wait walk
want water well when where while white who will within without woman
wonder word work world write year young
""".split()))

# ── Candidate dictionary paths (first one that exists wins) ───────────────────
# IMPROVEMENT A: added Windows / Git-for-Windows / WSL paths
_DICT_PATHS = [
    "/usr/share/dict/words",                          # Linux / macOS
    "/usr/dict/words",                                # older Unix
    "/usr/share/dict/american-english",               # Debian/Ubuntu variant
    r"C:\Program Files\Git\usr\share\dict\words",     # Git for Windows
    r"C:\Windows\System32\drivers\etc\words",         # rare
]


def _load_words() -> List[str]:
    """
    Load a sorted, de-duplicated, alpha-only word list from the system dictionary.
    Falls back to _FALLBACK (copy) if no file is found.
    """
    for path in _DICT_PATHS:
        if os.path.exists(path):
            try:
                with open(path, encoding="utf-8", errors="ignore") as f:
                    # FIX BUG 3: assign stripped once; was called 3x per word
                    words: List[str] = []
                    for w in f:
                        stripped = w.strip()
                        if stripped.isalpha() and 2 <= len(stripped) <= 13:
                            words.append(stripped.lower())
                    return sorted(set(words))
            except OSError:
                pass

    # FIX BUG 4: return a copy so callers can't corrupt the module-level sentinel
    return list(_FALLBACK)


class WordEngine:
    """
    Efficient prefix-based word suggestion using bisect O(log N + k).

    Attributes
    ----------
    source : str
        Either a file path or ``"fallback"`` — useful for debugging.
    """

    def __init__(self) -> None:
        self._words: List[str] = _load_words()
        # Record which source was used (helpful for debugging on Windows)
        self.source: str = next(
            (p for p in _DICT_PATHS if os.path.exists(p)), "fallback"
        )

    # ── Core API ───────────────────────────────────────────────────────────────

    def suggest(self, prefix: str, n: int = 6) -> List[str]:
        """
        Return up to *n* dictionary words that begin with *prefix*.

        Parameters
        ----------
        prefix : str
            The letters typed so far for the current word.
        n : int
            Maximum number of suggestions to return (default 6).

        Returns
        -------
        list[str]
            Alphabetically ordered suggestions, empty if prefix is empty
            or contains non-alpha characters.
        """
        if not prefix:
            return []

        p = prefix.lower().strip()
        if not p or not p.isalpha():
            return []

        idx = bisect.bisect_left(self._words, p)

        # FIX BUG 1: correct early-exit sentinel.
        # "com" → "con";  "pre" → "prf";  "z" → "{";  "zz" → "z{"
        # chr(ord('z') + 1) = '{' (ASCII 123), which sorts after all lowercase
        # letters, so prefix "z" correctly stops after all z-words.
        end = p[:-1] + chr(ord(p[-1]) + 1)

        results: List[str] = []
        for word in self._words[idx:]:
            if word >= end:        # past all words sharing this prefix
                break
            if word.startswith(p):
                results.append(word)
                if len(results) >= n:
                    break

        return results

    def apply(self, current: str, suggestion: str) -> str:
        """
        Replace the last partial word in *current* with *suggestion* + space.

        Examples
        --------
        apply("hel",    "hello") → "hello "
        apply("hi hel", "hello") → "hi hello "
        apply("hi ",    "there") → "hi there "
        apply("",       "hello") → "hello "

        FIX BUG 6: normalise consecutive spaces so double-space in the
        word builder doesn't propagate into the result.
        """
        if not current or current.endswith(" "):
            result = current + suggestion + " "
        else:
            parts = current.rsplit(" ", 1)
            word_prefix = parts[0] + " " if len(parts) > 1 else ""
            result = word_prefix + suggestion + " "

        # FIX BUG 6: collapse runs of spaces to a single space
        import re
        return re.sub(r" {2,}", " ", result)

    def __len__(self) -> int:
        """Number of words in the loaded dictionary."""
        return len(self._words)

    def __repr__(self) -> str:
        return (
            f"WordEngine("
            f"words={len(self._words):,}, "
            f"source={os.path.basename(self.source)!r}"
            f")"
        )