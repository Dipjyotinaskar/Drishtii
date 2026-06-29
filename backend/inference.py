"""
inference.py  —  Inference wrapper around GestureEngine.
Returns (letter, confidence, finger_states) for each frame.

SAVE THIS FILE TO:
    C:\\Users\\DIPJYOTI\\Desktop\\SIGN\\SiglanR\\inter\\backend\\inference.py

The error you saw:
    can't open file '...\\backend\\inference_engine.py': [Errno 2] No such file or directory

Root causes:
  1. Wrong filename — app.py imports `from backend.inference import SignLanguagePredictor`
     so the file MUST be named `inference.py`, NOT `inference_engine.py`.
  2. The file did not exist at all — save THIS file into the backend/ folder.

Run check (from inter/ directory, venv active):
    python -c "from backend.inference import SignLanguagePredictor; print('OK')"
"""

from __future__ import annotations

import os
import sys
from typing import Optional, Tuple, Dict

# ── Path bootstrap ─────────────────────────────────────────────────────────────
# Ensures `import gesture_engine` works whether this module is imported as a
# package (from backend.inference import …) or run directly (python backend/inference.py).
_BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

# ── Import guard for GestureEngine ────────────────────────────────────────────
try:
    from gesture_engine import GestureEngine
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "[inference] Could not import GestureEngine.\n"
        "  Make sure gesture_engine.py exists inside the backend/ folder.\n"
        f"  Original error: {exc}"
    ) from exc


# ── Type aliases ───────────────────────────────────────────────────────────────
# (letter_or_None, confidence_0_to_1, finger_states_dict)
PredictResult = Tuple[Optional[str], float, Dict[str, bool]]

_FINGER_NAMES = ("thumb", "index", "middle", "ring", "pinky")
_NULL_FINGERS: Dict[str, bool] = {k: False for k in _FINGER_NAMES}


class SignLanguagePredictor:
    """
    Thin wrapper around GestureEngine that provides a stable predict() /
    reset() interface for the Streamlit frontend.

    Parameters
    ----------
    buffer_size : int
        Number of consecutive frames used by GestureEngine for smoothing.
        Larger values = more stable but slightly laggier detection.
    """

    def __init__(self, buffer_size: int = 15) -> None:
        if buffer_size < 1:
            raise ValueError(f"buffer_size must be >= 1, got {buffer_size}")
        self.engine = GestureEngine(buffer_size=buffer_size)

    # ── Public API ─────────────────────────────────────────────────────────────

    def predict(self, landmarks) -> PredictResult:
        """
        Run one classification step.

        Parameters
        ----------
        landmarks : list[list[float]] | None
            21 landmarks, each a [x, y, z] list from MediaPipe HandTracker,
            or None when no hand is visible.

        Returns
        -------
        letter : str | None
            Predicted ISL letter (e.g. ``"A"``), or ``None`` when confidence
            is too low or no hand is present.
        confidence : float
            Score in [0.0, 1.0].
        finger_states : dict[str, bool]
            Keys: ``thumb``, ``index``, ``middle``, ``ring``, ``pinky``.
            ``True`` = finger is extended.
        """
        # Guard: pass-through for missing hand
        if landmarks is None:
            return None, 0.0, dict(_NULL_FINGERS)

        try:
            letter, confidence, finger_states = self.engine.classify(landmarks)
        except Exception as exc:
            # Don't crash the camera loop on a bad frame — log and return null
            print(f"[inference] classify() error: {exc}", file=sys.stderr)
            return None, 0.0, dict(_NULL_FINGERS)

        # Normalise outputs so callers never see None confidence or missing keys
        letter     = letter if isinstance(letter, str) and letter else None
        confidence = float(confidence) if confidence is not None else 0.0
        confidence = max(0.0, min(1.0, confidence))

        if not isinstance(finger_states, dict):
            finger_states = dict(_NULL_FINGERS)
        else:
            # Fill any missing finger keys with False
            finger_states = {k: bool(finger_states.get(k, False)) for k in _FINGER_NAMES}

        return letter, confidence, finger_states

    def reset(self) -> None:
        """
        Clear GestureEngine's internal smoothing buffer.
        Call this when the camera is (re-)started so stale frames don't
        bleed into new predictions.
        """
        try:
            self.engine.reset()
        except Exception as exc:
            print(f"[inference] reset() error: {exc}", file=sys.stderr)

    # ── Convenience ────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"SignLanguagePredictor("
            f"buffer_size={self.engine.buffer_size if hasattr(self.engine, 'buffer_size') else '?'})"
        )


# ── Standalone smoke-test ──────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Running SignLanguagePredictor smoke-test …")
    predictor = SignLanguagePredictor(buffer_size=5)

    # Simulate a flat hand (all zeros) — engine should return without crashing
    fake_landmarks = [[0.0, 0.0, 0.0]] * 21
    letter, conf, fs = predictor.predict(fake_landmarks)
    print(f"  Fake landmarks → letter={letter!r}, conf={conf:.2f}, fingers={fs}")

    letter, conf, fs = predictor.predict(None)
    print(f"  No hand        → letter={letter!r}, conf={conf:.2f}, fingers={fs}")

    predictor.reset()
    print("  reset() OK")
    print("Smoke-test passed ✓")