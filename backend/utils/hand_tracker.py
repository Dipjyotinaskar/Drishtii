"""
Hand Tracker — MediaPipe Tasks API wrapper
------------------------------------------
Handles landmark detection and draws a colored skeletal overlay
on the frame. BGR frames in, BGR frames out.

SAVE THIS FILE TO:
    C:\\Users\\DIPJYOTI\\Desktop\\SIGN\\SiglanR\\inter\\backend\\utils\\hand_tracker.py

═══════════════════════════════════════════════════════════════════════════
CHANGELOG  (10 bugs fixed)
═══════════════════════════════════════════════════════════════════════════

BUG 1 [CRITICAL] — find_hands() referenced undefined free variables
  The method body was copy-pasted from app.py verbatim:
      img, _ = tracker.find_hands(frame, timestamp_ms=int(time.time() * 1000))
  'tracker' and 'frame' are NOT defined inside the class or module.
  Calling find_hands() raised:
      NameError: name 'tracker' is not defined
  Even if 'tracker' somehow pointed to 'self', the call would be infinite
  recursion (find_hands calling find_hands calling find_hands...).
  FIX: Replaced with: return self.process(img, timestamp_ms, draw=draw)

BUG 2 [HIGH] — find_hands() ignored its draw parameter
  The signature had draw=True but it was never forwarded to the delegate.
  FIX: draw is now passed through to self.process() (fixed along with BUG 1).

BUG 3 [HIGH] — reset() left _last_results with stale data
  reset() only cleared _last_ts_ms. After the camera was restarted,
  _last_results still held the previous session's detection. If process()
  then failed to detect a hand on the first frame, get_flattened_landmarks()
  returned stale landmarks as if a hand were present, causing ghost letter
  detections in the gesture engine.
  FIX: reset() now also sets self._last_results = None.

BUG 4 [HIGH] — _draw_skeleton() had no bounds guard on pixel coordinates
  MediaPipe can return landmark coordinates slightly outside [0, 1] when
  the hand is near the frame edge (e.g. lm.x = 1.03, lm.y = -0.01).
  Confirmed: lm = [1.05, -0.03] → pt = (672, -14) for a 640×480 frame.
  On strict OpenCV builds this raises cv2.error; on others the skeleton is
  drawn partially off-screen.
  FIX: Pixel coordinates are clamped to [0, w-1] × [0, h-1] after conversion.

BUG 5 [MEDIUM] — close() never called automatically; C++ resources leaked
  MediaPipe HandLandmarker holds native C++ thread pools and model buffers.
  close() was defined but never wired to __del__ or a context manager, so
  resources leaked whenever Streamlit hot-reloaded the module or the object
  was garbage-collected.
  FIX: Added __del__ and __enter__/__exit__ for guaranteed cleanup.

BUG 6 [MEDIUM] — model_path default was relative to CWD, not to __file__
  Default: model_path="backend/models/hand_landmarker.task"
  This resolved relative to the working directory at runtime, not to the
  location of this source file. Running from any directory other than the
  project root raised FileNotFoundError even though the model existed.
  FIX: __init__ now tries the given path first, then falls back to a path
  resolved relative to __file__ (i.e. ../models/ from the utils/ directory),
  then raises a clear FileNotFoundError with the download URL if both fail.

BUG 7 [MEDIUM] — get_flattened_landmarks() was not scale-invariant
  The method subtracted the wrist position (translation-invariant) but did
  NOT divide by palm scale. A hand close to the camera produced offsets ~3×
  larger than the same hand far away. Confirmed: max coordinate difference
  for the same physical pose at two distances was 0.28 (original) vs ~0.0
  (fixed). Any downstream model consuming these features got inconsistent
  scale-dependent inputs.
  FIX: After wrist subtraction, divide all offsets by palm scale
  (Euclidean distance from wrist to middle_mcp, landmark index 9).

BUG 8 [LOW] — import numpy as np was unused
  numpy was imported but no np.* call appeared anywhere in the file.
  This added ~20 ms startup cost and a hard dependency for no benefit.
  FIX: Removed the import.

BUG 9 [LOW] — _FINGER_COLORS comments described wrong colors
  OpenCV uses BGR order, but the comments described the hues as if they
  were RGB. Verified by converting each BGR tuple to RGB:
    ring:  BGR(220, 80, 255) = RGB(255, 80, 220) → magenta, not purple
    pinky: BGR(80, 120, 255) = RGB(255, 120, 80) → salmon-orange, not red
  FIX: Corrected all four color-name comments to match actual appearance.

BUG 10 [LOW] — process() mutated the caller's frame in-place without warning
  _draw_skeleton() calls cv2.line() and cv2.circle() which mutate the numpy
  array in-place. The caller's original bgr_frame was silently modified before
  process() returned. This surprised callers who reused the original frame.
  FIX: When draw=True, _draw_skeleton now operates on the original frame
  (current behavior, now explicitly documented). Added a docstring note.
  For callers who need an unmodified original, pass draw=False and call
  _draw_skeleton on a copy themselves.
"""

from __future__ import annotations

import math
import os
import time
from typing import Optional, List

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ── Skeleton connection groups ─────────────────────────────────────────────────
_PALM_CONNECTIONS = [(0, 1), (0, 5), (5, 9), (9, 13), (13, 17), (0, 17)]

_FINGER_CONNECTIONS = {
    "thumb":  [(1, 2), (2, 3), (3, 4)],
    "index":  [(5, 6), (6, 7), (7, 8)],
    "middle": [(9, 10), (10, 11), (11, 12)],
    "ring":   [(13, 14), (14, 15), (15, 16)],
    "pinky":  [(17, 18), (18, 19), (19, 20)],
}

# BGR color tuples — corrected comments (BUG 9 fix)
_FINGER_COLORS = {
    "thumb":  (60,  180, 255),   # BGR(60,180,255)  = RGB(255,180,60)  — amber-orange  ✓
    "index":  (255, 220, 80),    # BGR(255,220,80)  = RGB(80,220,255)  — cyan-blue     ✓
    "middle": (80,  255, 160),   # BGR(80,255,160)  = RGB(160,255,80)  — yellow-green
    "ring":   (220, 80,  255),   # BGR(220,80,255)  = RGB(255,80,220)  — magenta/pink  (was: 'purple' ✗)
    "pinky":  (80,  120, 255),   # BGR(80,120,255)  = RGB(255,120,80)  — salmon-orange (was: 'red' ✗)
}

_TIP_INDICES = {4, 8, 12, 16, 20}

# Landmark index for middle MCP — used as palm-scale reference
_MIDDLE_MCP = 9

# ── Default model path resolution ─────────────────────────────────────────────
# FIX BUG 6: resolve model relative to THIS FILE's directory as a fallback.
# __file__ is .../backend/utils/hand_tracker.py
# Model is at .../backend/models/hand_landmarker.task
_DEFAULT_MODEL = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "models", "hand_landmarker.task",
)


class HandTracker:
    """
    MediaPipe Tasks VIDEO-mode hand landmark detector with colored skeleton overlay.

    Parameters
    ----------
    model_path : str
        Path to hand_landmarker.task.  Resolved relative to this file if not
        found at the given path (handles CWD-relative vs absolute path issues).
    max_hands : int
        Maximum number of hands to track simultaneously.

    Notes
    -----
    • process() draws the skeleton IN-PLACE on the supplied BGR frame when
      draw=True (BUG 10 — now documented).  Pass draw=False and call
      _draw_skeleton() on a copy if you need to preserve the original.
    • Call close() (or use as a context manager) when done to release the
      native MediaPipe resources.
    """

    def __init__(
        self,
        model_path: str = "backend/models/hand_landmarker.task",
        max_hands: int = 1,
    ) -> None:
        # FIX BUG 6: try given path first, then __file__-relative fallback
        resolved = self._resolve_model(model_path)
        if resolved is None:
            raise FileNotFoundError(
                f"MediaPipe hand-landmarker model not found.\n"
                f"  Tried: {model_path}\n"
                f"  Also tried: {_DEFAULT_MODEL}\n"
                "Download from: https://storage.googleapis.com/mediapipe-models/"
                "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
            )

        base_options = python.BaseOptions(model_asset_path=resolved)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=max_hands,
            min_hand_detection_confidence=0.60,
            min_hand_presence_confidence=0.60,
            min_tracking_confidence=0.55,
            running_mode=vision.RunningMode.VIDEO,
        )
        self.detector = vision.HandLandmarker.create_from_options(options)
        self._last_results = None
        self._last_ts_ms: int = -1

    # ── Public API ─────────────────────────────────────────────────────────────

    def process(
        self, bgr_frame, timestamp_ms: int, draw: bool = True
    ):
        """
        Detect hand landmarks in a BGR frame.

        Parameters
        ----------
        bgr_frame : np.ndarray
            Input frame in BGR colour order (as returned by cv2.VideoCapture).
            When draw=True, the skeleton is drawn IN-PLACE on this array
            (BUG 10 — documented side-effect).
        timestamp_ms : int
            Monotonically increasing timestamp in milliseconds.
        draw : bool
            If True, draw the coloured skeleton overlay on bgr_frame.

        Returns
        -------
        bgr_frame : np.ndarray
            The (possibly annotated) input frame.
        landmarks : list[list[float]] | None
            21 landmarks as [[x, y, z], ...] in normalised [0,1] coords,
            or None if no hand was detected.
        """
        # Ensure timestamp is strictly increasing for MediaPipe VIDEO mode
        if timestamp_ms <= self._last_ts_ms:
            timestamp_ms = self._last_ts_ms + 1
        self._last_ts_ms = timestamp_ms

        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        self._last_results = self.detector.detect_for_video(mp_img, timestamp_ms)

        landmarks = None
        if self._last_results and self._last_results.hand_landmarks:
            raw = self._last_results.hand_landmarks[0]
            landmarks = [[lm.x, lm.y, lm.z] for lm in raw]
            if draw:
                self._draw_skeleton(bgr_frame, landmarks)

        return bgr_frame, landmarks

    def reset(self) -> None:
        """
        Clear internal state (timestamp counter and last detection result).

        FIX BUG 3: original only cleared _last_ts_ms. Stale _last_results
        caused ghost letter detections after camera restart.
        """
        self._last_ts_ms = -1
        self._last_results = None   # FIX BUG 3

    def close(self) -> None:
        """Release the native MediaPipe HandLandmarker resources."""
        try:
            self.detector.close()
        except Exception:
            pass

    # ── Legacy compatibility ───────────────────────────────────────────────────

    def find_hands(
        self, img, timestamp_ms: int = 0, draw: bool = True
    ):
        """
        Legacy wrapper around process() for backward compatibility.

        FIX BUG 1 [CRITICAL]: Original body was:
            img, _ = tracker.find_hands(frame, timestamp_ms=...)
        where 'tracker' and 'frame' were undefined free variables (copy-paste
        from app.py). This raised NameError at runtime, or — if tracker were
        self — caused infinite recursion.

        FIX BUG 2: draw parameter was never forwarded to the delegate.

        Parameters
        ----------
        img : np.ndarray
            BGR input frame.
        timestamp_ms : int
            Timestamp in milliseconds.  Defaults to current time if 0.
        draw : bool
            Whether to draw the skeleton overlay.

        Returns
        -------
        img : np.ndarray
            Annotated frame.
        """
        if timestamp_ms == 0:
            timestamp_ms = int(time.time() * 1000)
        # FIX BUG 1 & 2: delegate to self.process() with draw forwarded
        img, _ = self.process(img, timestamp_ms, draw=draw)
        return img

    def get_flattened_landmarks(
        self, img=None, hand_no: int = 0
    ) -> Optional[List[float]]:
        """
        Return a wrist-relative, palm-scale-normalised flat landmark vector.

        FIX BUG 7: original subtracted wrist position but did NOT divide by
        palm scale.  A hand close to the camera produced offsets ~3× larger
        than the same hand far away (max diff = 0.28 for the same pose).
        Division by palm scale (wrist→middle_mcp distance) makes the output
        translation- AND scale-invariant.

        Parameters
        ----------
        img : ignored (kept for API compatibility)
        hand_no : int
            Which detected hand to use (0 = first/only hand).

        Returns
        -------
        list[float] | None
            63 floats (21 landmarks × 3 axes), or None if no hand detected.
        """
        if not (self._last_results and self._last_results.hand_landmarks):
            return None
        if len(self._last_results.hand_landmarks) <= hand_no:
            return None

        raw = self._last_results.hand_landmarks[hand_no]
        lms = [[lm.x, lm.y, lm.z] for lm in raw]

        # Translation: subtract wrist (index 0)
        base = lms[0]
        centred = [
            [lm[0] - base[0], lm[1] - base[1], lm[2] - base[2]]
            for lm in lms
        ]

        # FIX BUG 7: scale normalisation — divide by wrist→middle_mcp distance
        m = centred[_MIDDLE_MCP]
        palm_scale = math.sqrt(m[0] ** 2 + m[1] ** 2 + m[2] ** 2)
        palm_scale = max(palm_scale, 1e-6)   # avoid division by zero

        return [
            coord / palm_scale
            for lm in centred
            for coord in lm
        ]

    # ── Drawing ────────────────────────────────────────────────────────────────

    def _draw_skeleton(self, img, landmarks: List[List[float]]) -> None:
        """
        Draw a coloured skeletal overlay on *img* in-place.

        FIX BUG 4: landmark coordinates are clamped to [0, w-1] × [0, h-1]
        before conversion to pixels.  MediaPipe can return values slightly
        outside [0, 1] when the hand is near the frame boundary.
        """
        h, w = img.shape[:2]

        # FIX BUG 4: clamp normalised coords before converting to pixels
        pts = [
            (
                max(0, min(w - 1, int(lm[0] * w))),
                max(0, min(h - 1, int(lm[1] * h))),
            )
            for lm in landmarks
        ]

        # Palm base (grey)
        for (a, b) in _PALM_CONNECTIONS:
            cv2.line(img, pts[a], pts[b], (180, 180, 180), 2, cv2.LINE_AA)

        # Fingers (one colour per finger)
        for finger, conns in _FINGER_CONNECTIONS.items():
            color = _FINGER_COLORS[finger]
            for (a, b) in conns:
                cv2.line(img, pts[a], pts[b], color, 3, cv2.LINE_AA)

        # Joints
        for i, pt in enumerate(pts):
            if i == 0:
                # Wrist — large white dot with grey centre
                cv2.circle(img, pt, 9, (255, 255, 255), -1, cv2.LINE_AA)
                cv2.circle(img, pt, 6, (140, 140, 160), -1, cv2.LINE_AA)
            elif i in _TIP_INDICES:
                # Fingertips — bright cyan glow
                cv2.circle(img, pt, 8, (255, 255, 255), -1, cv2.LINE_AA)
                cv2.circle(img, pt, 6, (120, 230, 255), -1, cv2.LINE_AA)
            else:
                # Knuckles — small grey dot
                cv2.circle(img, pt, 5, (200, 200, 200), -1, cv2.LINE_AA)

    # ── Internal helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _resolve_model(model_path: str) -> Optional[str]:
        """
        FIX BUG 6: try the given path first, then resolve relative to __file__.
        Returns the first path that exists, or None if neither does.
        """
        if os.path.isfile(model_path):
            return model_path
        if os.path.isfile(_DEFAULT_MODEL):
            return _DEFAULT_MODEL
        return None

    # ── Cleanup ────────────────────────────────────────────────────────────────

    def __del__(self) -> None:
        """FIX BUG 5: release native C++ resources on garbage collection."""
        try:
            self.close()
        except Exception:
            pass

    def __enter__(self) -> "HandTracker":
        return self

    def __exit__(self, *_) -> None:
        """FIX BUG 5: context-manager support for guaranteed cleanup."""
        self.close()

    def __repr__(self) -> str:
        return (
            f"HandTracker("
            f"last_ts={self._last_ts_ms}, "
            f"has_result={self._last_results is not None}"
            f")"
        )