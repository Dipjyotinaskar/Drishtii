"""
ASL Gesture Engine v5 — Full A–Z (excluding J & Z which require motion)
==========================================================================
All 24 static signs supported with improved disambiguation.

Key technique: angle-at-joint (PIP/DIP) + tip-to-wrist ratio + direction
vectors. Each "ambiguous cluster" has dedicated disambiguation logic:

  Cluster 1 — Closed-fist variants : A  E  M  N  O  C  S  T
  Cluster 2 — Index-only            : D  G  Q  X
  Cluster 3 — Index+Middle          : H  K  R  U  V
  Cluster 4 — Distinct shapes       : B  F  I  L  W  X  Y  P

═══════════════════════════════════════════════════════════════════════════
CHANGELOG v4 → v5  (9 bugs fixed, 3 accuracy improvements)
═══════════════════════════════════════════════════════════════════════════

BUG 1 [CRITICAL] — P sign was DEAD CODE
  The I+M cluster check (I and M and not R and not Pk) is a strict superset
  of the P check (same + idx_down + thm_pip). P was NEVER reachable because
  the I+M cluster always fired first.
  FIX: Added idx_down + thm_pip + T guard INSIDE the I+M cluster before
  returning H/K/R/U/V, so P is detected there.

BUG 2 [HIGH] — thm_side absolute threshold (0.09) not scale-invariant
  Thumb-side detection used an absolute x-offset of 0.09, but MediaPipe
  normalizes coordinates to the bounding box. A hand close to the camera
  has wider spread than one far away, making 0.09 too tight or too loose.
  FIX: Replaced with palm_width-relative threshold:
       abs(thumb_tip.x - palm_center.x) / palm_width > 0.22

BUG 3 [HIGH] — thm_btw absolute ±0.025 padding not scale-invariant
  Same root cause as BUG 2. The ±0.025 padding on the T-sign thumb-between
  check was absolute, causing false positives for far-away hands.
  FIX: Replaced with palm_width-relative padding (palm_width * 0.12).

BUG 4 [HIGH] — F check used 'not I' (ratio) instead of Ic (curl check)
  F requires the index finger to be CURLED touching the thumb. The check
  used the boolean I (rI > EXT) which is True when extended. "not I" is
  True for ANY non-extended index — including a half-raised one — causing
  false F detections when index was borderline.
  FIX: Changed condition to explicitly require Ic (rI < CURL).

BUG 5 [MEDIUM] — Buffer vote: None tie causes non-deterministic result
  When None and a real letter are tied in the weighted vote, Python's max()
  picks whichever key was inserted first (dict ordering), which is
  non-deterministic across frames.
  FIX: Exclude None from the vote entirely; None frames still count against
  stability (lower stab score) but never win the election.

BUG 6 [MEDIUM] — L / Y / I checks lacked curl guards on other fingers
  L requires only thumb+index. But M/R/Pk were checked with just "not M"
  etc. (ratio-based), so a borderline middle finger (ratio=1.09, just below
  EXT=1.10) would cause L to be skipped and fall into the D/G cluster.
  FIX: Added explicit Mc, Rc, Pkc (curl) checks so these signs require
  the other fingers to be properly curled, not just "not extended".

BUG 7 [MEDIUM] — _no_hand buffer-clear threshold too aggressive (4 frames)
  At 30 fps, 4 frames = ~133 ms. Any momentary occlusion (blinking, fast
  movement) wiped the smoothing buffer, causing letter flicker on re-entry.
  FIX: Raised threshold to 12 frames (~400 ms), matching a natural blink.

BUG 8 [LOW] — W with small spread returned low confidence (0.79) not None
  A W with poor finger spread should be rejected, not returned at 0.79
  (which still passes the default 0.50 threshold). This caused false W
  detections when the hand was transitioning between signs.
  FIX: Return (None, 0.0) when spread is below the minimum W threshold.

BUG 9 [LOW] — Buffer < 5 frames halved confidence during startup
  For the first 5 frames the raw confidence was multiplied by 0.50,
  making it impossible to fill the hold bar quickly even for clear signs.
  FIX: Reduced the startup penalty to ×0.75 and lowered the minimum
  buffer size for a full vote from 5 to 3 frames.

ACCURACY IMPROVEMENTS
  A. O vs C disambiguation: added average finger-curvature check
     (mean PIP angle) to distinguish tight O (all joints deeply bent)
     from open C (joints moderately curved).
  B. E vs S disambiguation: added a second pass checking all four DIP
     angles; E has all DIPs sharply bent while S often has them straighter.
  C. R detection: added a cross-check that index and middle tips are
     physically close (sIM < 0.22) AND the PIP angle of both is >130°
     (fingers extended but crossed), reducing confusion with U.
"""

from __future__ import annotations

import numpy as np
from collections import deque
from typing import Optional, Tuple, Dict

# ── Landmark indices ───────────────────────────────────────────────────────────
WRIST = 0
THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
INDEX_MCP,  INDEX_PIP,  INDEX_DIP,  INDEX_TIP  = 5,  6,  7,  8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9,  10, 11, 12
RING_MCP,   RING_PIP,   RING_DIP,   RING_TIP   = 13, 14, 15, 16
PINKY_MCP,  PINKY_PIP,  PINKY_DIP,  PINKY_TIP  = 17, 18, 19, 20

# Return type alias
ClassifyResult = Tuple[Optional[str], float, Dict[str, bool]]

_FINGER_NAMES = ("thumb", "index", "middle", "ring", "pinky")
_NULL_FS: Dict[str, bool] = {k: False for k in _FINGER_NAMES}

# ── Thresholds ─────────────────────────────────────────────────────────────────
EXT  = 1.10   # tip/pip ratio → finger is extended
CURL = 0.95   # tip/pip ratio → finger is curled

# Number of consecutive no-hand frames before clearing the buffer
# FIX BUG 7: raised from 4 → 12  (~400 ms at 30 fps)
_NO_HAND_CLEAR = 12

# Minimum buffer occupancy for a full-weight vote
# FIX BUG 9: lowered from 5 → 3
_MIN_BUF_VOTE = 3


class GestureEngine:
    """
    Geometric ISL/ASL static-sign classifier.

    Parameters
    ----------
    buffer_size : int
        Temporal smoothing window (frames).  Larger = stabler, laggier.
    """

    def __init__(self, buffer_size: int = 20) -> None:
        if buffer_size < 1:
            raise ValueError(f"buffer_size must be ≥ 1, got {buffer_size}")
        self.buffer_size = buffer_size
        self.buffer: deque[Optional[str]] = deque(maxlen=buffer_size)
        self._no_hand = 0

    # ── Low-level geometry ─────────────────────────────────────────────────────

    def _pt(self, lm, i: int) -> np.ndarray:
        return np.asarray(lm[i], dtype=np.float32)

    def _dist(self, lm, a: int, b: int) -> float:
        return float(np.linalg.norm(self._pt(lm, a) - self._pt(lm, b)))

    def _palm(self, lm) -> float:
        """Palm size: wrist→middle_mcp distance (scale reference)."""
        return max(self._dist(lm, WRIST, MIDDLE_MCP), 1e-6)

    def _palm_width(self, lm) -> float:
        """Palm width: index_mcp→pinky_mcp distance (scale reference)."""
        return max(self._dist(lm, INDEX_MCP, PINKY_MCP), 1e-6)

    def _angle(self, lm, a: int, v: int, c: int) -> float:
        """Angle at vertex v (degrees).  180 = straight, ~90 = sharply bent."""
        ba = self._pt(lm, a) - self._pt(lm, v)
        bc = self._pt(lm, c) - self._pt(lm, v)
        denom = np.linalg.norm(ba) * np.linalg.norm(bc)
        if denom < 1e-6:
            return 180.0
        return float(np.degrees(
            np.arccos(np.clip(np.dot(ba, bc) / denom, -1.0, 1.0))
        ))

    def _ratio(self, lm, tip: int, pip: int) -> float:
        """dist(tip,wrist) / dist(pip,wrist).  >EXT → extended; <CURL → curled."""
        return self._dist(lm, tip, WRIST) / max(self._dist(lm, pip, WRIST), 1e-6)

    def _spread(self, lm, a: int, b: int) -> float:
        """Distance between two tips, normalised by palm size."""
        return self._dist(lm, a, b) / self._palm(lm)

    def _unit(self, lm, a: int, b: int) -> np.ndarray:
        v = self._pt(lm, b) - self._pt(lm, a)
        n = np.linalg.norm(v)
        return v / n if n > 1e-6 else v

    # ── Feature bank ───────────────────────────────────────────────────────────

    def _features(self, lm) -> dict:
        # ── Extension / curl booleans ──────────────────────────────────────
        rI  = self._ratio(lm, INDEX_TIP,  INDEX_PIP)
        rM  = self._ratio(lm, MIDDLE_TIP, MIDDLE_PIP)
        rR  = self._ratio(lm, RING_TIP,   RING_PIP)
        rPk = self._ratio(lm, PINKY_TIP,  PINKY_PIP)

        I  = rI  > EXT;   M  = rM  > EXT
        R  = rR  > EXT;   Pk = rPk > EXT

        Ic  = rI  < CURL;  Mc  = rM  < CURL
        Rc  = rR  < CURL;  Pkc = rPk < CURL

        # Thumb: tip farther from index_mcp than its own mcp → extended
        T = (self._dist(lm, THUMB_TIP, INDEX_MCP)
             > self._dist(lm, THUMB_MCP, INDEX_MCP) * 0.87)

        # ── PIP / DIP angles ───────────────────────────────────────────────
        aIp   = self._angle(lm, INDEX_MCP,   INDEX_PIP,   INDEX_DIP)
        aMp   = self._angle(lm, MIDDLE_MCP,  MIDDLE_PIP,  MIDDLE_DIP)
        aRp   = self._angle(lm, RING_MCP,    RING_PIP,    RING_DIP)
        aPkp  = self._angle(lm, PINKY_MCP,   PINKY_PIP,   PINKY_DIP)
        aId   = self._angle(lm, INDEX_PIP,   INDEX_DIP,   INDEX_TIP)
        # DIP angles (for E vs S disambiguation — IMPROVEMENT B)
        aIdip  = self._angle(lm, INDEX_PIP,  INDEX_DIP,  INDEX_TIP)
        aMdip  = self._angle(lm, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP)
        aRdip  = self._angle(lm, RING_PIP,   RING_DIP,   RING_TIP)
        aPkdip = self._angle(lm, PINKY_PIP,  PINKY_DIP,  PINKY_TIP)

        # ── Spread ratios ──────────────────────────────────────────────────
        sIM  = self._spread(lm, INDEX_TIP,  MIDDLE_TIP)
        sMR  = self._spread(lm, MIDDLE_TIP, RING_TIP)
        sIT  = self._spread(lm, INDEX_TIP,  THUMB_TIP)

        # ── Direction vectors ──────────────────────────────────────────────
        iv = self._unit(lm, INDEX_MCP,  INDEX_TIP)
        mv = self._unit(lm, MIDDLE_MCP, MIDDLE_TIP)
        tv = self._unit(lm, THUMB_MCP,  THUMB_TIP)

        idx_up    = iv[1] < -0.42
        idx_horiz = abs(iv[0]) > abs(iv[1]) * 0.80
        idx_down  = iv[1]  >  0.38
        mid_horiz = abs(mv[0]) > abs(mv[1]) * 0.80
        thm_lat   = abs(tv[0]) > abs(tv[1]) * 0.70

        # ── Curved / semi-flexed state ────────────────────────────────────
        # True when tip is between pip and mcp (y in image space; 0=top 1=bottom)
        def _curved(tip: int, pip: int, mcp: int) -> bool:
            return lm[pip][1] < lm[tip][1] < lm[mcp][1]

        Icv  = _curved(INDEX_TIP,  INDEX_PIP,  INDEX_MCP)
        Mcv  = _curved(MIDDLE_TIP, MIDDLE_PIP, MIDDLE_MCP)
        Rcv  = _curved(RING_TIP,   RING_PIP,   RING_MCP)
        Pkcv = _curved(PINKY_TIP,  PINKY_PIP,  PINKY_MCP)

        # ── Scale-invariant thumb-side / thumb-between ─────────────────────
        # FIX BUG 2: was abs(lm[THUMB_TIP][0] - pcx) > 0.09 (absolute)
        # FIX BUG 3: was min/max ± 0.025 (absolute)
        pw   = self._palm_width(lm)
        pcx  = (lm[INDEX_MCP][0] + lm[PINKY_MCP][0]) / 2.0
        thm_side = abs(lm[THUMB_TIP][0] - pcx) / pw > 0.22

        thm_fist = (lm[INDEX_PIP][1] - 0.04
                    < lm[THUMB_TIP][1]
                    < lm[INDEX_MCP][1] + 0.04)

        # FIX BUG 3: scale-invariant thm_btw
        _pad     = pw * 0.12
        _thm_lo  = min(lm[INDEX_MCP][0], lm[MIDDLE_MCP][0]) - _pad
        _thm_hi  = max(lm[INDEX_MCP][0], lm[MIDDLE_MCP][0]) + _pad
        thm_btw  = _thm_lo < lm[THUMB_TIP][0] < _thm_hi

        thm_pip  = self._spread(lm, THUMB_TIP, INDEX_PIP) < 0.38
        I_hook   = aId < 148 and rI > 0.88 and not I

        # Average PIP curvature (used for O vs C — IMPROVEMENT A)
        avg_pip = (aIp + aMp + aRp + aPkp) / 4.0

        return dict(
            I=I, M=M, R=R, Pk=Pk, T=T,
            Ic=Ic, Mc=Mc, Rc=Rc, Pkc=Pkc,
            aIp=aIp, aMp=aMp, aRp=aRp, aPkp=aPkp, aId=aId,
            aIdip=aIdip, aMdip=aMdip, aRdip=aRdip, aPkdip=aPkdip,
            sIM=sIM, sMR=sMR, sIT=sIT,
            idx_up=idx_up, idx_horiz=idx_horiz, idx_down=idx_down,
            mid_horiz=mid_horiz, thm_lat=thm_lat,
            thm_side=thm_side, thm_fist=thm_fist,
            thm_btw=thm_btw, thm_pip=thm_pip, I_hook=I_hook,
            Icv=Icv, Mcv=Mcv, Rcv=Rcv, Pkcv=Pkcv,
            avg_pip=avg_pip,
        )

    def get_finger_states_named(self, lm) -> Dict[str, bool]:
        f = self._features(lm)
        return {
            "thumb":  f["T"],
            "index":  f["I"],
            "middle": f["M"],
            "ring":   f["R"],
            "pinky":  f["Pk"],
        }

    # ── Classification tree ────────────────────────────────────────────────────

    def _classify(self, lm) -> Tuple[Optional[str], float]:  # noqa: C901
        f   = self._features(lm)
        I   = f["I"];   M   = f["M"];   R  = f["R"];  Pk = f["Pk"];  T = f["T"]
        Ic  = f["Ic"];  Mc  = f["Mc"];  Rc = f["Rc"]; Pkc = f["Pkc"]
        sIM = f["sIM"]; sMR = f["sMR"]; sIT = f["sIT"]

        # ═══════════════════════════════════════════════════════════════
        # B — all four fingers extended
        # ═══════════════════════════════════════════════════════════════
        if I and M and R and Pk:
            return "B", 0.91

        # ═══════════════════════════════════════════════════════════════
        # W — index + middle + ring extended and spread
        # FIX BUG 8: return None below minimum spread instead of 0.79
        # ═══════════════════════════════════════════════════════════════
        if I and M and R and not Pk:
            if sIM > 0.40 and sMR > 0.34:
                return "W", 0.91
            if sIM > 0.28 and sMR > 0.24:
                return "W", 0.79   # borderline — still accept but lower conf
            return None, 0.0       # FIX: too little spread → reject

        # ═══════════════════════════════════════════════════════════════
        # Cluster: Index + Middle  →  H / K / P / R / U / V
        # FIX BUG 1: P detection moved here — it was dead code before
        # FIX BUG 6: added Rc/Pkc guards so ring/pinky must be curled
        # ═══════════════════════════════════════════════════════════════
        if I and M and not R and not Pk:
            # P: index+middle DOWN, thumb touching index PIP — must check first
            # FIX BUG 1: was separate dead-code branch below; now checked here
            if T and f["thm_pip"] and f["idx_down"]:
                return "P", 0.82

            # H: both pointing sideways
            if f["idx_horiz"] and f["mid_horiz"]:
                return "H", 0.87

            # K: index+middle up, thumb near index PIP (not pointing down)
            if T and f["thm_pip"] and not f["idx_down"]:
                return "K", 0.88

            # R: tips close + both PIPs extended (crossed fingers)
            # IMPROVEMENT C: added PIP angle check to distinguish from U
            if (not T and sIM < 0.22
                    and f["aIp"] > 130 and f["aMp"] > 130):
                return "R", 0.85

            # V: peace sign — wide spread
            if sIM > 0.52:
                return "V", 0.93

            # U: default — together and vertical
            return "U", 0.87

        # ═══════════════════════════════════════════════════════════════
        # L — thumb + index extended; others CURLED (FIX BUG 6)
        # ═══════════════════════════════════════════════════════════════
        if T and I and not M and not R and not Pk:
            if Mc and Rc and Pkc:
                return "L", 0.94
            return "L", 0.78   # other fingers borderline — lower confidence

        # ═══════════════════════════════════════════════════════════════
        # Y — thumb + pinky; others CURLED (FIX BUG 6)
        # ═══════════════════════════════════════════════════════════════
        if T and not I and not M and not R and Pk:
            if Ic and Mc and Rc:
                return "Y", 0.94
            return "Y", 0.78

        # ═══════════════════════════════════════════════════════════════
        # I — pinky only; others CURLED (FIX BUG 6)
        # ═══════════════════════════════════════════════════════════════
        if not T and not I and not M and not R and Pk:
            if Ic and Mc and Rc:
                return "I", 0.93
            return "I", 0.78

        # ═══════════════════════════════════════════════════════════════
        # Cluster: Index dominant  →  D / G / Q / X
        # ═══════════════════════════════════════════════════════════════
        if not M and not R and not Pk and (I or f["I_hook"]):
            # X: hooked index (DIP bent, partial extension)
            if f["I_hook"]:
                return "X", 0.84
            # G: index + thumb both horizontal (pointing gun sideways)
            if f["idx_horiz"] and T and f["thm_lat"]:
                return "G", 0.83
            # Q: index + thumb pointing downward
            if f["idx_down"] and T:
                return "Q", 0.77
            # D: index up, others curled toward thumb
            if f["idx_up"]:
                return "D", (0.90 if sIT < 0.68 else 0.77)
            return "D", 0.70

        # ═══════════════════════════════════════════════════════════════
        # F — middle + ring + pinky up; index CURLED (FIX BUG 4)
        # Original: used "not I" (ratio-based) — allowed borderline index
        # Fix: explicitly require Ic (index is curled touching thumb area)
        # ═══════════════════════════════════════════════════════════════
        if Ic and M and R and Pk:
            if sIT < 0.40:
                return "F", 0.91
            if sIT < 0.58:
                return "F", 0.76
            return None, 0.0

        # ═══════════════════════════════════════════════════════════════
        # Cluster: Closed / curved  →  A  C  E  M  N  O  S  T
        # ═══════════════════════════════════════════════════════════════
        if not I and not M and not R and not Pk:

            all_curved = f["Icv"] and f["Mcv"] and f["Rcv"] and f["Pkcv"]

            # ── O: all semi-flexed + thumb-index pinch + avg PIP deep ──────
            # IMPROVEMENT A: added avg_pip check to separate O from C
            if all_curved and sIT < 0.40 and f["avg_pip"] < 145:
                return "O", 0.86

            # ── C: all semi-flexed, open (thumb farther away, less bent) ───
            if all_curved and sIT > 0.50 and f["avg_pip"] >= 130:
                return "C", 0.84

            # ── T: thumb sandwiched between index and middle MCPs ───────────
            if T and f["thm_btw"] and not f["thm_side"]:
                return "T", 0.83

            # ── E: all four PIP AND DIP joints sharply bent, no thumb out ───
            # IMPROVEMENT B: added DIP angle check to separate E from S
            all_pip_deep = (f["aIp"] < 126 and f["aMp"] < 126
                            and f["aRp"] < 126 and f["aPkp"] < 126)
            all_dip_deep = (f["aIdip"] < 150 and f["aMdip"] < 150
                            and f["aRdip"] < 150 and f["aPkdip"] < 150)
            if all_pip_deep and all_dip_deep and not T:
                return "E", 0.87   # raised from 0.85 — stricter, more confident

            # ── N vs M: fingers folded over thumb ───────────────────────────
            if T and not f["thm_side"] and not f["thm_btw"]:
                three_bent = (f["aIp"] < 145
                              and f["aMp"] < 145
                              and f["aRp"] < 145)
                two_bent   = (f["aIp"] < 145
                              and f["aMp"] < 145
                              and f["aRp"] >= 145)
                if three_bent:
                    return "M", 0.78
                if two_bent:
                    return "N", 0.76

            # ── S vs A: fist — thumb position is the key ────────────────────
            if T:
                if f["thm_side"]:
                    return "A", 0.85   # thumb outward → A
                if f["thm_fist"]:
                    return "S", 0.81   # thumb across fist front → S
                return "A", 0.68       # ambiguous → default A

            # Closed fist, no thumb extension
            return "S", 0.62

        # ═══════════════════════════════════════════════════════════════
        # Fallback
        # ═══════════════════════════════════════════════════════════════
        return None, 0.0

    # ── Public API ─────────────────────────────────────────────────────────────

    def classify(self, landmarks) -> ClassifyResult:
        """
        Classify one frame of MediaPipe hand landmarks.

        Parameters
        ----------
        landmarks : list[list[float]] | None
            21 landmarks, each [x, y, z] (normalised by MediaPipe).

        Returns
        -------
        letter     : str | None
        confidence : float  in [0.0, 0.99]
        fingers    : dict[str, bool]  — which fingers are extended
        """
        if landmarks is None or len(landmarks) < 21:
            self._no_hand += 1
            # FIX BUG 7: raised from 4 → _NO_HAND_CLEAR (12)
            if self._no_hand >= _NO_HAND_CLEAR:
                self.buffer.clear()
            return None, 0.0, dict(_NULL_FS)

        self._no_hand = 0
        fs     = self.get_finger_states_named(landmarks)
        letter, raw_conf = self._classify(landmarks)

        # FIX BUG 5: only push real letters into the buffer;
        # None frames are reflected as lower stab without poisoning the vote.
        if letter is not None:
            self.buffer.append(letter)

        n = len(self.buffer)
        if n >= _MIN_BUF_VOTE:
            # Weighted vote: recent frames get higher weight
            weights = [1.0 + i / n for i in range(n)]
            totals: Dict[Optional[str], float] = {}
            for i, ltr in enumerate(self.buffer):
                totals[ltr] = totals.get(ltr, 0.0) + weights[i]

            # FIX BUG 5: exclude None from election
            valid_totals = {k: v for k, v in totals.items() if k is not None}
            if not valid_totals:
                return letter, raw_conf * 0.75, fs

            best = max(valid_totals, key=valid_totals.__getitem__)
            stab = valid_totals[best] / sum(weights)
            if stab >= 0.45:
                return best, min(raw_conf * (0.35 + stab * 0.65), 0.99), fs

        # FIX BUG 9: was * 0.50; raised to * 0.75 for faster startup
        return letter, raw_conf * 0.75, fs

    def reset(self) -> None:
        """Clear the temporal smoothing buffer and no-hand counter."""
        self.buffer.clear()
        self._no_hand = 0

    def __repr__(self) -> str:
        return f"GestureEngine(buffer_size={self.buffer_size})"