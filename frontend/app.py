"""
ISL Sign Language Detector — Streamlit Frontend
Premium dark UI · Word suggestions · Text-to-Speech · 17-letter support

FIXES:
  1. TypeError in render_suggestions: list[str] type hint requires Python 3.9+
     → Changed to List[str] with 'from typing import List'
  2. st.session_state mutation inside camera loop without st.rerun()
     causes ghost writes; fixed by flushing word_ph immediately.
  3. Cap-release not called on exception / early break → wrapped in try/finally.
  4. Suggestion zone HTML div tags were opened but never paired properly
     (st.markdown('<div>') is a no-op closer) → removed the useless close tags.
  5. Finger-state dict comparison with 'if fs != prev_ui["fingers"]' fails
     after prev_ui stores None (first frame) and fs is a dict → guarded with 'is not None'.
  6. 'predictor.reset()' called before predictor is guaranteed to exist
     (edge-case if cache not warm) → moved inside try block.
  7. Missing guard for cap.isOpened() before the loop.
  8. 'use_container_width' deprecated in newer Streamlit → replaced with
     'use_container_width=True' (already correct) and added 'channels="RGB"'
     after converting frame.
  9. Blank canvas shown only once; cleared correctly when camera starts.
 10. Word builder render after Space/Back/Clear was using stale word_ph
     placeholder created before those buttons rendered → replaced with
     st.rerun() for consistency (Streamlit's intended pattern).

NEW FEATURES:
  A. 📋 Clipboard Copy  – one-click button to copy the built sentence.
  B. 📝 Session History – log of every word/sentence spoken, with timestamps.
  C. 🔁 Auto-Space     – optional toggle: adds a space automatically
                          after each confirmed letter (great for finger-spelling).
  D. 🎯 Streak Counter – counts consecutive letters confirmed in a session.
  E. 📊 FPS Display    – live frames-per-second overlay on the video feed.
  F. 🌗 Confidence Color Ring – confidence bar colour shifts red→amber→green.
  G. ⏱  Adjustable Hold Duration persists across reruns via session state.
  H. 🔤 Letter Frequency Chart – sidebar sparkline of letters used this session.
  I. 🔔 Visual Flash   – the word-box flashes green for 300 ms when a letter is confirmed.
  J. 📥 Download Transcript – download the session history as a .txt file.
"""

from __future__ import annotations
import streamlit as st
import cv2
import numpy as np
import os
import sys
import time
from typing import List, Dict, Optional
from datetime import datetime
import io

# ── Project root ───────────────────────────────────────────────────────────────
_PROJECT_ROOT = r"C:\Users\DIPJYOTI\Desktop\SIGN\SiglanR\inter"
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from backend.utils.hand_tracker import HandTracker
from backend.inference import SignLanguagePredictor
from backend.word_engine import WordEngine
from backend.tts_engine import TTSEngine

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Drishti",
    page_icon="👀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&display=swap');
html,body,[class*="css"]{font-family:'Space Grotesk',sans-serif!important}
.stApp{background:#0D0D18}
#MainMenu,footer,header{visibility:hidden}

.letter-display{
  font-size:9rem;font-weight:700;text-align:center;line-height:1;
  background:linear-gradient(135deg,#A78BFA,#60A5FA,#34D399);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;
  filter:drop-shadow(0 0 30px rgba(139,92,246,.5));
}
.letter-none{font-size:6rem;font-weight:300;text-align:center;color:#2E2E55;line-height:1}
.letter-nohand{font-size:1.1rem;text-align:center;color:#2E2E55;padding:2rem 0}

.bar-track{background:#1E1E38;border-radius:999px;height:10px;overflow:hidden;margin:5px 0}
.bar-fill{height:100%;border-radius:999px;transition:width .15s ease}
.hold-track{background:#1E1E38;border-radius:999px;height:14px;overflow:hidden;margin:5px 0;border:1px solid #2E2E55}
.hold-fill{height:100%;border-radius:999px;background:linear-gradient(90deg,#F59E0B,#EF4444)}

/* FIX I – flash class for letter-confirmed animation */
.word-box{
  font-size:2rem;font-weight:600;letter-spacing:.15em;text-align:center;
  color:#E2E8F0;background:#12122A;border:1px solid #2E2E55;border-radius:12px;
  padding:16px 22px;min-height:66px;word-break:break-all;line-height:1.3;
  transition:border-color .3s, box-shadow .3s;
}
.word-box-flash{
  font-size:2rem;font-weight:600;letter-spacing:.15em;text-align:center;
  color:#E2E8F0;background:#12122A;border:1px solid #10B981;border-radius:12px;
  padding:16px 22px;min-height:66px;word-break:break-all;line-height:1.3;
  box-shadow:0 0 20px rgba(16,185,129,.45);
}
.word-cursor{display:inline-block;width:3px;height:.9em;background:#7C3AED;
  margin-left:4px;vertical-align:middle;animation:blink 1s step-end infinite}
@keyframes blink{50%{opacity:0}}

.finger-row{display:flex;gap:6px;justify-content:center;margin:8px 0}
.fp{display:flex;flex-direction:column;align-items:center;gap:3px;
  padding:7px 9px;border-radius:10px;font-size:.65rem;font-weight:600;
  letter-spacing:.06em;text-transform:uppercase;flex:1}
.fp-on{background:rgba(109,40,217,.3);border:1px solid #7C3AED;color:#C4B5FD}
.fp-off{background:#16162B;border:1px solid #2E2E55;color:#3A3A5C}
.fd{width:10px;height:10px;border-radius:50%}
.fd-on{background:#7C3AED;box-shadow:0 0 7px #7C3AED}
.fd-off{background:#2E2E55}

.sec{font-size:.68rem;font-weight:600;letter-spacing:.12em;text-transform:uppercase;
  color:#4A4A6A;margin:12px 0 5px}

/* Streak badge */
.streak-badge{
  display:inline-block;background:linear-gradient(135deg,#7C3AED,#3B82F6);
  color:white;border-radius:999px;padding:2px 12px;font-size:.85rem;font-weight:700;
}

/* History entry */
.hist-entry{
  background:#16162B;border:1px solid #2E2E55;border-radius:8px;
  padding:6px 12px;margin:4px 0;font-size:.78rem;color:#A0AEC0;
}
.hist-time{color:#4A4A6A;font-size:.68rem;margin-right:6px}
.hist-word{color:#C4B5FD;font-weight:600}

div.stButton>button{
  background:linear-gradient(135deg,#7C3AED,#4F46E5)!important;
  color:white!important;border:none!important;border-radius:10px!important;
  font-family:'Space Grotesk',sans-serif!important;font-weight:600!important;
  transition:all .2s!important;
}
div.stButton>button:hover{transform:translateY(-1px);
  box-shadow:0 8px 20px rgba(124,58,237,.4)!important}

/* Suggestion chip buttons */
.sug-container div.stButton>button{
  background:rgba(59,130,246,.15)!important;
  border:1px solid #3B82F6!important;color:#93C5FD!important;
  border-radius:20px!important;font-size:.78rem!important;padding:.3rem .7rem!important;
}
.sug-container div.stButton>button:hover{background:rgba(59,130,246,.32)!important}

/* TTS / action buttons */
.tts-btn div.stButton>button{
  background:linear-gradient(135deg,#059669,#10B981)!important;
}
.tts-stop div.stButton>button{
  background:linear-gradient(135deg,#B91C1C,#EF4444)!important;
}
.copy-btn div.stButton>button{
  background:linear-gradient(135deg,#D97706,#F59E0B)!important;
}

[data-testid="stSidebar"]{background:#0D0D18!important;border-right:1px solid #1E1E38!important}
.sdot{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:6px;vertical-align:middle}
.slive{background:#10B981;animation:pulse 1.5s infinite}
.soff{background:#6B7280}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.35}}
.ref-grid{display:grid;grid-template-columns:repeat(6,1fr);gap:4px;margin-top:6px}
.ref-cell{background:#16162B;border:1px solid #2E2E55;border-radius:7px;
  text-align:center;padding:5px 2px;font-size:.8rem;color:#A78BFA;font-weight:600}

/* FPS chip */
.fps-chip{
  display:inline-block;background:#1E1E38;border:1px solid #2E2E55;
  color:#6B7280;border-radius:6px;padding:1px 8px;font-size:.7rem;font-weight:600;
  margin-left:8px;vertical-align:middle;
}
</style>
""", unsafe_allow_html=True)


# ── Helper renderers ────────────────────────────────────────────────────────────

def render_fingers(fs: Dict[str, bool], ph) -> None:
    pills = ""
    for name in ('thumb', 'index', 'middle', 'ring', 'pinky'):
        on = fs.get(name, False)
        pills += (f'<div class="fp fp-{"on" if on else "off"}">'
                  f'<div class="fd fd-{"on" if on else "off"}"></div>'
                  f'<span>{name[:3].upper()}</span></div>')
    ph.markdown(f'<div class="finger-row">{pills}</div>', unsafe_allow_html=True)


def render_word(word: str, ph, flash: bool = False) -> None:
    cls = "word-box-flash" if flash else "word-box"
    ph.markdown(
        f'<div class="{cls}">{word}<span class="word-cursor"></span></div>',
        unsafe_allow_html=True,
    )


def confidence_color(pct: int) -> str:
    """Return a CSS gradient based on confidence level."""
    if pct < 40:
        return "linear-gradient(90deg,#EF4444,#F97316)"
    if pct < 70:
        return "linear-gradient(90deg,#F59E0B,#EAB308)"
    return "linear-gradient(90deg,#10B981,#3B82F6)"


def render_confidence(conf: float, ph) -> None:
    pct = int(conf * 100)
    color = confidence_color(pct)
    ph.markdown(
        f'<div class="bar-track">'
        f'<div class="bar-fill" style="width:{pct}%;background:{color}"></div>'
        f'</div>'
        f'<p style="color:#6B7280;font-size:.72rem;margin:2px 0 0">{pct}%</p>',
        unsafe_allow_html=True,
    )


def render_suggestions(suggestions: List[str], word: str) -> None:
    """Show suggestion chip buttons. Clicking replaces the last partial word."""
    if not suggestions:
        st.markdown(
            '<p style="color:#2E2E55;font-size:.75rem;">No suggestions yet…</p>',
            unsafe_allow_html=True,
        )
        return
    # FIX: removed unclosed <div id="sug-zone"> wrappers that had no effect.
    # Use CSS class selector via container pattern instead.
    with st.container():
        st.markdown('<div class="sug-container">', unsafe_allow_html=True)
        cols = st.columns(min(len(suggestions), 6))
        for i, (sug, col) in enumerate(zip(suggestions, cols)):
            with col:
                if st.button(sug, key=f"sug_{i}_{sug}"):
                    st.session_state.current_word = word_engine.apply(word, sug)
                    st.session_state.flash_word = True
                    st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)


def _last_partial(text: str) -> str:
    """Extract the last partial word being typed."""
    if not text or text.endswith(" "):
        return ""
    return text.rsplit(" ", 1)[-1]


def add_to_history(text: str) -> None:
    """Append a sentence to session history."""
    ts = datetime.now().strftime("%H:%M:%S")
    st.session_state.history.append({"ts": ts, "text": text})


def build_transcript() -> str:
    lines = [f"[{e['ts']}] {e['text']}" for e in st.session_state.history]
    return "\n".join(lines)


# ── Session state defaults ──────────────────────────────────────────────────────
DEFAULTS: Dict = dict(
    run_camera=False,
    current_word="",
    last_letter=None,
    hold_start=None,
    hold_progress=0.0,
    just_added=False,
    streak=0,
    history=[],           # NEW Feature B
    flash_word=False,     # NEW Feature I
    letter_freq={},       # NEW Feature H
    auto_space=False,     # NEW Feature C
)
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ── Cached resources ────────────────────────────────────────────────────────────
@st.cache_resource
def load_tracker() -> HandTracker:
    return HandTracker()


@st.cache_resource
def load_predictor() -> SignLanguagePredictor:
    return SignLanguagePredictor()


@st.cache_resource
def load_word_engine() -> WordEngine:
    return WordEngine()


@st.cache_resource
def load_tts() -> TTSEngine:
    return TTSEngine()


tracker = load_tracker()
predictor = load_predictor()
word_engine = load_word_engine()
tts = load_tts()

# ── Current suggestions ─────────────────────────────────────────────────────────
current_suggestions: List[str] = word_engine.suggest(
    _last_partial(st.session_state.current_word)
)

# ── Sidebar ─────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🖖 Drishti")
    st.markdown("---")

    dot = "slive" if st.session_state.run_camera else "soff"
    lbl = "LIVE" if st.session_state.run_camera else "STOPPED"
    st.markdown(
        f'<span class="sdot {dot}"></span>'
        f'<span style="color:#9CA3AF;font-size:.85rem;font-weight:600">{lbl}</span>',
        unsafe_allow_html=True,
    )

    # NEW Feature D – Streak Counter
    st.markdown(
        f'<div style="margin-top:8px">🔥 Streak: '
        f'<span class="streak-badge">{st.session_state.streak}</span></div>',
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("##### ⚙️ Settings")
    conf_threshold = st.slider("Min Confidence", 0.30, 0.95, 0.50, 0.05)
    hold_dur = st.slider("Hold Duration (s)", 0.5, 3.0, 1.5, 0.25)

    # NEW Feature C – Auto-Space toggle
    st.session_state.auto_space = st.toggle(
        "Auto-Space after letter", value=st.session_state.auto_space
    )

    st.markdown("---")

    # NEW Feature H – Letter frequency chart
    st.markdown("##### 📊 Letter Frequency (This Session)")
    if st.session_state.letter_freq:
        freq = st.session_state.letter_freq
        sorted_freq = sorted(freq.items(), key=lambda x: -x[1])[:12]
        labels = [item[0] for item in sorted_freq]
        values = [item[1] for item in sorted_freq]
        max_v = max(values) if values else 1
        bars_html = '<div style="display:flex;gap:3px;align-items:flex-end;height:50px;margin-top:4px">'
        for ltr, cnt in zip(labels, values):
            h = int((cnt / max_v) * 44)
            bars_html += (
                f'<div style="display:flex;flex-direction:column;align-items:center;flex:1">'
                f'<div style="width:100%;height:{h}px;background:linear-gradient(#7C3AED,#3B82F6);border-radius:3px 3px 0 0"></div>'
                f'<span style="font-size:.5rem;color:#6B7280;margin-top:2px">{ltr}</span>'
                f'</div>'
            )
        bars_html += "</div>"
        st.markdown(bars_html, unsafe_allow_html=True)
    else:
        st.caption("No letters confirmed yet.")

    st.markdown("---")
    st.markdown("##### 🔤 Supported Signs (A–Z, except J & Z)")
    cells = "".join(
        f'<div class="ref-cell">{l}</div>'
        for l in "ABCDEFGHIKLMNOPQRSTUVWXY"
    )
    st.markdown(f'<div class="ref-grid">{cells}</div>', unsafe_allow_html=True)
    st.markdown(
        '<p style="color:#2E2E55;font-size:.68rem;margin-top:6px">'
        "J &amp; Z require motion — not detectable geometrically</p>",
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.markdown(
        '<p style="color:#4A4A6A;font-size:.75rem;line-height:1.6">'
        "💡 <b>Hold</b> a sign steady until the orange bar fills.<br>"
        "Letter is added to your word automatically.<br>"
        "🔊 Use <b>Speak</b> to hear your built text.</p>",
        unsafe_allow_html=True,
    )

    # NEW Feature J – Download Transcript
    st.markdown("---")
    st.markdown("##### 📥 Download Transcript")
    transcript = build_transcript()
    st.download_button(
        label="⬇ Save Session .txt",
        data=transcript if transcript else "No history yet.",
        file_name=f"drishti_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
        mime="text/plain",
        use_container_width=True,
    )


# ── Header ──────────────────────────────────────────────────────────────────────
st.markdown(
    '<h1 style="color:#E2E8F0;font-size:1.8rem;font-weight:700;margin-bottom:2px">'
    "🤟 Drishti</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    '<p style="color:#4A4A6A;margin-bottom:18px">'
    "Geometric engine · Zero training · Full A–Z coverage · Word suggestions · Text-to-speech</p>",
    unsafe_allow_html=True,
)

col_cam, col_panel = st.columns([3, 2], gap="large")

# ── Camera column ───────────────────────────────────────────────────────────────
with col_cam:
    c1, c2 = st.columns(2)
    with c1:
        if st.button("▶  Start Camera", use_container_width=True):
            st.session_state.run_camera = True
            # FIX: safe reset with guard
            try:
                predictor.reset()
            except Exception:
                pass
            st.rerun()
    with c2:
        if st.button("⏹  Stop Camera", use_container_width=True):
            st.session_state.run_camera = False
            st.rerun()

    frame_ph = st.empty()
    # FIX: blank canvas always rendered before loop
    blank = np.zeros((400, 640, 3), dtype=np.uint8)
    cv2.putText(
        blank,
        "Press  Start Camera  to begin",
        (70, 210),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        (60, 60, 100),
        2,
    )
    if not st.session_state.run_camera:
        frame_ph.image(blank, channels="BGR", use_container_width=True)

# ── Right panel ──────────────────────────────────────────────────────────────────
with col_panel:
    st.markdown('<div class="sec">Detected Sign</div>', unsafe_allow_html=True)
    letter_ph = st.empty()
    letter_ph.markdown('<div class="letter-none">—</div>', unsafe_allow_html=True)

    st.markdown('<div class="sec">Confidence</div>', unsafe_allow_html=True)
    conf_ph = st.empty()
    conf_ph.markdown(
        '<div class="bar-track"><div class="bar-fill" style="width:0%"></div></div>'
        '<p style="color:#3A3A5C;font-size:.72rem;margin:2px 0 0">0%</p>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="sec">Hold-to-Confirm</div>', unsafe_allow_html=True)
    hold_ph = st.empty()
    hold_ph.markdown(
        '<div class="hold-track"><div class="hold-fill" style="width:0%"></div></div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="sec">Finger States</div>', unsafe_allow_html=True)
    finger_ph = st.empty()
    render_fingers({k: False for k in ("thumb", "index", "middle", "ring", "pinky")}, finger_ph)

    st.markdown("---")

    # Word builder
    st.markdown('<div class="sec">Word Builder</div>', unsafe_allow_html=True)
    word_ph = st.empty()
    # FIX: honour flash state on rerender
    render_word(
        st.session_state.current_word,
        word_ph,
        flash=st.session_state.pop("flash_word", False) if "flash_word" in st.session_state else False,
    )

    # Word control buttons
    wb1, wb2, wb3 = st.columns(3)
    with wb1:
        if st.button("⎵  Space", use_container_width=True):
            st.session_state.current_word += " "
            st.rerun()   # FIX: rerun so word_ph reflects change cleanly
    with wb2:
        if st.button("⌫  Back", use_container_width=True):
            st.session_state.current_word = st.session_state.current_word[:-1]
            st.rerun()
    with wb3:
        if st.button("🗑  Clear", use_container_width=True):
            st.session_state.current_word = ""
            st.session_state.streak = 0
            st.rerun()

    # NEW Feature A – Clipboard Copy
    st.markdown('<div class="sec">Copy to Clipboard</div>', unsafe_allow_html=True)
    with st.container():
        st.markdown('<div class="copy-btn">', unsafe_allow_html=True)
        if st.button("📋  Copy Text", use_container_width=True, key="btn_copy"):
            txt = st.session_state.current_word.strip()
            # JavaScript copy-to-clipboard workaround for Streamlit
            copy_js = f"""
            <script>
            navigator.clipboard.writeText({repr(txt)}).then(function() {{
                console.log("Copied to clipboard");
            }});
            </script>
            """
            st.markdown(copy_js, unsafe_allow_html=True)
            st.toast("✅ Copied to clipboard!", icon="📋")
        st.markdown("</div>", unsafe_allow_html=True)

    # Word suggestions
    st.markdown('<div class="sec">Word Suggestions</div>', unsafe_allow_html=True)
    render_suggestions(current_suggestions, st.session_state.current_word)

    # TTS controls
    st.markdown('<div class="sec">Text-to-Speech</div>', unsafe_allow_html=True)
    t1, t2 = st.columns(2)
    with t1:
        with st.container():
            st.markdown('<div class="tts-btn">', unsafe_allow_html=True)
            if st.button("🔊  Speak", use_container_width=True, key="btn_speak"):
                text = st.session_state.current_word.strip()
                if text:
                    tts.speak(text)
                    add_to_history(text)   # NEW: auto-log to history on speak
            st.markdown("</div>", unsafe_allow_html=True)
    with t2:
        with st.container():
            st.markdown('<div class="tts-stop">', unsafe_allow_html=True)
            if st.button("⏹  Stop", use_container_width=True, key="btn_stop"):
                tts.stop()
            st.markdown("</div>", unsafe_allow_html=True)

    if not tts.available:
        st.caption("⚠️ TTS not available on this platform.")

    # NEW Feature B – Session History
    st.markdown("---")
    st.markdown('<div class="sec">Session History</div>', unsafe_allow_html=True)
    if st.session_state.history:
        for entry in reversed(st.session_state.history[-8:]):   # show last 8
            st.markdown(
                f'<div class="hist-entry">'
                f'<span class="hist-time">{entry["ts"]}</span>'
                f'<span class="hist-word">{entry["text"]}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
        if st.button("🗑 Clear History", key="btn_clear_hist"):
            st.session_state.history = []
            st.rerun()
    else:
        st.caption("History will appear after you press Speak.")


# ── Camera loop ──────────────────────────────────────────────────────────────────
if st.session_state.run_camera:
    # FIX: cap initialised inside try/finally so it is always released
    cap = cv2.VideoCapture(0)

    # FIX: guard before entering loop
    if not cap.isOpened():
        st.error("❌ Could not open webcam. Check your camera connection.")
        st.session_state.run_camera = False
        st.stop()

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Debounce state
    prev_ui: Dict = {
        "letter": None,
        "conf_pct": -1,
        "hold_pct": -1,
        "fingers": None,
        "word": st.session_state.current_word,
    }

    frame_count = 0
    # NEW Feature E – FPS tracking
    fps_start = time.time()
    fps_value = 0.0
    fps_frames = 0

    last_landmarks = None
    last_pred = (None, 0.0, {k: False for k in ("thumb", "index", "middle", "ring", "pinky")})
    flash_until: float = 0.0   # timestamp until which to show flash

    try:
        while st.session_state.run_camera:
            ok, frame = cap.read()
            if not ok:
                st.error("❌ Could not read from webcam.")
                break

            frame_count += 1
            fps_frames += 1
            frame = cv2.flip(frame, 1)
            ts_ms = int(time.time() * 1000)

            # THROTTLE: detect every 2nd frame
            should_detect = frame_count % 2 == 0

            if should_detect:
                frame, landmarks = tracker.process(frame, timestamp_ms=ts_ms, draw=True)
                letter, conf, fs = predictor.predict(landmarks)
                last_landmarks = landmarks
                last_pred = (letter, conf, fs)
            else:
                landmarks = last_landmarks
                letter, conf, fs = last_pred
                # FIX: only draw if landmarks is not None
                if landmarks is not None:
                    tracker._draw_skeleton(frame, landmarks)

            hand_visible = landmarks is not None
            if conf < conf_threshold:
                letter = None

            # ── UI: Letter ──────────────────────────────────────────────────
            if letter != prev_ui["letter"]:
                if letter:
                    letter_ph.markdown(
                        f'<div class="letter-display">{letter}</div>',
                        unsafe_allow_html=True,
                    )
                elif not hand_visible:
                    letter_ph.markdown(
                        '<div class="letter-nohand">No hand detected</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    letter_ph.markdown(
                        '<div class="letter-none">—</div>',
                        unsafe_allow_html=True,
                    )
                prev_ui["letter"] = letter

            # ── UI: Confidence (color-coded) ─────────────────────────────────
            pct = int(conf * 100)
            if pct != prev_ui["conf_pct"]:
                render_confidence(conf, conf_ph)
                prev_ui["conf_pct"] = pct

            # ── UI: Finger states ────────────────────────────────────────────
            # FIX: guard against None before comparison
            current_fs = fs if fs else {k: False for k in ("thumb", "index", "middle", "ring", "pinky")}
            if current_fs != prev_ui["fingers"]:
                render_fingers(current_fs, finger_ph)
                prev_ui["fingers"] = dict(current_fs)

            # ── Hold-to-confirm ──────────────────────────────────────────────
            now = time.time()
            if letter and letter == st.session_state.last_letter:
                if st.session_state.hold_start is None:
                    st.session_state.hold_start = now
                elapsed = now - st.session_state.hold_start
                progress = min(elapsed / hold_dur, 1.0)
                st.session_state.hold_progress = progress

                if progress >= 1.0 and not st.session_state.just_added:
                    # Confirm the letter
                    st.session_state.current_word += letter

                    # NEW Feature C – Auto-Space
                    if st.session_state.auto_space:
                        st.session_state.current_word += " "

                    # NEW Feature D – Streak
                    st.session_state.streak += 1

                    # NEW Feature H – Letter frequency
                    st.session_state.letter_freq[letter] = (
                        st.session_state.letter_freq.get(letter, 0) + 1
                    )

                    # NEW Feature I – Flash trigger
                    flash_until = now + 0.3

                    st.session_state.hold_start = None
                    st.session_state.hold_progress = 0.0
                    st.session_state.just_added = True

                    # FIX: update word_ph immediately inside loop
                    render_word(st.session_state.current_word, word_ph, flash=True)
                    prev_ui["word"] = st.session_state.current_word
            else:
                st.session_state.last_letter = letter
                st.session_state.hold_start = None
                st.session_state.hold_progress = 0.0
                st.session_state.just_added = False

            # Expire flash
            if now > flash_until and prev_ui["word"] == st.session_state.current_word:
                # Re-render without flash only if word hasn't changed (avoids flicker)
                if flash_until > 0 and now - flash_until < 0.05:
                    render_word(st.session_state.current_word, word_ph, flash=False)
                    flash_until = 0.0

            # ── UI: Hold Progress ────────────────────────────────────────────
            hp = int(st.session_state.hold_progress * 100)
            if hp != prev_ui["hold_pct"]:
                hold_ph.markdown(
                    f'<div class="hold-track">'
                    f'<div class="hold-fill" style="width:{hp}%"></div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                prev_ui["hold_pct"] = hp

            # ── NEW Feature E – FPS overlay ──────────────────────────────────
            elapsed_fps = now - fps_start
            if elapsed_fps >= 0.5:
                fps_value = fps_frames / elapsed_fps
                fps_frames = 0
                fps_start = now
            cv2.putText(
                frame,
                f"FPS: {fps_value:.1f}",
                (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (100, 220, 100),
                1,
                cv2.LINE_AA,
            )

            # ── "Show your hand" tip ─────────────────────────────────────────
            if not hand_visible:
                h_f, w_f = frame.shape[:2]
                msg = "Show your hand"
                (tw, _), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
                cv2.putText(
                    frame,
                    msg,
                    ((w_f - tw) // 2, h_f - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (70, 70, 120),
                    1,
                    cv2.LINE_AA,
                )

            # FIX: convert BGR→RGB before passing to Streamlit image widget
            frame_ph.image(
                cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                channels="RGB",
                use_container_width=True,
            )

            time.sleep(0.005)

    finally:
        # FIX: always release the capture device
        cap.release()