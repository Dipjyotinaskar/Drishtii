# 🤟 Drishti — ISL Sign Language Detector

A real-time **Indian Sign Language (ISL) detection system** that runs entirely in your browser via a Streamlit web app. No deep learning training required — the system uses a **geometric rule engine** on top of MediaPipe hand landmarks to classify 24 single-handed ISL letters.

> **Note on ISL scope:** Full ISL fingerspelling includes both one-handed and two-handed signs. This version detects the **single-handed** component of the ISL alphabet (24 static postures). Two-handed sign support is a planned future enhancement requiring dual-hand landmark tracking.

---

## ✨ Features

- 🎥 **Live webcam** hand tracking at ~30 fps with live FPS overlay
- 🔤 **24 ISL letters** (single-handed static postures; J & Z excluded — they require motion)
- 🧠 **Zero training** — pure geometric analysis of hand joint angles and finger directions
- 📖 **Word suggestions** — real-time prefix-based autocomplete from the system dictionary
- 🔊 **Text-to-speech** — cross-platform TTS (Windows SAPI5, macOS NSS, Linux espeak)
- ✍️ **Word builder** — hold a sign steady to auto-append the letter; Space, Backspace, and Clear controls
- 🎨 **Premium dark UI** — glassmorphic design with live finger state display and confidence bars
- 🔥 **Streak counter** — tracks consecutive letters confirmed in a session
- 📊 **Letter frequency chart** — sidebar sparkline of letters used this session
- 📋 **Clipboard copy** — one-click button to copy the built sentence
- 📝 **Session history** — log of every sentence spoken, with timestamps
- 🔁 **Auto-Space toggle** — optionally inserts a space after each confirmed letter
- 📥 **Transcript download** — save the full session history as a `.txt` file

---

## 📁 Project Structure

```
signLanguageDetector/
├── frontend/
│   └── app.py                    # Streamlit UI (Drishti)
├── backend/
│   ├── __init__.py               # (empty — marks backend as a package)
│   ├── gesture_engine.py         # Core geometric rule engine (24 ISL letters)
│   ├── inference.py              # Wrapper: landmarks → letter + confidence
│   ├── word_engine.py            # Prefix-based word suggestions
│   ├── tts_engine.py             # Cross-platform text-to-speech
│   ├── speak_worker.py           # Subprocess worker for pyttsx3
│   ├── models/
│   │   └── hand_landmarker.task  # MediaPipe hand landmark model (download separately)
│   └── utils/
│       ├── __init__.py           # (empty — marks utils as a package)
│       └── hand_tracker.py       # MediaPipe hand tracking + skeleton overlay
├── requirements.txt
└── README.md
```

---

## 🖥️ System Requirements

| Requirement | Minimum |
|---|---|
| Python | 3.9 – 3.11 |
| Webcam | Any USB or built-in camera |
| OS | Windows 10+, macOS 11+, Ubuntu 20.04+ |
| RAM | 4 GB (8 GB recommended) |

---

## 🚀 Setup & Installation

### Step 1 — Clone the repository

```bash
git clone https://github.com/Adi15Jain/signLanguageDetector.git
cd signLanguageDetector
```

### Step 2 — Create a virtual environment

**macOS / Linux**
```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows**
```bash
python -m venv venv
venv\Scripts\activate
```

> You should see `(venv)` appear at the start of your terminal prompt.

### Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

This installs: `streamlit`, `opencv-python`, `mediapipe`, `numpy`, `pyttsx3`, and `protobuf` at tested, compatible versions.

> ⚠️ **protobuf version matters.** MediaPipe is incompatible with `protobuf >= 5.0.0`. If you see `TypeError: Descriptors cannot be created directly`, run:
> ```bash
> pip install "protobuf>=3.20.0,<5.0.0"
> ```

### Step 4 — Download the MediaPipe hand landmark model

The model file is **not included in the repository** (too large for GitHub).

1. Download from the official MediaPipe page:  
   👉 https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task

2. Place the downloaded file here:
   ```
   signLanguageDetector/backend/models/hand_landmarker.task
   ```

### Step 5 — Create the package `__init__.py` files

If they don't already exist, create two empty files:

```bash
# macOS / Linux
touch backend/__init__.py
touch backend/utils/__init__.py
```

```powershell
# Windows (PowerShell)
New-Item backend/__init__.py -ItemType File
New-Item backend/utils/__init__.py -ItemType File
```

### Step 6 — Run Drishti

Run this from the **`signLanguageDetector/` root directory**:

```bash
streamlit run frontend/app.py
```

The app will open automatically in your browser at **http://localhost:8501**.

> ⚠️ **Always run from the project root** (`signLanguageDetector/`), not from inside `frontend/`. The backend imports are resolved relative to the project root.

---

## 🎮 How to Use

1. **Start Camera** — Click the `▶ Start Camera` button.
2. **Show a sign** — Hold your hand in front of the webcam with a clear background.
3. **Hold to confirm** — Keep the same sign steady; the orange bar will fill up, then the letter is added to your word.
4. **Word suggestions** — Clickable suggestions appear below the word builder based on what you've typed.
5. **Speak** — Press `🔊 Speak` to hear the built text read aloud.
6. **Word controls** — Use `⎵ Space`, `⌫ Back`, and `🗑 Clear` to edit.
7. **Auto-Space** — Enable the sidebar toggle to automatically insert a space after each confirmed letter.

### Tips for best accuracy

- Use a **plain, contrasting background**.
- Ensure **good, even lighting** — avoid strong backlight.
- Hold your hand at a natural distance — **40–70 cm** from the camera.
- J and Z are excluded because they require drawing a letter in the air (motion-based signs).

---

## 🔤 Supported Signs

| A | B | C | D | E | F |
|---|---|---|---|---|---|
| G | H | I | K | L | M |
| N | O | P | Q | R | S |
| T | U | V | W | X | Y |

> J and Z require motion detection and are not supported in this static-pose version.

---

## 🧠 How It Works

Instead of training a neural network, Drishti uses **geometric rules applied to 21 3D hand landmarks** detected by MediaPipe:

1. **Tip-to-wrist ratio** — Determines if each finger is extended or curled (robust to hand tilt).
2. **PIP joint angles** — Measures how sharply each finger is bent at its middle joint.
3. **Direction vectors** — Detects whether the index finger is pointing up, sideways, or down.
4. **Spread distances** — Palm-normalised distances between fingertips to detect V vs U vs R.
5. **Thumb position** — Detects A vs S vs T based on where the thumb tip sits relative to the fist.
6. **Weighted temporal buffer** — A 15-frame sliding window with linear weights smooths flickery predictions.

---

## 🌐 Cross-platform TTS

Text-to-speech uses `pyttsx3`, which automatically selects the right engine:
- **Windows** → Microsoft SAPI5 voice engine
- **macOS** → Apple NSSpeechSynthesizer
- **Linux** → espeak (install with `sudo apt install espeak espeak-ng`)

To test TTS independently:
```bash
python backend/speak_worker.py "Namaste"
```

You can also customise TTS behaviour with environment variables:
```bash
TTS_RATE=120 TTS_VOLUME=0.9 python backend/speak_worker.py "Hello"
# TTS_VOICE=Zira  # Windows: select a specific SAPI5 voice by name
```

---

## 🐛 Troubleshooting

| Problem | Solution |
|---|---|
| `ModuleNotFoundError: pyttsx3` | Run `pip install pyttsx3` with the venv active |
| `ModuleNotFoundError` (other) | Run `pip install -r requirements.txt` with the venv active |
| `No module named 'backend'` | Ensure you run `streamlit run frontend/app.py` from the **project root** (`signLanguageDetector/`), not from inside `frontend/`. Also ensure `backend/__init__.py` and `backend/utils/__init__.py` exist (empty files). |
| Camera not opening | Check webcam permissions in System Settings / Device Manager |
| `hand_landmarker.task not found` | Download model from Step 4 and place it at `backend/models/hand_landmarker.task` |
| TTS not working on Linux | Install espeak: `sudo apt install espeak espeak-ng` |
| `TypeError: Descriptors cannot be created directly` | Run `pip install "protobuf>=3.20.0,<5.0.0"` |
| Low accuracy | Improve lighting; use a plain background; hold hand ~50 cm from camera |
| Prediction flickering | Increase the Hold Duration slider in the sidebar |
| Signs detected after camera restart | Known fix applied in `hand_tracker.py` — call `tracker.reset()` which now also clears stale landmark data |
| TTS speaks but wrong voice on Windows | Set environment variable `TTS_VOICE=<name>` (e.g. `TTS_VOICE=Zira`) before running |

---

## 📦 `requirements.txt`

```
streamlit>=1.32.0
opencv-python>=4.8.0
mediapipe>=0.10.14
numpy>=1.24.0
pyttsx3>=2.90
protobuf>=3.20.0,<5.0.0
```

---

## 📄 License

MIT License — free to use, modify, and distribute.

---

## 🙏 Acknowledgements

- [MediaPipe](https://developers.google.com/mediapipe) by Google — hand landmark detection
- [Streamlit](https://streamlit.io) — web app framework
- [pyttsx3](https://pyttsx3.readthedocs.io) — cross-platform TTS