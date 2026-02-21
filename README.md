# 🤖 J.A.R.V.I.S. — MacBook Air M1 Edition

> A fully local, voice-activated AI assistant running 100% on Apple Silicon.
> No cloud. No paid APIs. No internet required at runtime.
> Built on a $999 laptop with 8GB RAM.



---

## 🎬 Demo

> Say **"Hey Jarvis"** → Ask anything → Get a spoken response — all running locally.

```
┌─────────────────────────────────────────────────┐
│                                                 │
│    ╔══════════════════════════════════════╗      │
│    ║                                      ║      │
│    ║    ░█ ░█▀█ ░█▀▄ ░█  ░█ ░█ ░█▀▀     ║      │
│    ║    ░█ ░█▀█ ░█▀▄ ░▀▄▀  ░█ ░▀▀█      ║      │
│    ║    █▄ ░█ █ ░█ █  ░█   ░█ ░▀▀▀       ║      │
│    ║                                      ║      │
│    ║    MacBook Air M1 Edition            ║      │
│    ║    100% Local • Zero Cost            ║      │
│    ║                                      ║      │
│    ╚══════════════════════════════════════╝      │
│                                                 │
│  ✅ All systems online                          │
│  🎙️  Say 'Hey Jarvis' to activate               │
│                                                 │
│  🔊 Wake word detected! Heard: "hey jarvis"     │
│  🎯 WAKE WORD TRIGGERED                         │
│  🎙️  Listening... (speak now)                    │
│  📝 Transcription: "What can you do for me?"    │
│  🧠 Thinking...                                 │
│  🤖 Jarvis: "I can help with questions,         │
│     set reminders, search the web..."           │
│  🔊 Speaking...                                 │
│  🟢 RAM: 73.4% — Cycle 1 complete               │
│                                                 │
└─────────────────────────────────────────────────┘
```

---

## 🎯 What This Is

J.A.R.V.I.S. is a personal AI assistant that runs **entirely on a MacBook Air M1 with 8GB RAM**. Every component — wake word detection, speech recognition, language understanding, and text-to-speech — runs locally with aggressive memory optimization.

**This is not a wrapper around ChatGPT.** Every model runs on-device using Apple's Neural Engine and Metal GPU.

---

## 🏗️ Architecture

```
                         🎤 MacBook Air Microphone
                                  │
                                  ▼
                    ┌──────────────────────────┐
                    │   Continuous InputStream   │
                    │   (zero-gap streaming)     │
                    └────────────┬─────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │   85Hz High-Pass Filter    │  ← Removes fan/AC hum
                    │   (IIR in audio callback)  │     before any processing
                    └────────────┬─────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │   Dual Energy Gate         │  ← avg RMS > 15
                    │   (anti-hallucination)     │     AND peak RMS > 80
                    └────────────┬─────────────┘
                                 │ speech detected
                                 ▼
                    ┌──────────────────────────┐
                    │   Wake Word Detection      │  ← mlx-whisper base (~140MB)
                    │   "Hey Jarvis" / variants  │     2.5s sliding windows
                    └────────────┬─────────────┘
                                 │ triggered
                                 ▼
                    ┌──────────────────────────┐
                    │   Speech Recording         │  ← Same stream, zero gaps
                    │   (VAD + min 2s capture)   │     High-pass pre-filtered
                    └────────────┬─────────────┘
                                 │ audio captured
                                 ▼
                    ┌──────────────────────────┐
                    │   Speech-to-Text           │  ← mlx-whisper small (~240MB)
                    │   (anti-hallucination)     │     Apple Neural Engine
                    └────────────┬─────────────┘
                                 │ text
                                 ▼
                    ┌──────────────────────────┐
                    │   NLU / Brain              │  ← Phi-3 Mini 3.8B via Ollama
                    │   (fault-isolated process) │     Metal GPU, Q4, ctx=2048
                    └────────────┬─────────────┘
                                 │ response
                                 ▼
                    ┌──────────────────────────┐
                    │   Text-to-Speech           │  ← macOS native `say`
                    │   (zero RAM overhead)      │     Voice: Daniel
                    └──────────────────────────┘
```

---

## 💾 Memory Budget

Running on **8GB unified memory** — every megabyte is a conscious decision:

| Component | RAM Usage | Device | Why This Choice |
|-----------|-----------|--------|-----------------|
| Python + deps | ~200MB | CPU | Minimal dependency footprint |
| Wake word (whisper-base) | ~140MB | Neural Engine | Tiny was too inaccurate, base is the sweet spot |
| STT (whisper-small) | ~240MB | Neural Engine | Best accuracy-to-size ratio for transcription |
| Phi-3 Mini (Ollama) | ~2.3GB | Metal GPU | Separate process — fault isolation by design |
| High-pass filter | ~0MB | CPU | Pure math on existing arrays |
| macOS TTS | ~0MB | System | Native `say` command, no model to load |
| **Python process total** | **~580MB** | | |
| **System total (peak)** | **~5.4GB / 8GB** | | **27% headroom** ✅ |

---

## 🛡️ Engineering Decisions

### Why Ollama Stays Separate (Not mlx-lm)
On 8GB, process isolation is a **feature**. If Phi-3 OOMs during a complex query, only the Ollama process dies — the audio stream, wake word listener, and Python app survive and can retry. Loading the LLM in-process (via mlx-lm) would mean one memory spike kills everything. This is the same architecture Apple uses for Siri — separate daemons per subsystem.

### Why Not Async/Multiprocessing
M1 has one Metal GPU shared across all processes. Whisper and Phi-3 can't run in parallel — they'd fight over the same GPU memory. Every pipeline step is either mic-blocked, GPU-blocked, or intentionally blocking (don't listen while speaking). Async adds complexity with zero throughput gain on this hardware.

### Why STT-Based Wake Word (Not openWakeWord)
openWakeWord's embedding model produces **dead inference on M1** — max confidence of 0.000017 across 250 audio chunks. The TFLite runtime has no Apple Silicon wheel for Python 3.11, and the ONNX fallback loads but produces zero-confidence predictions. We replaced it with whisper-base doing 2.5-second sliding window transcription with a dual energy gate — more accurate and proven to work.

### The "Thank You" Hallucination Fix
Whisper hallucinates "Thank you for watching" and "Subscribe" on near-silence because it feeds its own previous output as context (`condition_on_previous_text=True` by default). Our fix: set it to `False`, tighten `compression_ratio_threshold` to 1.8, and raise `no_speech_threshold` to 0.5. Combined with the 85Hz high-pass filter removing fan noise, hallucinations are eliminated.

### Accent-Aware Trigger Phrases
Whisper-base transcribes "Jarvis" as "Jalvis" with Indian English pronunciation. Rather than fighting the model, we include phonetic variants in the trigger list: `jarvis`, `jalvis`, `hey jarvis`, `hey jalvis`, etc.

---

## 🛠️ Tech Stack

| Layer | Technology | RAM Cost |
|-------|-----------|----------|
| Audio Streaming | `sounddevice` InputStream + callback queue | ~0MB |
| DSP Filter | 85Hz IIR high-pass (numpy, no scipy) | ~0MB |
| Wake Word | `mlx-whisper` base model | ~140MB |
| Speech-to-Text | `mlx-whisper` small model | ~240MB |
| NLU / Brain | Phi-3 Mini 3.8B via Ollama (Metal GPU) | ~2.3GB |
| Text-to-Speech | macOS native `say` command | ~0MB |
| Memory Profiling | `psutil` + `Rich` logging | ~5MB |
| Config | YAML | ~0MB |

---

## 🚀 Quick Start

### Prerequisites

- macOS with Apple Silicon (M1/M2/M3)
- Python 3.11 (`brew install python@3.11`)
- [Ollama](https://ollama.com) installed and running
- [Homebrew](https://brew.sh)

### Setup

```bash
# Clone
git clone https://github.com/YOUR_USERNAME/jarvis.git
cd jarvis

# System dependencies
brew install portaudio

# AI brain
ollama pull phi3:mini

# Python environment
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements-phase1.txt

# Launch
python -m src.main
```

### Usage

1. Wait for **"Jarvis is online and ready, sir."**
2. Say **"Hey Jarvis"** at normal conversation volume
3. Wait for **"Yes?"**
4. Ask your question
5. Jarvis responds out loud
6. Repeat

---

## 📁 Project Structure

```
jarvis/
├── config/
│   └── jarvis_config.yaml          # All tunable parameters (thresholds, models, prompts)
├── src/
│   ├── main.py                     # Entry point — voice loop with GC + crash recovery
│   ├── core/
│   │   ├── audio.py                # Streaming mic capture + 85Hz high-pass DSP filter
│   │   ├── wake_word.py            # STT-based wake word with dual energy gate
│   │   ├── stt.py                  # Speech-to-text (mlx-whisper, anti-hallucination)
│   │   ├── nlu.py                  # Language understanding (Ollama + Phi-3 + fallback)
│   │   └── tts.py                  # Text-to-speech (macOS native)
│   ├── utils/
│   │   ├── config.py               # YAML config loader
│   │   └── logger.py               # Rich logging + psutil RAM monitoring
│   ├── memory/                     # Phase 2: ChromaDB persistent memory
│   ├── tools/                      # Phase 3: Mac automation, web search, email
│   ├── vision/                     # Phase 5: Screen OCR, webcam analysis
│   └── ui/                         # Phase 6: Streamlit dashboard
├── tests/
│   ├── diagnose_audio.py           # Audio pipeline diagnostic tool
│   └── fix_wake_word.py            # Wake word nuclear troubleshooter
├── logs/                           # Runtime logs with memory profiling
├── models/                         # Local model weights
├── docs/                           # Architecture notes
└── requirements-phase1.txt         # Phase 1 Python dependencies
```

---

## 📋 Roadmap

- [x] **Phase 1: Voice Core** — Wake word → STT → NLU → TTS
  - [x] Continuous streaming audio (zero-gap InputStream)
  - [x] 85Hz high-pass DSP filter
  - [x] Dual-gate wake word detection (avg + peak RMS)
  - [x] Anti-hallucination whisper parameters
  - [x] Fault-isolated NLU via Ollama
  - [x] Mic disconnect auto-recovery
  - [x] GC optimization for 8GB RAM baseline
- [ ] **Phase 2: Memory & Context** — ChromaDB persistent memory
- [ ] **Phase 3: Tools & Actions** — Mac control, web search, emails, reminders
- [ ] **Phase 4: Code Writing** — Autocoding, execution, error loop
- [ ] **Phase 5: Vision** — Screen OCR and webcam analysis
- [ ] **Phase 6: Dashboard UI** — Streamlit command center

---

## 🔧 Debugging Journey

Building a local AI assistant on 8GB taught me things no tutorial covers:

1. **openWakeWord is broken on M1** — TFLite has no Apple Silicon wheel for Python 3.11. ONNX fallback loads but returns zero confidence. Pivoted to STT-based detection.

2. **`sd.rec()` in a loop drops 40% of audio** — Each call has a ~15ms gap while Python runs logic. Over 3 seconds, "Who are you?" becomes garbled and whisper hears "Thank you." Fixed with persistent `InputStream` + callback queue.

3. **Silence threshold is hardware-specific** — Started at 500, then 100, finally 30. The MacBook Air M1 mic has very low baseline RMS (~2-7). Every deployment needs threshold tuning.

4. **Whisper hallucinates on silence** — Produces "Thank you for watching" from fan noise because `condition_on_previous_text=True` creates feedback loops. Setting it to `False` is mandatory for voice assistants.

5. **Accent handling beats model upgrades** — Instead of downloading a larger model, adding "jalvis" to trigger phrases solved recognition instantly.

6. **Process isolation > in-process performance** — On constrained hardware, keeping the LLM in a separate process (Ollama) is safer than loading everything into one Python process. Crash isolation matters more than 50ms of IPC overhead.

---

## 💡 What Makes This Different

Most "AI assistant" projects are thin wrappers around OpenAI's API. This one:

- **Runs 100% offline** after initial model downloads
- **Costs $0** to run — no API keys, no subscriptions, no cloud
- **Respects privacy** — your voice never leaves your machine
- **Engineered for constraints** — every component chosen to fit in 8GB
- **Production-grade audio** — DSP filtering, streaming capture, anti-hallucination
- **Actually documented** — every design decision explained with the "why"

---

## 🤝 About

Built by **Swapnil Hazra** (@Swapnil-bo) as a portfolio project for AI Product Management.

This project proves that useful AI doesn't need a datacenter — it can run on a $999 laptop with zero ongoing costs.

⭐ **Star this repo** if you think local AI is the future.

---

*Built with ❤️ on Apple Silicon. Part of the 100 Days of Vibe Coding challenge.*
