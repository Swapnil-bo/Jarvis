# 🤖 J.A.R.V.I.S. — MacBook Air M1 Edition

> A fully local, voice-activated AI assistant with real-time dashboard, persistent memory, tool automation, and WhatsApp messaging — running 100% on Apple Silicon.
> No cloud. No paid APIs. No internet required at runtime.
> Built on a $999 laptop with 8GB RAM.

[![Built with](https://img.shields.io/badge/Built_on-Apple_Silicon_M1-black?logo=apple)](https://apple.com)
[![RAM](https://img.shields.io/badge/RAM-8GB_Unified-blue)](https://apple.com)
[![Models](https://img.shields.io/badge/LLM-Phi--3_Mini_3.8B-green)](https://ollama.com)
[![Cost](https://img.shields.io/badge/API_Cost-$0-brightgreen)](https://github.com)
[![Challenge](https://img.shields.io/badge/100_Days_of-Vibe_Coding-orange)](https://github.com)

---

## 🎬 What It Looks Like

```
┌──────────── J.A.R.V.I.S. DASHBOARD ─────────────┐
│                                                   │
│  ┌─────────┐  ┌──────────────┐  ┌─────────────┐  │
│  │  STATUS  │  │ CONVERSATION │  │   SYSTEM    │  │
│  │   ◎◎◎   │  │              │  │ ┌───┐ ┌───┐ │  │
│  │  ████   │  │ YOU: What's  │  │ │78%│ │42%│ │  │
│  │ LISTEN  │  │  the time?   │  │ │BAT│ │RAM│ │  │
│  │         │  │              │  │ └───┘ └───┘ │  │
│  │ ┌──┬──┐ │  │ JARVIS: It's │  ├─────────────┤  │
│  │ │12│ 8│ │  │  3:42 PM,    │  │ TOOL ROUTER │  │
│  │ │EX│FC│ │  │  Sonu.       │  │ ⚡ system    │  │
│  │ └──┴──┘ │  │              │  │   → time    │  │
│  └─────────┘  └──────────────┘  └─────────────┘  │
│                                                   │
│  J.A.R.V.I.S.  ● SPEAKING    ⏱ 00:05:32  🎙 12  │
└───────────────────────────────────────────────────┘
```

**Live glassmorphic dashboard** at `http://127.0.0.1:8765` — real-time status ring, conversation feed, tool router log, system telemetry, all over WebSocket.

---

## 🎯 What This Is

J.A.R.V.I.S. is a personal AI assistant that runs **entirely on a MacBook Air M1 with 8GB RAM**. Every component — wake word detection, speech recognition, language understanding, tool execution, persistent memory, and a real-time dashboard — runs locally with aggressive memory optimization.

**This is not a wrapper around ChatGPT.** Every model runs on-device using Apple's Neural Engine and Metal GPU.

### What It Can Do

| Category | Examples |
|----------|---------|
| 💬 **Conversation** | Chat naturally with context from past conversations |
| 🕐 **System Info** | "What time is it?" • "Battery level?" • "What day is today?" |
| 💻 **Mac Control** | "Open Safari" • "Set volume to 50%" • "Take a screenshot" • "Lock screen" |
| 🌐 **Web Search** | "What's the weather in Mumbai?" • "Bitcoin price?" • "Latest AI news?" |
| ⏰ **Reminders** | "Set a timer for 5 minutes" • "Remind me to call Mom in 10 minutes" |
| 💬 **WhatsApp** | "Send a WhatsApp message to Mom saying I'll be late" |
| 🧠 **Memory** | Remembers your name, interests, past conversations across sessions |
| 📊 **Dashboard** | Live glassmorphic UI with status, telemetry, chat feed, tool router |

---

## 🏗️ Architecture

```
                          🎤 MacBook Air Microphone
                                   │
                                   ▼
                     ┌──────────────────────────┐
                     │   Continuous InputStream   │  Zero-gap streaming
                     │   + 85Hz High-Pass Filter  │  Removes fan/AC hum
                     └────────────┬───────────────┘
                                  │
                                  ▼
                     ┌──────────────────────────┐
                     │   Dual Energy Gate         │  avg RMS > 15 AND peak > 80
                     │   (anti-hallucination)     │
                     └────────────┬───────────────┘
                                  │ speech detected
                                  ▼
                     ┌──────────────────────────┐
                     │   Wake Word Detection      │  mlx-whisper base (~140MB)
                     │   "Hey Jarvis" / variants  │  2.5s sliding windows
                     └────────────┬───────────────┘
                                  │ triggered
                                  ▼
                     ┌──────────────────────────┐
                     │   Speech Recording         │  VAD + min 2s capture
                     │   (same stream, zero gap)  │  High-pass pre-filtered
                     └────────────┬───────────────┘
                                  │ audio captured
                                  ▼
                     ┌──────────────────────────┐
                     │   Speech-to-Text           │  mlx-whisper small (~240MB)
                     │   (anti-hallucination)     │  Apple Neural Engine
                     └────────────┬───────────────┘
                                  │ text
                                  ▼
                    ┌─────────────────────────────┐
                    │     Two-Stage Tool Router     │
                    │                               │
                    │  Stage 1: Keyword Pre-Filter  │  ⚡ 0ms — catches 90%
                    │  Stage 2: Phi-3 Classifier    │  🧠 ~3s  — complex cases
                    └──────────┬──────────┬─────────┘
                     tool found│          │no tool
                               ▼          ▼
                    ┌───────────────┐  ┌──────────────────────┐
                    │  Tool Execute  │  │  NLU / Brain          │
                    │  system_info   │  │  Phi-3 Mini 3.8B      │
                    │  mac_control   │  │  + Memory Context     │
                    │  web_search    │  │  + Identity Firewall  │
                    │  reminder      │  │  + Post-Processing    │
                    │  whatsapp      │  └──────────┬───────────┘
                    └──────┬────────┘             │
                           │                      │
                           ▼                      ▼
                    ┌──────────────────────────────────┐
                    │   Text-to-Speech (macOS native)   │  Voice: Daniel
                    └──────────────┬───────────────────┘
                                   │
                                   ▼
                    ┌──────────────────────────────────┐
                    │   Event Bus → WebSocket → Dashboard│
                    │   Real-time UI at :8765            │
                    └──────────────────────────────────┘
```

---

## 📊 Real-Time Dashboard

A stunning glassmorphic dashboard powered by FastAPI + WebSocket:

**Design:** Dark mode, `backdrop-blur`, animated gradient orbs, cyan/purple accents, Orbitron + JetBrains Mono fonts.

| Component | What It Shows |
|-----------|--------------|
| **Status Ring** | Animated SVG — changes color/speed for idle, listening, thinking, speaking |
| **Conversation Feed** | Live chat bubbles — user (cyan) and Jarvis (white), auto-scroll |
| **System Gauges** | Battery % and RAM % with circular SVG gauges, color-coded |
| **Tool Router Feed** | Every tool classification with tool name, action, params, timestamp |
| **Memory Stats** | Conversation count, user facts, tools used |
| **Header Bar** | Uptime, wake count, exchange count, connection status |

**Tech:** Single self-contained HTML file (~550 lines), no build step, no npm, no React. Telemetry pushed every 3s via WebSocket. Auto-reconnect on disconnect.

---

## 🧠 Memory System

Persistent memory using **ChromaDB** with sentence-transformer embeddings:

| Feature | How It Works |
|---------|-------------|
| **Conversation History** | Every exchange stored with embeddings. Top 3 most relevant retrieved per query via semantic search. |
| **User Profile** | Auto-extracted facts ("The user is an AI engineer", "The user studies at Brainware University"). Stored as key-value pairs. |
| **Context Injection** | Profile facts + relevant past exchanges injected into Phi-3's system prompt every turn. |
| **Persistence** | Stored at `~/.jarvis/memory/` — survives app restarts, accumulates over time. |
| **Identity Firewall** | Every memory line rewritten to "The user: ..." before injection. Prevents Phi-3 from adopting user traits. |

---

## 🔧 Two-Stage Tool Router

Most voice assistants use either keyword matching (brittle) or LLM classification (slow). J.A.R.V.I.S. uses **both**:

### Stage 1: Keyword Pre-Filter (0ms)

Catches 90% of commands instantly with zero LLM calls:

```python
"What time is it?"           → system_info/time      ⚡ instant
"Open Brave"                 → mac_control/open_app   ⚡ instant
"Set volume to 50%"          → mac_control/volume_set  ⚡ instant
"Weather in Mumbai?"         → web_search              ⚡ instant
"Bitcoin price?"             → web_search              ⚡ instant
"Set timer for 5 minutes"    → reminder/timer          ⚡ instant
"Take a screenshot"          → mac_control/screenshot  ⚡ instant
```

Smart app name extraction: `"Can you please open Brave browser for me?"` → `Brave`

### Stage 2: Phi-3 Classification (~3s)

Only called for complex cases needing parameter extraction:

```python
"Send Mom a WhatsApp saying I'll be late"  → whatsapp/send (contact + message)
"Tell me a joke about programming"          → none (conversation)
```

### Why Two Stages?

Phi-3-mini (3.8B) occasionally hallucinates tool names (`macOS System` instead of `mac_control`) or misroutes obvious commands. The keyword pre-filter eliminates this for common patterns while Phi-3 handles the long tail.

---

## 🛡️ Identity Protection (3 Layers)

Phi-3-mini has a fundamental weakness: it reads memory facts like "aspiring AI engineer" and thinks they describe **itself**. This produces responses like *"As an aspiring AI engineer myself..."*. Three layers prevent this:

| Layer | Where | What It Does |
|-------|-------|-------------|
| **Layer 0** | `nlu.py` — before Phi-3 | Identity questions ("Who am I?", "Who are you?") return hardcoded responses. Phi-3 is never called. |
| **Layer 1** | `nlu.py` — system prompt | Every memory line rewritten to "The user: ..." with explicit markers: "These facts describe the human, NOT you." |
| **Layer 2** | `nlu.py` — post-processing | 30+ poison phrases detected in output ("as an engineer myself", "within Brainware", "quest for knowledge"). If found, entire response replaced. |

---

## 💾 Memory Budget

Running on **8GB unified memory** — every megabyte is a conscious decision:

| Component | RAM Usage | Device | Purpose |
|-----------|-----------|--------|---------|
| Python + deps | ~200MB | CPU | Minimal dependency footprint |
| Wake word (whisper-base) | ~140MB | Neural Engine | Wake detection in 2.5s windows |
| STT (whisper-small) | ~240MB | Neural Engine | Speech transcription |
| Phi-3 Mini (Ollama) | ~2.3GB | Metal GPU | NLU + tool routing |
| ChromaDB + embeddings | ~130MB | CPU | Persistent vector memory |
| FastAPI dashboard | ~15MB | CPU | Real-time WebSocket UI |
| macOS TTS | ~0MB | System | Native `say` command |
| **Total (peak)** | **~3.0GB / 8GB** | | **~42% usage, 58% headroom** ✅ |

---

## 🛠️ Tech Stack

| Layer | Technology | Why |
|-------|-----------|-----|
| Audio Streaming | `sounddevice` InputStream + callback | Zero-gap, continuous capture |
| DSP Filter | 85Hz IIR high-pass (numpy) | Removes fan/AC hum before processing |
| Wake Word | `mlx-whisper` base | STT-based detection (openWakeWord broken on M1) |
| Speech-to-Text | `mlx-whisper` small | Best accuracy-to-size ratio on Neural Engine |
| NLU / Brain | Phi-3 Mini 3.8B via Ollama | Fault-isolated, Metal GPU, Q4 quantized |
| Tool Router | Keyword pre-filter + Phi-3 | 0ms for common commands, LLM for complex |
| Memory | ChromaDB + sentence-transformers | Persistent vector search, ~130MB |
| Dashboard | FastAPI + WebSocket + vanilla HTML | ~15MB, single file, no build step |
| Text-to-Speech | macOS native `say` (Daniel) | Zero RAM, built into the OS |
| Config | YAML | Single source of truth for all parameters |

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
git clone https://github.com/Swapnil-bo/jarvis.git
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
2. Open **http://127.0.0.1:8765** for the live dashboard
3. Say **"Hey Jarvis"** (or "Buddy") at normal conversation volume
4. Wait for **"Yes?"**
5. Ask your question or give a command
6. Watch the dashboard update in real-time
7. Repeat

---

## 📁 Project Structure

```
jarvis/
├── config/
│   └── jarvis_config.yaml              # All tunable parameters
├── src/
│   ├── main.py                         # Entry point — voice loop + event bus hooks
│   ├── core/
│   │   ├── audio.py                    # Streaming mic + 85Hz high-pass DSP
│   │   ├── wake_word.py                # STT-based wake word + dual energy gate
│   │   ├── stt.py                      # Speech-to-text (anti-hallucination)
│   │   ├── nlu.py                      # NLU + 3-layer identity protection
│   │   └── tts.py                      # Text-to-speech (macOS native)
│   ├── memory/
│   │   ├── memory_manager.py           # Orchestrates conversation + profile
│   │   ├── conversation_store.py       # ChromaDB conversation history
│   │   └── user_profile.py             # ChromaDB user fact storage
│   ├── tools/
│   │   ├── router.py                   # Two-stage router (keyword + Phi-3)
│   │   ├── system_info.py              # Time, date, battery
│   │   ├── mac_control.py              # Apps, volume, brightness, screenshot, lock
│   │   ├── web_search.py               # DuckDuckGo web search
│   │   ├── reminder.py                 # Timers and reminders
│   │   └── whatsapp.py                 # WhatsApp Desktop automation
│   ├── dashboard/
│   │   ├── events.py                   # Thread-safe event bus (queue.Queue)
│   │   ├── server.py                   # FastAPI + WebSocket server
│   │   └── static/
│   │       └── index.html              # Glassmorphic dashboard UI (~550 lines)
│   └── utils/
│       ├── config.py                   # YAML config loader
│       └── logger.py                   # Rich logging + psutil RAM monitoring
├── tests/
│   ├── diagnose_audio.py               # Audio pipeline diagnostic
│   └── fix_wake_word.py                # Wake word troubleshooter
├── logs/                               # Runtime logs with memory profiling
├── models/                             # Local model weights
├── docs/                               # Architecture notes
└── requirements-phase1.txt             # Python dependencies
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
- [x] **Phase 2: Memory & Context** — ChromaDB persistent memory
  - [x] Conversation history with semantic search
  - [x] User profile auto-extraction
  - [x] Context injection into NLU prompts
  - [x] Identity confusion firewall (3 layers)
- [x] **Phase 3: Tools & Actions** — 5 tool modules
  - [x] System info (time, date, battery)
  - [x] Mac control (apps, volume, brightness, screenshot, lock)
  - [x] Web search (DuckDuckGo)
  - [x] Reminders and timers
  - [x] WhatsApp messaging (Desktop automation)
  - [x] Two-stage router (keyword pre-filter + Phi-3)
- [x] **Phase 4: Visual Dashboard** — Real-time glassmorphic UI
  - [x] FastAPI + WebSocket backend
  - [x] Event bus architecture
  - [x] Status ring with state animations
  - [x] System telemetry gauges
  - [x] Live conversation feed
  - [x] Tool router activity log
- [ ] **Phase 5: Vision** — Screen OCR and webcam analysis
- [ ] **Phase 6: Code Writing** — Autocoding, execution, error loop

---

## 🛡️ Engineering Decisions

### Why Ollama Stays Separate (Not mlx-lm)
On 8GB, process isolation is a **feature**. If Phi-3 OOMs, only Ollama dies — audio stream, wake word, and Python app survive. This is the same architecture Apple uses for Siri.

### Why Two-Stage Routing (Not Just LLM)
Phi-3-mini (3.8B params) is too small to reliably classify every command. It hallucinates tool names (`macOS System`), misroutes weather to `system_info`, and fails on phrasing variations. The keyword pre-filter catches 90% of commands in 0ms with 100% accuracy. Phi-3 handles only the remaining complex cases.

### Why Not Async/Multiprocessing
M1 has one Metal GPU. Whisper and Phi-3 can't run in parallel — they'd fight over GPU memory. The pipeline is inherently sequential: don't listen while speaking, don't think while listening.

### Why STT-Based Wake Word (Not openWakeWord)
openWakeWord's embedding model produces dead inference on M1 — max confidence 0.000017 across 250 chunks. No Apple Silicon TFLite wheel for Python 3.11. We use whisper-base doing 2.5s sliding window transcription with a dual energy gate instead.

### The Identity Confusion Problem
Small LLMs (3-4B params) can't reliably separate "facts about the user in the system prompt" from "facts about themselves." Phi-3 reads `"The user is an AI engineer"` and responds with `"As an AI engineer myself..."`. Our 3-layer defense (hardcoded shortcuts + memory rewriting + output filtering) is the only reliable solution short of upgrading to a larger model.

### The "Thank You" Hallucination Fix
Whisper hallucinates "Thank you for watching" on silence because `condition_on_previous_text=True` creates feedback loops. Fix: `False`, tighter `compression_ratio_threshold=1.8`, higher `no_speech_threshold=0.5`.

---

## 💡 What Makes This Different

| Typical AI Assistant | J.A.R.V.I.S. |
|---------------------|---------------|
| Wraps OpenAI's API | Every model runs on-device |
| Costs $$$/month | $0 after setup |
| Sends voice to cloud | Voice never leaves machine |
| Single-purpose chatbot | 5 tool modules + dashboard + memory |
| No documentation | Every design decision explained |
| Needs 16GB+ RAM | Runs on 8GB with 58% headroom |
| Keyword-only routing | Two-stage: keywords (fast) + LLM (smart) |
| No persistence | ChromaDB memory across sessions |

---

## 🔧 Debugging War Stories

Building a local AI assistant on 8GB taught me things no tutorial covers:

1. **openWakeWord is dead on M1** — TFLite has no Apple Silicon wheel. ONNX loads but returns zero confidence. Pivoted to STT-based detection.

2. **`sd.rec()` drops 40% of audio** — Each call has ~15ms gaps. Over 3 seconds, speech becomes garbled. Fixed with persistent InputStream + callback queue.

3. **Silence threshold is hardware-specific** — Started at 500, then 100, finally 30. MacBook Air M1 mic baseline RMS is ~2-7.

4. **Whisper hallucinates on silence** — "Thank you for watching" from fan noise. `condition_on_previous_text=False` is mandatory.

5. **Phi-3 adopts user identity** — Reads memory facts and says "as an engineer myself." Required 3-layer identity protection system.

6. **Phi-3 invents tool names** — Returns `macOS System` instead of `mac_control`. Keyword pre-filter handles this.

7. **WhatsApp Desktop automation is fragile** — Electron app doesn't respond to AppleScript reliably. Required coordinate-based clicking with cliclick.

8. **ChromaDB needs metadata** — Adding documents without `metadatas` param causes `NoneType.get()` crashes on retrieval.

9. **YAML multiline strings break on colons** — System prompt with colons corrupts config parsing. Use `|` pipe syntax.

10. **Accent handling beats model upgrades** — Adding "jalvis" to trigger phrases solved Indian English recognition instantly.

---

## 🤝 About

Built by **Swapnil Hazra** ([@Swapnil-bo](https://github.com/Swapnil-bo)) — aspiring AI engineer, student at Brainware University.

Part of the **100 Days of Vibe Coding** challenge. This project proves that useful AI doesn't need a datacenter — it can run on a $999 laptop with zero ongoing costs.

⭐ **Star this repo** if you think local AI is the future.

---

*Built with ❤️ on Apple Silicon. Every decision documented. Every megabyte justified.*