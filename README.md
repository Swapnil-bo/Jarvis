<div align="center">

# J.A.R.V.I.S.

### Just A Rather Very Intelligent System

**A fully local, voice-activated AI assistant running on a MacBook Air M1 (8GB)**

No cloud APIs. No subscriptions. No data leaves your machine. Ever.

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://python.org)
[![Ollama](https://img.shields.io/badge/Ollama-Local%20LLM-green.svg)](https://ollama.com)
[![Apple Silicon](https://img.shields.io/badge/Apple%20Silicon-M1%20Optimized-black.svg)](https://apple.com)
[![100 Days of Vibe Coding](https://img.shields.io/badge/Challenge-100%20Days%20of%20Vibe%20Coding-orange.svg)](#)

</div>

---

<!-- Replace with your actual demo recording -->
<!-- 
## 🎬 Demo

<div align="center">

![Jarvis Demo](docs/demo.gif)

*"Hey Jarvis, read my screen and tell me which libraries I'm using."*

</div>

--- 
-->

## Why This Exists

Every "AI assistant" tutorial uses OpenAI's API and calls it a day. This project asks a harder question: **can you build a genuinely useful voice assistant that runs 100% locally on the cheapest Apple Silicon Mac?**

The answer is yes. Jarvis handles voice commands, controls your Mac, searches the web, reads your screen, sees through your webcam, writes and executes code, remembers your conversations, and self-heals when things break — all within 3.1GB of RAM on an 8GB machine.

---

## What It Can Do

| You say | Jarvis does | Response time |
|---|---|---|
| "What time is it?" | Returns system time | < 200ms |
| "Open Spotify" | Launches any macOS app | < 200ms |
| "Set volume to 40 percent" | Adjusts system volume precisely | < 200ms |
| "Take a screenshot" | Captures screen to desktop | < 200ms |
| "What's the weather in Kolkata?" | Searches the web, speaks result | ~2s |
| "What's the price of Bitcoin?" | Real-time web search | ~2s |
| "Send a WhatsApp to Mom saying I'll be late" | Composes and sends via WhatsApp Web | ~4s |
| "Set a timer for 10 minutes" | Starts countdown timer | < 200ms |
| "Read my screen" | Native OCR, reads all visible text | < 1s |
| "Read my screen and tell me what libraries I'm using" | OCR → LLM reasoning → spoken analysis | ~5s |
| "What's on my screen?" | Screenshot → LLaVA-Phi3 visual description | ~8s |
| "Can you see me?" | Webcam → LLaVA-Phi3 description | ~8s |
| "Write a script to check my disk usage" | Generates Python → executes → speaks output | ~12s |
| "Write a script to list files in my downloads" | Code gen → execution → auto-fix if error | ~12s |
| "Who am I?" | Recalls name, interests, goals from memory | < 200ms |
| "Who are you?" | Identity-safe hardcoded response | < 200ms |

**95% of commands never touch the LLM** — they're handled by the keyword pre-filter in under 200ms.

---

## Architecture

```
                            ┌──────────────────┐
                            │   "Hey Jarvis"    │
                            └────────┬─────────┘
                                     │
                            ┌────────▼─────────┐
                            │   Wake Word       │  openWakeWord
                            │   Detection       │  (Neural Engine)
                            └────────┬─────────┘
                                     │
                            ┌────────▼─────────┐
                            │   Speech-to-Text  │  mlx-whisper
                            │   (Whisper Small) │  (Neural Engine)
                            └────────┬─────────┘
                                     │
                     ┌───────────────▼───────────────┐
                     │       Two-Stage Router         │
                     │                                │
                     │  Stage 1: Keyword Pre-Filter   │  < 1ms
                     │  (catches 95% of commands)     │
                     │                                │
                     │  Stage 2: Phi-3 Classification │  ~3s
                     │  (only for ambiguous queries)  │
                     └──┬────┬────┬────┬────┬────┬───┘
                        │    │    │    │    │    │
                 ┌──────┘    │    │    │    │    └──────┐
                 ▼           ▼    ▼    ▼    ▼           ▼
            ┌────────┐  ┌──────┐ ... ┌───────┐   ┌──────────┐
            │System  │  │ Mac  │     │Vision │   │  Code    │
            │Info    │  │Ctrl  │     │(OCR/  │   │Executor  │
            │        │  │      │     │LLaVA) │   │+ Self-   │
            │time,   │  │apps, │     │       │   │  Heal    │
            │date,   │  │vol,  │     │screen,│   │          │
            │battery │  │bright│     │webcam │   │write,run,│
            └────────┘  └──────┘     └───────┘   │fix,rerun │
                                                  └──────────┘
                        │    │    │    │
                        ▼    ▼    ▼    ▼
                   ┌─────────────────────────┐
                   │    NLU Engine (Phi-3)    │  Metal GPU
                   │    + Memory Context      │  ~2.3GB
                   │    + Identity Firewall   │
                   └────────────┬────────────┘
                                │
                       ┌────────▼─────────┐
                       │  Text-to-Speech   │  macOS native
                       │  (Daniel voice)   │  (0 RAM)
                       └────────┬─────────┘
                                │
                       ┌────────▼─────────┐
                       │  Real-Time        │  FastAPI
                       │  Dashboard        │  + WebSocket
                       └──────────────────┘
```

---

## The 6 Phases

### Phase 1: Voice Core Pipeline
The foundation of everything. A complete voice loop running locally on Apple Silicon.

- **Wake Word Detection:** STT-based matching using Whisper Base on the Neural Engine. Listens for "Jarvis", "Buddy", "Hey Jarvis", "Hey Buddy" and variants. Anti-hallucination v5 engine with configurable sensitivity thresholds
- **Speech-to-Text:** MLX-accelerated Whisper Small (~240MB). Transcribes speech in ~2 seconds with high accuracy
- **Audio Capture:** Zero-gap streaming at 16kHz with 80ms chunks. High-pass DSP filter at 85Hz cuts air conditioning, traffic, and room hum. Adaptive silence detection with configurable thresholds
- **Text-to-Speech:** macOS native `say` command with Daniel voice at 190 WPM. Zero RAM, zero latency, zero setup

### Phase 2: Memory & Context
Jarvis remembers who you are and what you've talked about.

- **Conversation Store:** ChromaDB vector database stores past exchanges (~130MB). Semantic search retrieves relevant context for each new query
- **User Profile:** Persistent facts about the user (name, interests, goals, university). Currently stores 8+ profile facts
- **Identity Firewall (3 layers):** Phi-3 Mini (3.8B) confuses "facts about the user" with "facts about itself." Three protection layers prevent this:
  - **Layer 0:** Hardcoded shortcut — identity questions bypass Phi-3 entirely
  - **Layer 1:** Memory rewriting — every fact is prefixed with "The user:" before injection
  - **Layer 2:** Output poison-phrase detection — catches and replaces confused responses

### Phase 3: Tools & Actions
Seven registered tools, each with a dedicated handler.

- **System Info:** Time, date, battery level — instant, no LLM needed
- **Mac Control:** Open/close apps, volume (up/down/mute/set level), brightness (up/down), screenshot, lock screen. Smart extraction parses app names and numeric levels from natural speech
- **Web Search:** Weather, news, prices, scores, general queries. Keyword detection for common patterns, LLM classification for complex searches
- **WhatsApp:** Sends messages via WhatsApp Web automation. Phi-3 extracts contact name and message from natural speech
- **Reminders:** Set timers and reminders with automatic time extraction. "Remind me in 10 minutes to call Mom" → parses minutes + message
- **Vision:** Screen OCR, screen description, webcam description (see Phase 5)
- **Code Executor:** Write, run, and self-heal Python scripts (see Phase 6)

### Phase 4: Real-Time Dashboard
A glassmorphic web interface served via FastAPI + WebSocket at `http://127.0.0.1:8765`.

- **System Telemetry:** Animated status rings showing RAM usage, CPU load, model state
- **Live Conversation:** Messages stream in real-time as Jarvis listens, thinks, and speaks
- **Pipeline State:** Visual indicator shows current stage (🎙️ listening → 🧠 thinking → 🔊 speaking)
- **Memory Stats:** Live count of past exchanges and user profile facts

### Phase 5: Multimodal Vision
Three vision capabilities with zero permanent RAM overhead.

| Capability | Engine | Speed | RAM Cost |
|---|---|---|---|
| **Screen OCR** | macOS Vision framework | ~200ms | 0 MB (native) |
| **Describe Screen** | LLaVA-Phi3 via Ollama | 5–8s | Shared with Phi-3 |
| **Webcam Describe** | imagesnap + LLaVA-Phi3 | 5–8s | Shared with Phi-3 |

**Key design choice:** LLaVA-Phi3 shares base layers with Phi-3 Mini. Ollama auto-swaps between them — no manual model management, no wasted RAM.

**Vision + Reasoning pipeline:** Vision results aren't just returned raw. When you ask "Read my screen and tell me which libraries I'm using," the OCR text is fed back into Phi-3 along with your original question. Jarvis *reasons* about what it sees, not just repeats it.

### Phase 6: Code Execution & Agentic Flow
Jarvis writes, runs, and fixes Python scripts from voice commands.

- **Code Generation:** Phi-3 generates Python code in `raw` mode (bypasses response cleaning that would strip backticks)
- **Sandboxed Execution:** Scripts run in `src/workspace/` with a 30-second timeout. Bad code can't freeze your Mac
- **Self-Healing Loop:** If a script throws an error:
  1. Catches the traceback
  2. Classifies the error type (AttributeError, TypeError, ImportError, SyntaxError)
  3. Builds a targeted fix prompt with the error hint
  4. Asks Phi-3 to fix only the broken line
  5. Re-extracts and re-executes the corrected code
  6. If it fails twice, gives up gracefully instead of looping

---

## RAM Budget

Everything fits in 8GB with room to breathe. Peak usage: **39% of total RAM.**

| Component | RAM | Runs On |
|---|---|---|
| Python + all dependencies | ~200 MB | CPU |
| Wake word detector (Whisper Base) | ~140 MB | Neural Engine |
| Whisper STT (Small) | ~240 MB | Neural Engine |
| Phi-3 Mini OR LLaVA-Phi3 | ~2.3 GB | Metal GPU |
| ChromaDB vector memory | ~130 MB | CPU |
| FastAPI + WebSocket dashboard | ~15 MB | CPU |
| macOS TTS + OCR | ~0 MB | System frameworks |
| **Peak total** | **~3.1 GB** | **39% of 8 GB** |

Models share the Metal GPU — Ollama swaps between Phi-3 and LLaVA-Phi3 automatically. Only one is loaded at a time.

---

## Tech Stack

| Layer | Technology | Why This One |
|---|---|---|
| Language Model | Phi-3 Mini 3.8B (Q4) | Best reasoning quality per GB at this size |
| Vision Model | LLaVA-Phi3 3.8B | Shares base weights with Phi-3 — efficient swapping |
| Speech-to-Text | mlx-whisper (Small) | MLX-native, runs on Neural Engine, not GPU |
| Text-to-Speech | macOS `say` (Daniel) | Zero RAM, zero latency, zero configuration |
| Screen OCR | macOS Vision framework | Native API, ~200ms, zero RAM overhead |
| Webcam Capture | imagesnap | Lightweight macOS CLI, no heavy deps |
| Vector Memory | ChromaDB | Local embedded database, no server needed |
| Model Serving | Ollama | Handles Metal GPU allocation + model swapping |
| Dashboard | FastAPI + WebSocket | Async, lightweight, real-time bidirectional |
| ML Framework | MLX (Apple) | Purpose-built for Apple Silicon |
| Audio DSP | NumPy + custom filters | High-pass at 85Hz, minimal overhead |

**Zero cloud dependencies.** No OpenAI. No Google. No Anthropic API. No API keys. Everything runs on `localhost`.

---

## Project Structure

```
jarvis/
├── src/
│   ├── core/                     # Voice pipeline
│   │   ├── audio.py              # Zero-gap streaming capture + 85Hz high-pass DSP
│   │   ├── wake_word.py          # STT-based wake word detection (v5 anti-hallucination)
│   │   ├── stt.py                # MLX-Whisper speech-to-text
│   │   ├── nlu.py                # Phi-3 NLU + identity firewall + raw mode
│   │   └── tts.py                # macOS native text-to-speech
│   │
│   ├── memory/                   # Persistent context
│   │   ├── conversation.py       # ChromaDB conversation store (137+ exchanges)
│   │   └── user_profile.py       # User facts (name, interests, goals)
│   │
│   ├── tools/                    # Registered tool handlers
│   │   ├── router.py             # Two-stage routing (keyword + Phi-3)
│   │   ├── system_info.py        # Time, date, battery
│   │   ├── mac_control.py        # Apps, volume, brightness, screenshot, lock
│   │   ├── web_search.py         # Internet search
│   │   ├── whatsapp.py           # WhatsApp Web messaging
│   │   ├── reminder.py           # Timers and reminders
│   │   └── code_executor.py      # Code gen + execution + self-healing
│   │
│   ├── vision/                   # Multimodal vision
│   │   └── vision.py             # OCR (native), screen describe, webcam describe
│   │
│   ├── dashboard/                # Real-time web UI
│   │   └── server.py             # FastAPI + WebSocket server
│   │
│   ├── ui/                       # Dashboard frontend (HTML/CSS/JS)
│   ├── utils/                    # Shared utilities
│   │   ├── config.py             # YAML config loader
│   │   └── logger.py             # Colored logging + RAM tracking
│   │
│   ├── workspace/                # Sandboxed code execution directory
│   └── main.py                   # Entry point — orchestrates everything
│
├── config/
│   └── config.yaml               # All settings in one place
├── tests/                        # Test suite
├── docs/                         # Documentation + screenshots
├── logs/                         # Runtime logs
├── .env                          # Environment variables
├── requirements.txt              # Python dependencies
└── README.md
```

---

## Getting Started

### Prerequisites

| Requirement | Minimum | Recommended |
|---|---|---|
| macOS | 13 (Ventura) | 14+ (Sonoma) |
| Chip | Apple M1 | Any Apple Silicon |
| RAM | 8 GB | 8 GB+ |
| Python | 3.11 | 3.11 |
| Disk Space | ~5 GB | ~5 GB (models + deps) |

### Step 1: Install System Dependencies

```bash
# Install Ollama (model server)
# Download from https://ollama.com or:
brew install ollama

# Install webcam capture tool
brew install imagesnap

# Verify
ollama --version
imagesnap --help
```

### Step 2: Clone & Setup Python

```bash
git clone https://github.com/swapnil-hazra/jarvis.git
cd jarvis

# Create virtual environment with Python 3.11
python3.11 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Step 3: Pull Models

```bash
# Primary language model (~2.2GB)
ollama pull phi3:mini

# Vision model (~2.9GB, shares base with Phi-3)
ollama pull llava-phi3

# Verify both are downloaded
ollama list
```

### Step 4: Grant macOS Permissions

Go to **System Settings → Privacy & Security** and enable:

| Permission | For | Why |
|---|---|---|
| **Microphone** | Terminal / iTerm2 | Voice input |
| **Screen Recording** | Terminal / iTerm2 | Screen OCR + describe |
| **Camera** | Terminal / iTerm2 | Webcam describe |

> Restart Terminal after granting permissions.

### Step 5: Run

```bash
source venv/bin/activate
python -m src.main
```

You'll see the startup banner with all 6 phases listed. Say **"Hey Jarvis"** or **"Hey Buddy"** to activate.

**Dashboard:** Open [http://127.0.0.1:8765](http://127.0.0.1:8765) in your browser.

---

## Configuration

All settings live in `config/config.yaml`:

```yaml
nlu:
  model: phi3:mini                    # Primary language model
  fallback_model: llama3.2:3b         # Fallback (optional)
  base_url: http://localhost:11434    # Ollama API
  context_window: 2048                # Token context length
  temperature: 0.7                    # Response creativity
  max_tokens: 300                     # Max response length
  system_prompt: "You are Jarvis..."  # Personality prompt

vision:
  vision_model: llava-phi3            # Vision model
  ollama_base_url: http://localhost:11434
  vision_timeout: 90                  # Max seconds for vision response

wake_word:
  trigger_phrases:
    - jarvis
    - buddy
    - hey jarvis
    - hey buddy
    - hi jarvis
    - okay jarvis
  listen_window: 2.5                  # Seconds to listen for wake word
  min_rms: 15.0                       # Minimum audio energy threshold
  peak_rms: 80.0                      # Peak energy threshold

audio:
  sample_rate: 16000
  channels: 1
  silence_threshold: 30
  min_record_seconds: 2.0
  max_record_seconds: 8.0
  highpass_freq: 85                   # DSP filter cutoff

tts:
  voice: Daniel
  rate: 190                           # Words per minute

dashboard:
  host: 127.0.0.1
  port: 8765
```

### Environment Variables (`.env`)

```bash
# Optional: override config.yaml settings
OLLAMA_BASE_URL=http://localhost:11434
JARVIS_LOG_LEVEL=INFO
```

---

## Performance Benchmarks

Measured on MacBook Air M1, 8GB RAM, macOS Sonoma.

| Operation | Time | Notes |
|---|---|---|
| Wake word detection | ~100ms | STT-based, Whisper Base |
| Speech-to-text | ~2s | Whisper Small, 8s audio |
| Keyword routing | < 1ms | No LLM call |
| Phi-3 classification | 3–5s | Only for ambiguous queries |
| Phi-3 chat response | 3–8s | Depending on length |
| System commands | < 200ms | Time, battery, app launch |
| Web search | ~2s | Query + parse |
| Native OCR | ~200ms | macOS Vision framework |
| LLaVA-Phi3 (first call) | 15–30s | Model loading from disk |
| LLaVA-Phi3 (cached) | 5–8s | Model already in RAM |
| Code generation | 8–12s | Phi-3 raw mode |
| Code execution | < 1s | subprocess with timeout |
| Self-healing retry | 8–12s | Re-generate + re-execute |
| Full voice loop | 5–10s | Wake → answer → speak |

---

## How the Two-Stage Router Works

Most AI assistants send every query to an LLM for classification. On an 8GB machine, that's 3–5 seconds wasted on "what time is it?"

Jarvis uses a two-stage approach:

**Stage 1 — Keyword Pre-Filter (< 1ms)**

```python
# Example: "set volume to 50 percent"
# Keyword match → instant routing, no LLM needed
if "volume" in text_lower:
    numbers = re.findall(r'\d+', text_lower)
    return {"tool": "mac_control", "action": "volume_set", "params": {"level": 50}}
```

Covers: time, date, battery, weather, news, prices, app launch/close, volume, brightness, screenshot, lock screen, timers, reminders, OCR, screen describe, webcam, code execution.

**Stage 2 — Phi-3 Classification (~3s)**

Only fires for queries the keyword filter can't handle — primarily WhatsApp messages (needs contact + message extraction) and genuinely ambiguous commands.

```
"Send a WhatsApp message to Mom saying I'll be late"
→ Phi-3 extracts: contact="Mom", message="I'll be late"
→ Routes to WhatsApp tool
```

**Result:** 95% of commands are handled in under 200ms. The LLM is reserved for where it actually adds value.

---

## Design Decisions

### Why Phi-3 Mini over larger models?
On 8GB RAM, every megabyte counts. Phi-3 Mini (3.8B, Q4 quantized) delivers the best reasoning quality per gigabyte at this parameter count. It handles routing, conversation, code generation, and vision reasoning — all within 2.3GB. Larger models (7B+) would leave no headroom for STT, memory, or vision.

### Why keyword routing instead of LLM-for-everything?
An LLM call takes 3–5 seconds on this hardware. Keyword matching takes < 1ms. For "set volume to 50" or "what time is it," burning 3 seconds on classification is wasteful. The two-stage approach gives instant response for obvious commands and LLM intelligence only when it's genuinely needed.

### Why macOS native OCR over LLaVA for text?
Apple's Vision framework reads screen text in ~200ms with zero RAM. LLaVA-Phi3 takes 5–8 seconds and requires 2.5GB. For pure text extraction, native always wins. LLaVA is reserved for visual understanding — "what app is open?" requires seeing the UI, not just reading text.

### Why STT-based wake word instead of a dedicated model?
Dedicated wake word models (like Porcupine) add another dependency and ~50MB of RAM. Since Whisper Base is already loaded for STT, we reuse it for wake word detection in a short 2.5-second listening window. Same engine, zero additional cost.

### Why no cloud fallback?
This project proves that a useful AI assistant can run entirely on consumer hardware. Adding a cloud fallback would undermine the core thesis. Every single feature works offline, on the cheapest Apple Silicon Mac you can buy.

### Why `raw=True` mode in the NLU engine?
The NLU post-processor strips markdown backticks to clean chat responses. But code generation needs those backticks to extract the code block. Rather than building a separate code-generation pipeline, a single `raw=True` flag bypasses cleaning — simple, surgical, zero overhead.

---

## Challenges & Lessons Learned

### Identity Confusion in Small LLMs
**Problem:** Phi-3 Mini (3.8B) cannot reliably separate "facts about the user" from "facts about itself." When given memory context like "The user studies at Brainware University," it responds "I study at Brainware University."

**Solution:** A three-layer identity firewall:
- Layer 0: Hardcoded shortcut bypasses Phi-3 entirely for identity questions
- Layer 1: Every memory fact is rewritten to start with "The user:" before prompt injection
- Layer 2: Output scanning detects 30+ poison phrases and replaces the response

### Model Swapping on Limited RAM
**Problem:** Switching between Phi-3 and LLaVA-Phi3 caused HTTP 404 errors. One model unloads while the other is still loading, and the API returns "model not found."

**Solution:** Retry loops with 3-second delays in the NLU engine, 60-second timeouts, and choosing models that share base weights (LLaVA-Phi3 is built on Phi-3) so swaps are faster.

### Code Generation vs. Post-Processing
**Problem:** The NLU engine strips markdown backticks (` ``` `) from responses to clean up chat output. But code generation returns code wrapped in backticks — the cleaner was deleting the actual code, leaving an empty response that got fed to the executor as "code."

**Solution:** A `raw=True` parameter that bypasses all post-processing. Code generation uses raw mode; normal chat uses cleaned mode.

### Self-Healing Without Infinite Loops
**Problem:** When generated code fails, asking the LLM to "fix it" can produce the same bug. Retrying infinitely wastes time and RAM.

**Solution:** One retry maximum with error-type classification. The fix prompt includes a specific hint ("An attribute does not exist on that object") so Phi-3 knows what went wrong, not just that something failed. If the fix also fails, Jarvis gives up gracefully.

---

## Troubleshooting

| Problem | Cause | Fix |
|---|---|---|
| `Ollama is not running` | Ollama server not started | Run `ollama serve` or open Ollama app |
| `404 error` on model call | Model unloaded during swap | Wait 3s, retry. NLU handles this automatically |
| `screencapture failed` | Missing Screen Recording permission | System Settings → Privacy → Screen Recording → enable Terminal |
| Webcam not working | Missing Camera permission | System Settings → Privacy → Camera → enable Terminal |
| No audio detected | Microphone permission or threshold | Check Privacy → Microphone. Lower `silence_threshold` in config |
| `ModuleNotFoundError` | Venv not activated or deps missing | `source venv/bin/activate && pip install -r requirements.txt` |
| High RAM (90%+) | Normal during model swap | Transient — drops after swap completes. `gc.collect()` runs automatically |
| Wake word not triggering | Background noise or wrong phrase | Move closer, reduce noise, or try "Hey Buddy" instead |
| Vision model slow first time | Model loading from disk | First call takes 15–30s. Subsequent calls: 5–8s |
| Code execution timeout | Script has infinite loop | 30s timeout kills it automatically. No action needed |

---

## Comparison with Cloud Assistants

| Feature | J.A.R.V.I.S. | Siri | Alexa | ChatGPT Voice |
|---|---|---|---|---|
| Runs 100% locally | ✅ | ❌ | ❌ | ❌ |
| No internet required | ✅ | ❌ | ❌ | ❌ |
| No subscription | ✅ | ✅ | ✅ | ❌ ($20/mo) |
| No data collection | ✅ | ❌ | ❌ | ❌ |
| Screen reading (OCR) | ✅ | ❌ | ❌ | ❌ |
| Webcam vision | ✅ | ❌ | ❌ | ✅ |
| Code generation + execution | ✅ | ❌ | ❌ | ✅ |
| Self-healing code | ✅ | ❌ | ❌ | ❌ |
| Persistent memory | ✅ | ❌ | ❌ | ✅ |
| Custom wake word | ✅ | ❌ | ❌ | ❌ |
| Open source | ✅ | ❌ | ❌ | ❌ |
| Works on 8GB RAM | ✅ | N/A | N/A | N/A |

---

## What's Next

- [ ] **Smarter NLU** — Upgrade to a larger local model when hardware allows (Phi-3 Medium, Mistral 7B)
- [ ] **Multi-step planning** — Chain multiple tools in a single voice command ("check disk space and clean temp files if low")
- [ ] **Proactive suggestions** — Jarvis notices patterns and offers help before you ask
- [ ] **Plugin system** — Custom tools without modifying core code
- [ ] **Streaming TTS** — Start speaking before the full response is generated
- [ ] **Multi-language** — Support for Hindi and Bengali voice input

---

## Contributing

Contributions are welcome! This project is built with a "vibe coding" philosophy — speed and working code over perfection.

```bash
# Fork the repo, then:
git clone https://github.com/YOUR_USERNAME/jarvis.git
cd jarvis
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Make your changes, test, then PR
```

**Areas that need help:**
- Better prompts for Phi-3 code generation
- Additional tool handlers (Spotify control, calendar, email)
- Dashboard UI improvements
- Test coverage

---

## Acknowledgments

- **[Ollama](https://ollama.com)** — For making local LLM serving dead simple
- **[MLX](https://github.com/ml-explore/mlx)** — Apple's ML framework that makes Whisper fly on Apple Silicon
- **[Phi-3](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct)** — Microsoft's tiny giant that powers everything
- **[LLaVA-Phi3](https://huggingface.co/xtuner/llava-phi-3-mini-hf)** — Vision + language in 3.8B params
- **[ChromaDB](https://www.trychroma.com)** — Embedded vector database with zero fuss
- **[Claude Opus](https://anthropic.com)** — AI engineering partner throughout the entire build

---



---

<div align="center">

**Built by [Swapnil Hazra](https://github.com/swapnil-bo)**

*100 Days of Vibe Coding Challenge — Day 1 to Done*

MacBook Air M1, 8GB RAM — proving you don't need a GPU cluster to build real AI.

---

*"It's not about the hardware in your hands. It's about the code in your head."*

⭐ **Star this repo if you think local AI assistants are the future** ⭐

</div>
