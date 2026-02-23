"""
J.A.R.V.I.S. — Main Entry Point
==================================
Phase 1: Voice Core Pipeline
Phase 2: Memory & Context
Phase 3: Tools & Actions
Phase 4: Visual Dashboard

Run:  python -m src.main
"""

import gc
import os
import signal
import sys
import threading
import time

import psutil

from src.core.audio import AudioCapture
from src.core.wake_word import WakeWordDetector
from src.core.stt import SpeechToText
from src.core.nlu import NLUEngine
from src.core.tts import TextToSpeech
from src.memory import MemoryManager
from src.tools.router import ToolRouter
from src.tools.system_info import SystemInfoTool
from src.tools.mac_control import MacControlTool
from src.tools.reminder import ReminderTool
from src.tools.web_search import WebSearchTool
from src.tools.whatsapp import WhatsAppTool
from src.dashboard import events as dash_events
from src.utils.logger import get_logger, log_memory

logger = get_logger("main")

# Global reference for graceful shutdown
audio_capture = None


def graceful_shutdown(sig, frame):
    """Handle Ctrl+C cleanly."""
    global audio_capture
    print()
    logger.info("👋 J.A.R.V.I.S. shutting down. Goodbye, sir.")
    dash_events.emit({"type": "status", "state": "offline"})
    if audio_capture:
        audio_capture.close()
    sys.exit(0)


def print_banner():
    """Print startup banner."""
    banner = """
    ╔══════════════════════════════════════════════════════╗
    ║                                                      ║
    ║        ░█ ░█▀█ ░█▀▄ ░█  ░█ ░█ ░█▀▀                 ║
    ║        ░█ ░█▀█ ░█▀▄ ░▀▄▀  ░█ ░▀▀█                  ║
    ║        █▄ ░█ █ ░█ █  ░█   ░█ ░▀▀▀                   ║
    ║                                                      ║
    ║        MacBook Air M1 Edition — 100% Local           ║
    ║        Phase 1: Voice Core                           ║
    ║        Phase 2: Memory & Context                     ║
    ║        Phase 3: Tools & Actions                      ║
    ║        Phase 4: Visual Dashboard                     ║
    ║                                                      ║
    ╚══════════════════════════════════════════════════════╝
    """
    print(banner)


def start_telemetry(memory=None):
    """
    Background thread: emits RAM, battery, and memory stats every 3 seconds.
    """
    process = psutil.Process(os.getpid())

    def _loop():
        while True:
            try:
                proc_mem = process.memory_info().rss // (1024 * 1024)
                sys_mem = psutil.virtual_memory()

                event = {
                    "type": "telemetry",
                    "ram_process": proc_mem,
                    "ram_system": sys_mem.used // (1024 * 1024),
                    "ram_total": sys_mem.total // (1024 * 1024),
                }

                batt = psutil.sensors_battery()
                if batt:
                    event["battery"] = int(batt.percent)
                    event["battery_charging"] = batt.power_plugged

                if memory:
                    try:
                        stats = memory.get_stats()
                        event["exchanges"] = stats.get("total_exchanges", 0)
                        event["facts"] = stats.get("total_facts", 0)
                    except Exception:
                        pass

                dash_events.emit(event)
            except Exception:
                pass

            time.sleep(3)

    t = threading.Thread(target=_loop, daemon=True, name="telemetry")
    t.start()


def main():
    global audio_capture

    signal.signal(signal.SIGINT, graceful_shutdown)
    print_banner()

    # --------------------------------------------------
    # STARTUP
    # --------------------------------------------------
    logger.info("Initializing subsystems...")
    log_memory(logger)

    # 1. Audio capture
    try:
        audio_capture = AudioCapture()
    except Exception as e:
        logger.error(f"❌ Failed to open microphone: {e}")
        logger.error("   Check: System Settings → Privacy → Microphone → enable for Terminal/Cursor")
        sys.exit(1)

    # 2. Wake word
    wake_word = WakeWordDetector(audio_capture)
    log_memory(logger)

    # 3. Speech-to-Text
    stt = SpeechToText()

    # 4. NLU Brain
    nlu = NLUEngine()

    # 5. TTS
    tts = TextToSpeech()

    # 6. Memory system
    try:
        memory = MemoryManager()
        stats = memory.get_stats()
        logger.info(
            f"🧠 Memory: {stats.get('total_exchanges', 0)} past exchanges, "
            f"{stats.get('total_facts', 0)} user facts"
        )
    except Exception as e:
        logger.warning(f"⚠️  Memory system failed to load: {e}")
        logger.warning("   Continuing without memory (conversations won't be saved)")
        memory = None

    # 7. Tool Router
    tool_router = ToolRouter()
    tool_router.register_tool("system_info", SystemInfoTool())
    tool_router.register_tool("mac_control", MacControlTool())
    tool_router.register_tool("reminder", ReminderTool())
    tool_router.register_tool("web_search", WebSearchTool())
    tool_router.register_tool("whatsapp", WhatsAppTool())

    # 8. Dashboard Server (Phase 4)
    try:
        from src.dashboard import server as dash_server
        dash_server.start(port=8765)
    except Exception as e:
        logger.warning(f"⚠️  Dashboard failed to start: {e}")
        logger.warning("   Continuing without dashboard. Run: pip install fastapi uvicorn")

    # 9. Telemetry thread
    start_telemetry(memory)

    log_memory(logger)
    logger.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    logger.info("✅ All systems online")
    logger.info("🎙️  Say 'Hey Jarvis' to activate")
    logger.info("📊 Dashboard: http://127.0.0.1:8765")
    logger.info("⌨️  Press Ctrl+C to quit")
    logger.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    tts.speak("Jarvis is online and ready, sir.")
    audio_capture.flush_queue()

    dash_events.emit({"type": "status", "state": "online"})

    # --------------------------------------------------
    # MAIN LOOP
    # --------------------------------------------------
    cycle_count = 0

    while True:
        try:
            # Step 1: Listen for wake word
            if wake_word.listen_and_detect():
                logger.info("🎯 Wake word triggered")
                dash_events.emit({"type": "status", "state": "wake_detected"})
                dash_events.emit({"type": "wake_word", "phrase": getattr(wake_word, 'last_match', 'hey jarvis') or 'hey jarvis'})

                tts.speak("Yes?")
                audio_capture.flush_queue()

                # Step 2: Record speech
                dash_events.emit({"type": "status", "state": "listening"})
                speech_audio = audio_capture.record_speech()

                if speech_audio is None:
                    tts.speak("I didn't hear anything. Try again.")
                    audio_capture.flush_queue()
                    dash_events.emit({"type": "status", "state": "idle"})
                    gc.collect()
                    continue

                # Step 3: Transcribe
                dash_events.emit({"type": "status", "state": "thinking"})
                user_text = stt.transcribe(speech_audio)

                del speech_audio
                gc.collect()

                if not user_text:
                    tts.speak("I couldn't understand that. Could you repeat?")
                    audio_capture.flush_queue()
                    dash_events.emit({"type": "status", "state": "idle"})
                    continue

                logger.info(f"👤 You: \"{user_text}\"")
                dash_events.emit({"type": "transcription", "text": user_text})

                # Step 4: Try tool routing first
                tool_result = tool_router.route(user_text)

                if tool_result:
                    response = tool_result

                    # Emit routing event
                    route_info = getattr(tool_router, "last_route", {})
                    dash_events.emit({
                        "type": "routing",
                        "tool": route_info.get("tool", "unknown"),
                        "action": route_info.get("action", ""),
                        "params": route_info.get("params", {}),
                    })
                    logger.info(f"🤖 Jarvis: \"{response}\"")
                else:
                    # No tool — normal conversation
                    dash_events.emit({
                        "type": "routing",
                        "tool": "none",
                        "action": "chat",
                        "params": {},
                    })

                    memory_context = ""
                    if memory:
                        try:
                            memory_context = memory.build_context(user_text)
                        except Exception as e:
                            logger.warning(f"  Memory search failed (non-critical): {e}")

                    response = nlu.think(user_text, memory_context)
                    logger.info(f"🤖 Jarvis: \"{response}\"")

                # Step 5: Speak
                dash_events.emit({"type": "status", "state": "speaking"})
                dash_events.emit({"type": "response", "text": response})
                tts.speak(response)
                audio_capture.flush_queue()

                # Step 6: Save to memory
                if memory:
                    try:
                        memory.after_exchange(user_text, response)
                    except Exception as e:
                        logger.warning(f"  Memory save failed (non-critical): {e}")

                # Cycle complete
                cycle_count += 1
                gc.collect()
                dash_events.emit({"type": "status", "state": "idle"})

                logger.info(f"── Cycle {cycle_count} complete ──")
                log_memory(logger)
                wake_word.reset()

        except KeyboardInterrupt:
            graceful_shutdown(None, None)

        except OSError as e:
            logger.error(f"🎤 Audio device error: {e}")
            logger.info("   Attempting to recover in 3 seconds...")
            time.sleep(3)
            try:
                audio_capture.close()
                audio_capture = AudioCapture()
                wake_word = WakeWordDetector(audio_capture)
                logger.info("✅ Audio recovered")
            except Exception as recovery_error:
                logger.error(f"❌ Recovery failed: {recovery_error}")
                logger.error("   Please check your microphone and restart.")
                sys.exit(1)

        except Exception as e:
            logger.error(f"Unexpected error: {e}", exc_info=True)
            time.sleep(1)


if __name__ == "__main__":
    main()