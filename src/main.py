"""
J.A.R.V.I.S. — Main Entry Point (Final — Demo Ready)
=======================================================
Phase 1: Voice Core Pipeline

Run:  python -m src.main
"""

import gc
import signal
import sys
import time

from src.core.audio import AudioCapture
from src.core.wake_word import WakeWordDetector
from src.core.stt import SpeechToText
from src.core.nlu import NLUEngine
from src.core.tts import TextToSpeech
from src.memory import MemoryManager
from src.utils.logger import get_logger, log_memory

logger = get_logger("main")

# Global reference for graceful shutdown
audio_capture = None


def graceful_shutdown(sig, frame):
    """Handle Ctrl+C cleanly."""
    global audio_capture
    print()  # Clean newline after ^C
    logger.info("👋 J.A.R.V.I.S. shutting down. Goodbye, sir.")
    if audio_capture:
        audio_capture.close()
    sys.exit(0)


def print_banner():
    """Print a clean startup banner for demo videos."""
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
    ║                                                      ║
    ╚══════════════════════════════════════════════════════╝
    """
    print(banner)


def main():
    global audio_capture

    signal.signal(signal.SIGINT, graceful_shutdown)
    print_banner()

    # --------------------------------------------------
    # STARTUP
    # --------------------------------------------------
    logger.info("Initializing subsystems...")
    log_memory(logger)

    # 1. Audio capture — persistent mic stream with 85Hz high-pass filter
    try:
        audio_capture = AudioCapture()
    except Exception as e:
        logger.error(f"❌ Failed to open microphone: {e}")
        logger.error("   Check: System Settings → Privacy → Microphone → enable for Terminal/Cursor")
        sys.exit(1)

    # 2. Wake word — reads from shared audio stream
    wake_word = WakeWordDetector(audio_capture)
    log_memory(logger)

    # 3. Speech-to-Text
    stt = SpeechToText()

    # 4. NLU Brain
    nlu = NLUEngine()

    # 5. TTS
    tts = TextToSpeech()

    # 6. Memory system (ChromaDB + embeddings, ~130MB)
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

    log_memory(logger)
    logger.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    logger.info("✅ All systems online")
    logger.info("🎙️  Say 'Hey Jarvis' to activate")
    logger.info("⌨️  Press Ctrl+C to quit")
    logger.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    tts.speak("Jarvis is online and ready, sir.")

    # Flush the audio queue — mic just heard Jarvis speak through the laptop
    # speakers. Without this, wake word detector triggers on his own voice.
    audio_capture.flush_queue()

    # --------------------------------------------------
    # MAIN LOOP
    # --------------------------------------------------
    cycle_count = 0

    while True:
        try:
            # Step 1: Listen for wake word
            if wake_word.listen_and_detect():
                logger.info("🎯 Wake word triggered")

                tts.speak("Yes?")
                audio_capture.flush_queue()

                # Step 2: Record speech
                speech_audio = audio_capture.record_speech()

                if speech_audio is None:
                    tts.speak("I didn't hear anything. Try again.")
                    audio_capture.flush_queue()
                    gc.collect()
                    continue

                # Step 3: Transcribe
                user_text = stt.transcribe(speech_audio)

                # Free the raw audio immediately — it's large and no longer needed
                del speech_audio
                gc.collect()

                if not user_text:
                    tts.speak("I couldn't understand that. Could you repeat?")
                    audio_capture.flush_queue()
                    continue

                logger.info(f"👤 You: \"{user_text}\"")

                # Step 4: Build memory context
                memory_context = ""
                if memory:
                    try:
                        memory_context = memory.build_context(user_text)
                    except Exception as e:
                        logger.warning(f"  Memory search failed (non-critical): {e}")

                # Step 5: Think (with memory context injected)
                response = nlu.think(user_text, memory_context)
                logger.info(f"🤖 Jarvis: \"{response}\"")

                # Step 6: Speak
                tts.speak(response)

                # Flush audio queue — mic heard Jarvis speak through speakers
                audio_capture.flush_queue()

                # Step 7: Save to memory
                if memory:
                    try:
                        memory.after_exchange(user_text, response)
                    except Exception as e:
                        logger.warning(f"  Memory save failed (non-critical): {e}")

                # Cycle complete — force garbage collection to reclaim
                # whisper's temporary buffers (~100-200MB of tensors).
                # On 8GB this gets us back to baseline faster.
                cycle_count += 1
                gc.collect()

                logger.info(f"── Cycle {cycle_count} complete ──")
                log_memory(logger)
                wake_word.reset()

        except KeyboardInterrupt:
            graceful_shutdown(None, None)

        except OSError as e:
            # Mic disconnect, audio device error, etc.
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