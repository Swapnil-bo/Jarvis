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

    log_memory(logger)
    logger.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    logger.info("✅ All systems online")
    logger.info("🎙️  Say 'Hey Jarvis' to activate")
    logger.info("⌨️  Press Ctrl+C to quit")
    logger.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    tts.speak("Jarvis is online and ready, sir.")

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

                # Step 2: Record speech
                speech_audio = audio_capture.record_speech()

                if speech_audio is None:
                    tts.speak("I didn't hear anything. Try again.")
                    gc.collect()
                    continue

                # Step 3: Transcribe
                user_text = stt.transcribe(speech_audio)

                # Free the raw audio immediately — it's large and no longer needed
                del speech_audio
                gc.collect()

                if not user_text:
                    tts.speak("I couldn't understand that. Could you repeat?")
                    continue

                logger.info(f"👤 You: \"{user_text}\"")

                # Step 4: Think
                response = nlu.think(user_text)
                logger.info(f"🤖 Jarvis: \"{response}\"")

                # Step 5: Speak
                tts.speak(response)

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