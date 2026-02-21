"""
J.A.R.V.I.S. Text-to-Speech Engine
=====================================
Uses macOS native `say` command for speech output.

Why macOS `say` instead of Piper TTS?
  - Zero additional RAM — it's a system process, not loaded into our Python memory.
  - Zero setup — it's built into every Mac.
  - Decent quality with voices like "Daniel" (British) or "Samantha" (American).
  - Piper TTS would add ~200MB RAM. On 8GB, that's significant.
  - We'll add Piper as an upgrade option in a later phase if RAM allows.

Available voices (run `say -v '?'` in Terminal to see all):
  - "Daniel"   → British English male (recommended for J.A.R.V.I.S. vibe)
  - "Samantha" → American English female
  - "Alex"     → American English male (most natural, but ~800MB download)
"""

import subprocess

from src.utils.config import load_config
from src.utils.logger import get_logger

logger = get_logger("core.tts")


class TextToSpeech:
    """
    Speaks text aloud using macOS native text-to-speech.
    """

    def __init__(self):
        config = load_config()
        tts_cfg = config["tts"]

        self.voice: str = tts_cfg["macos_voice"]    # "Daniel"
        self.rate: int = tts_cfg["macos_rate"]       # 190

        logger.info(f"TTS engine: macOS say, voice={self.voice}, rate={self.rate}")

        # Verify the voice exists
        self._verify_voice()

    def _verify_voice(self):
        """Check if the configured voice is available on this Mac."""
        try:
            result = subprocess.run(
                ["say", "-v", "?"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            available_voices = result.stdout
            if self.voice.lower() not in available_voices.lower():
                logger.warning(
                    f"Voice '{self.voice}' not found. "
                    f"Falling back to system default. "
                    f"Run `say -v '?'` to see available voices."
                )
                self.voice = ""  # Empty = system default
            else:
                logger.info(f"✅ Voice '{self.voice}' is available")
        except Exception as e:
            logger.warning(f"Could not verify voice: {e}")

    def speak(self, text: str):
        """
        Speak the given text aloud.

        Args:
            text: The text to speak. Will be sanitized for shell safety.

        The `say` command runs as a subprocess. It blocks until speech
        is complete, which is what we want — we don't want to listen
        for a new wake word while J.A.R.V.I.S. is still talking.
        """
        if not text:
            logger.warning("TTS called with empty text")
            return

        logger.info(f"🔊 Speaking: \"{text[:80]}{'...' if len(text) > 80 else ''}\"")

        cmd = ["say"]
        if self.voice:
            cmd.extend(["-v", self.voice])
        cmd.extend(["-r", str(self.rate)])
        cmd.append(text)

        try:
            subprocess.run(
                cmd,
                timeout=30,  # Safety: don't let TTS hang forever
                check=True,
            )
        except subprocess.TimeoutExpired:
            logger.warning("TTS timed out after 30 seconds")
        except subprocess.CalledProcessError as e:
            logger.error(f"TTS error: {e}")
        except Exception as e:
            logger.error(f"Unexpected TTS error: {e}")