"""
J.A.R.V.I.S. Wake Word Detection (v5 — Anti-Hallucination Whisper Params)
===========================================================================
Upgrades in v5:
  - Whisper transcription now uses condition_on_previous_text=False.
    This is THE fix for phantom "Thank you" / "Subscribe" hallucinations.
    Without it, whisper feeds its own previous output as context, which
    causes it to hallucinate common phrases when given near-silence.
  - compression_ratio_threshold=1.8 (tighter than default 2.4).
    Rejects transcriptions that are suspiciously repetitive (a sign of
    hallucination — real speech rarely has compression ratio > 1.8).
  - no_speech_threshold=0.5: whisper's internal "is this speech?" detector.
    At 0.5, it's more aggressive at rejecting non-speech audio.
  - Audio is already high-pass filtered by AudioCapture (85Hz cutoff),
    so fan hum is removed before it reaches whisper.

Peak RMS filter (unchanged from v4):
  - avg_rms > 15: basic activity check
  - peak_rms > 80: ensures at least one chunk has a speech-like spike
"""

import numpy as np
import mlx_whisper

from src.utils.config import load_config
from src.utils.logger import get_logger

logger = get_logger("core.wake_word")


class WakeWordDetector:
    """
    Listens for "Hey Jarvis" using mlx-whisper with anti-hallucination tuning.
    """

    def __init__(self, audio_capture):
        config = load_config()
        ww_cfg = config["wake_word"]
        audio_cfg = config["audio"]

        self.audio = audio_capture

        self.trigger_phrases: list = ww_cfg.get("trigger_phrases", [
            "jarvis", "jalvis",
            "hey jarvis", "hey jalvis",
            "hi jarvis", "hi jalvis",
            "okay jarvis", "okay jalvis",
        ])

        self.sample_rate: int = audio_cfg["sample_rate"]
        self.listen_duration: float = ww_cfg.get("listen_duration_sec", 2.5)

        chunk_duration = audio_cfg["chunk_duration_ms"] / 1000.0
        self.chunks_per_window: int = int(self.listen_duration / chunk_duration)

        self.model_id: str = ww_cfg.get(
            "stt_model", "mlx-community/whisper-base-mlx"
        )

        # Energy gates
        self.min_rms: float = 15.0
        self.peak_rms: float = 80.0

        logger.info("Wake word detector: STT-based (v5 — anti-hallucination)")
        logger.info(f"  Trigger phrases: {self.trigger_phrases}")
        logger.info(
            f"  Listen window: {self.listen_duration}s ({self.chunks_per_window} chunks)"
        )
        logger.info(f"  Model: {self.model_id}")
        logger.info(f"  Sensitivity: min_rms={self.min_rms}, peak_rms={self.peak_rms}")

        # Warm up
        logger.info("  Warming up whisper model...")
        silent = np.zeros(
            int(self.sample_rate * self.listen_duration), dtype=np.float32
        )
        mlx_whisper.transcribe(
            silent,
            path_or_hf_repo=self.model_id,
            language="en",
            fp16=False,
        )
        logger.info("  ✅ Wake word detector ready")

    def _chunk_rms(self, chunk: np.ndarray) -> float:
        return float(np.sqrt(np.mean(chunk.astype(np.float32) ** 2)))

    def listen_and_detect(self) -> bool:
        """
        Listen for wake word with two-stage energy gate + anti-hallucination whisper.
        """
        chunks = []
        chunk_energies = []

        for _ in range(self.chunks_per_window):
            chunk = self.audio.get_audio_chunk()
            chunks.append(chunk)
            chunk_energies.append(self._chunk_rms(chunk))

        avg_rms = sum(chunk_energies) / len(chunk_energies)
        max_rms = max(chunk_energies)

        # Gate 1: Average energy
        if avg_rms < self.min_rms:
            return False

        # Gate 2: Peak energy (speech has peaks, hum doesn't)
        if max_rms < self.peak_rms:
            return False

        # --- Run whisper with anti-hallucination settings ---
        audio_int16 = np.concatenate(chunks)
        audio_float = audio_int16.astype(np.float32) / 32768.0

        try:
            result = mlx_whisper.transcribe(
                audio_float,
                path_or_hf_repo=self.model_id,
                language="en",
                fp16=False,
                # === Anti-hallucination parameters ===
                condition_on_previous_text=False,   # KEY: prevents "Thank you" loops
                compression_ratio_threshold=1.8,    # Reject repetitive hallucinations
                no_speech_threshold=0.5,            # Stricter speech detection
            )
            text = result.get("text", "").strip().lower()

            # Clean punctuation
            for char in ",.!?;:\"'":
                text = text.replace(char, "")
            text = text.strip()

            # Reject known hallucination patterns
            hallucinations = {
                "thank you", "thanks for watching", "subscribe",
                "you", "bye", "the end", "music", "applause",
                "silence", "so", "okay", "oh", "hmm", "um",
                "thank you for watching", "please subscribe",
                "like and subscribe", "see you next time",
                "thanks", "the", "and", "i", "a", "it",
            }
            if text in hallucinations:
                return False

            # Reject very short outputs (1-2 chars are always noise)
            if len(text) < 3:
                return False

            if text:
                for phrase in self.trigger_phrases:
                    if phrase in text:
                        logger.info(
                            f"🔊 Wake word detected! Heard: \"{text}\" "
                            f"(matched: \"{phrase}\") "
                            f"[avg_rms={avg_rms:.0f}, peak={max_rms:.0f}]"
                        )
                        return True

                logger.debug(
                    f"  heard: \"{text}\" (no match) "
                    f"[avg_rms={avg_rms:.0f}, peak={max_rms:.0f}]"
                )

        except Exception as e:
            logger.error(f"Wake word transcription error: {e}")

        return False

    def reset(self):
        pass