"""
J.A.R.V.I.S. Speech-to-Text Engine (v5 — Anti-Hallucination)
===============================================================
Upgrades in v5:
  - condition_on_previous_text=False: Prevents whisper from using its own
    previous output as context. This was THE cause of "Thank you" hallucinations.
    Each transcription now starts fresh — no accumulated errors.
  - compression_ratio_threshold=2.0: Rejects outputs that look like repetitive
    hallucinations (e.g., "eating eating eating eating").
  - no_speech_threshold=0.4: Slightly more lenient than wake word (0.5) because
    at this stage we KNOW there's real speech (wake word already confirmed it).
  - Audio arrives pre-filtered (85Hz high-pass) from the streaming AudioCapture,
    so low-frequency noise is already removed.

Model: whisper-small (~240MB) via MLX for Apple Neural Engine acceleration.
"""

import numpy as np
import mlx_whisper

from src.utils.config import load_config
from src.utils.logger import get_logger, log_memory

logger = get_logger("core.stt")


class SpeechToText:
    """
    Transcribes audio to text using mlx-whisper with optimized parameters.
    """

    def __init__(self):
        config = load_config()
        stt_cfg = config["stt"]

        self.model_size: str = stt_cfg["model_size"]  # "small"
        self.language: str = stt_cfg["language"]       # "en"

        # Allow explicit model_id override from config (for quantized models)
        self.model_id: str = stt_cfg.get(
            "model_id",
            f"mlx-community/whisper-{self.model_size}-mlx"
        )

        logger.info(f"STT engine: mlx-whisper, model={self.model_id}")
        logger.info(
            "Model will be downloaded on first use (~240MB for 'small'). "
            "Subsequent runs use the cached version."
        )

    def transcribe(self, audio: np.ndarray) -> str:
        """
        Transcribe audio to text with anti-hallucination settings.

        Args:
            audio: numpy array of int16 audio at 16kHz mono.
                   Already high-pass filtered by AudioCapture.

        Returns:
            Transcribed text string.
        """
        logger.info("🧠 Transcribing speech...")

        # Convert int16 → float32 [-1.0, 1.0]
        audio_float = audio.astype(np.float32) / 32768.0

        try:
            result = mlx_whisper.transcribe(
                audio_float,
                path_or_hf_repo=self.model_id,
                language=self.language,
                fp16=False,
                # === Anti-hallucination parameters ===
                condition_on_previous_text=False,   # Fresh context every time
                compression_ratio_threshold=2.0,    # Reject repetitive outputs
                no_speech_threshold=0.4,            # Slightly lenient (speech confirmed by wake word)
            )

            text = result.get("text", "").strip()

            if text:
                logger.info(f"📝 Transcription: \"{text}\"")
            else:
                logger.warning("Transcription returned empty text")

            log_memory(logger)
            return text

        except Exception as e:
            logger.error(f"STT error: {e}")
            return ""