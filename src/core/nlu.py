"""
J.A.R.V.I.S. NLU (Natural Language Understanding) Engine
==========================================================
Sends user queries to Ollama running Phi-3 Mini locally.

Why Ollama + Phi-3 Mini?
  - Ollama manages model loading/unloading and Metal GPU acceleration automatically.
  - Phi-3 Mini (3.8B params, Q4 quantized) uses ~2.3GB VRAM via Metal.
  - It runs as a separate process, so if it crashes, our Python app survives.
  - We communicate via Ollama's REST API (localhost:11434).

Why context_window=2048?
  - Larger context = more VRAM. At 4096 tokens, Phi-3 can spike to ~3.5GB.
  - 2048 tokens ≈ 1500 words — more than enough for voice assistant Q&A.
  - We'll increase this in Phase 2 when we add conversation memory.

Fallback strategy:
  If Phi-3 fails (OOM, timeout), we try LLaMA 3.2 3B as a backup.
  If both fail, we return a graceful error message spoken via TTS.
"""

import requests

from src.utils.config import load_config
from src.utils.logger import get_logger, log_memory

logger = get_logger("core.nlu")


class NLUEngine:
    """
    Sends user input to Ollama and returns the AI response.
    """

    def __init__(self):
        config = load_config()
        nlu_cfg = config["nlu"]

        self.model: str = nlu_cfg["model"]                  # "phi3:mini"
        self.fallback_model: str = nlu_cfg["fallback_model"]  # "llama3.2:3b"
        self.base_url: str = nlu_cfg["base_url"]             # "http://localhost:11434"
        self.context_window: int = nlu_cfg["context_window"]  # 2048
        self.temperature: float = nlu_cfg["temperature"]      # 0.7
        self.max_tokens: int = nlu_cfg["max_tokens"]          # 256
        self.system_prompt: str = nlu_cfg["system_prompt"]

        logger.info(
            f"NLU engine: Ollama @ {self.base_url}, "
            f"model={self.model}, ctx={self.context_window}"
        )

    def _check_ollama_running(self) -> bool:
        """Check if Ollama server is reachable."""
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=3)
            return resp.status_code == 200
        except requests.ConnectionError:
            return False

    def _call_ollama(self, prompt: str, model: str) -> str:
        """
        Make a request to Ollama's generate API.

        Args:
            prompt: The user's transcribed speech.
            model: Which Ollama model to use.

        Returns:
            The model's response text.
        """
        payload = {
            "model": model,
            "prompt": prompt,
            "system": self.system_prompt,
            "stream": False,               # Get complete response at once
            "options": {
                "num_ctx": self.context_window,
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
            },
        }

        response = requests.post(
            f"{self.base_url}/api/generate",
            json=payload,
            timeout=60,  # Phi-3 on M1 can take 10-30s for longer responses
        )
        response.raise_for_status()

        data = response.json()
        return data.get("response", "").strip()

    def think(self, user_input: str) -> str:
        """
        Process user input and return J.A.R.V.I.S.'s response.

        Args:
            user_input: The transcribed text from the user.

        Returns:
            The AI assistant's response string.

        Fallback chain:
            1. Try Phi-3 Mini (primary)
            2. Try LLaMA 3.2 3B (fallback)
            3. Return hardcoded error message (last resort)
        """
        if not user_input:
            return "I didn't catch that. Could you repeat?"

        logger.info(f"🧠 Thinking about: \"{user_input}\"")

        # Check if Ollama is running
        if not self._check_ollama_running():
            logger.error(
                "Ollama is not running! Start it with: open -a Ollama "
                "(or launch from Applications)"
            )
            return (
                "I can't reach my brain right now. "
                "Please make sure Ollama is running."
            )

        # Try primary model
        try:
            logger.info(f"Using primary model: {self.model}")
            response = self._call_ollama(user_input, self.model)
            if response:
                logger.info(f"💬 Response: \"{response[:100]}...\"")
                log_memory(logger)
                return response
        except Exception as e:
            logger.warning(f"Primary model failed: {e}")

        # Try fallback model
        try:
            logger.info(f"Falling back to: {self.fallback_model}")
            response = self._call_ollama(user_input, self.fallback_model)
            if response:
                logger.info(f"💬 Fallback response: \"{response[:100]}...\"")
                log_memory(logger)
                return response
        except Exception as e:
            logger.error(f"Fallback model also failed: {e}")

        # Last resort
        return "I'm having trouble thinking right now. Please try again."