"""
J.A.R.V.I.S. NLU (Natural Language Understanding) Engine
==========================================================
Sends user queries to Ollama running Phi-3 Mini locally.

Phi-3 Mini (3.8B params, Q4 quantized) uses ~2.3GB VRAM via Metal.
Communicates via Ollama's REST API (localhost:11434).

Fallback chain: Phi-3 → LLaMA 3.2 3B → hardcoded error message.
"""

import re
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

        self.model: str = nlu_cfg["model"]
        self.fallback_model: str = nlu_cfg["fallback_model"]
        self.base_url: str = nlu_cfg["base_url"]
        self.context_window: int = nlu_cfg["context_window"]
        self.temperature: float = nlu_cfg["temperature"]
        self.max_tokens: int = nlu_cfg["max_tokens"]
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

    def _build_system_prompt(self, memory_context: str = "") -> str:
        """
        Build the full system prompt with memory context.
        Wraps memory facts with explicit identity markers so Phi-3
        never confuses user facts with its own identity.
        """
        if not memory_context:
            return self.system_prompt

        # Rewrite every memory line to explicitly say "The user" or "Sonu"
        safe_lines = []
        for line in memory_context.strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            # Skip section headers
            if line.startswith("=") or line.startswith("─") or line.startswith("---"):
                continue
            # Force third-person attribution if not already present
            line_lower = line.lower()
            if not any(line_lower.startswith(p) for p in [
                "the user", "sonu", "swapnil", "user's", "- the user"
            ]):
                line = f"The user: {line}"
            safe_lines.append(f"  - {line}")

        safe_context = "\n".join(safe_lines)

        return (
            f"{self.system_prompt}\n\n"
            f"ABOUT YOUR USER (these facts are about Sonu/Swapnil, NOT about you Jarvis):\n"
            f"{safe_context}\n\n"
            f"REMINDER: You are Jarvis. The above facts describe your user. "
            f"When asked 'who am I' or 'what is my name', answer about the USER using the facts above."
        )

    def _clean_response(self, raw_response: str) -> str:
        """
        Clean up Phi-3's response:
        1. Remove hallucinated instructions/markdown
        2. Fix identity confusion
        3. Trim to reasonable length
        """
        if not raw_response:
            return ""

        # --- Phi-3 hallucination cleanup ---
        for poison in ["---", "###", "REFERENCE:", "Instruction:", "```",
                        "Note:", "Disclaimer:", "As an AI language"]:
            if poison in raw_response:
                raw_response = raw_response[:raw_response.index(poison)].strip()

        # --- Identity confusion cleanup ---
        # Phi-3 sometimes adopts user facts as its own
        identity_poisons = [
            "as an engineer myself",
            "as an aspiring engineer myself",
            "as an aspiring ai engineer myself",
            "my aspirations towards",
            "my passion for ai",
            "as a student myself",
            "my engineering background",
            "part of my aspirations",
            "i am an aspiring",
            "not directed at myself",
            "the query is not directed",
            "acknowledges that he serves",
        ]
        response_lower = raw_response.lower()
        for phrase in identity_poisons:
            if phrase in response_lower:
                # Nuclear option: replace the whole response
                raw_response = (
                    "You're Sonu, also known as Swapnil Hazra. "
                    "You're an aspiring AI engineer and a student at Brainware University. "
                    "How can I help you today?"
                )
                break

        # --- Remove quotes wrapping the response ---
        if raw_response.startswith('"') and raw_response.endswith('"'):
            raw_response = raw_response[1:-1]
        if raw_response.startswith("'") and raw_response.endswith("'"):
            raw_response = raw_response[1:-1]

        # --- Remove "As Jarvis, " prefix ---
        for prefix in ["As Jarvis, ", "As your assistant, ", "Well, as Jarvis, "]:
            if raw_response.startswith(prefix):
                raw_response = raw_response[len(prefix):]
                # Capitalize first letter
                if raw_response:
                    raw_response = raw_response[0].upper() + raw_response[1:]

        return raw_response.strip()

    def _call_ollama(self, prompt: str, model: str, memory_context: str = "") -> str:
        """
        Make a request to Ollama's generate API.

        Args:
            prompt: The user's transcribed speech.
            model: Which Ollama model to use.
            memory_context: Optional memory context to inject into system prompt.

        Returns:
            The model's cleaned response text.
        """
        full_system = self._build_system_prompt(memory_context)

        payload = {
            "model": model,
            "prompt": prompt,
            "system": full_system,
            "stream": False,
            "options": {
                "num_ctx": self.context_window,
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
            },
        }

        response = requests.post(
            f"{self.base_url}/api/generate",
            json=payload,
            timeout=60,
        )
        response.raise_for_status()

        data = response.json()
        raw_response = data.get("response", "").strip()

        return self._clean_response(raw_response)

    def think(self, user_input: str, memory_context: str = "") -> str:
        """
        Process user input and return Jarvis's response.

        Fallback chain:
            1. Try Phi-3 Mini (primary)
            2. Try LLaMA 3.2 3B (fallback)
            3. Return hardcoded error message (last resort)
        """
        if not user_input:
            return "I didn't catch that. Could you repeat?"

        logger.info(f"🧠 Thinking about: \"{user_input}\"")

        if memory_context:
            logger.debug(f"  Memory context injected ({len(memory_context)} chars)")

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
            response = self._call_ollama(user_input, self.model, memory_context)
            if response:
                logger.info(f"💬 Response: \"{response[:100]}...\"")
                log_memory(logger)
                return response
        except Exception as e:
            logger.warning(f"Primary model failed: {e}")

        # Try fallback model
        try:
            logger.info(f"Falling back to: {self.fallback_model}")
            response = self._call_ollama(user_input, self.fallback_model, memory_context)
            if response:
                logger.info(f"💬 Fallback response: \"{response[:100]}...\"")
                log_memory(logger)
                return response
        except Exception as e:
            logger.error(f"Fallback model also failed: {e}")

        # Last resort
        return "I'm having trouble thinking right now. Please try again."