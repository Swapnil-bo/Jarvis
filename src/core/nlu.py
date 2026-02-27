"""
J.A.R.V.I.S. NLU (Natural Language Understanding) Engine
==========================================================
Sends user queries to Ollama running Phi-3 Mini locally.

Phi-3 Mini (3.8B params, Q4 quantized) uses ~2.3GB VRAM via Metal.
Communicates via Ollama's REST API (localhost:11434).

Identity protection (3 layers):
  Layer 0: Hardcoded shortcut for identity questions (bypasses Phi-3)
  Layer 1: Memory context rewriting + system prompt firewall
  Layer 2: Post-processing poison phrase detection

Raw mode:
  When raw=True, skips all post-processing (Layer 2).
  Used by code generation so backticks/markdown aren't stripped.
"""

import re
import requests
import time

from src.utils.config import load_config
from src.utils.logger import get_logger, log_memory

logger = get_logger("core.nlu")

# ──────────────────────────────────────────────────────────────
# Identity confusion detection
# ──────────────────────────────────────────────────────────────

IDENTITY_POISON_PHRASES = [
    # Phi-3 adopting user's career/education
    "as an engineer myself",
    "as an aspiring engineer",
    "as an aspiring ai engineer",
    "my aspirations",
    "my passion for ai",
    "as a student myself",
    "my engineering",
    "i am an aspiring",
    "quest for knowledge",
    "professional growth",
    # Phi-3 adopting user's university
    "within brainware",
    "brainware university",
    "concierge to",
    "guidance within",
    "resources on ai",
    # Phi-3 confused about identity direction
    "not directed at myself",
    "query is not directed",
    "acknowledges that he",
    "delve into your identity",
    "uncover your identity",
    "explore your identity",
    "bit of a mystery",
    "in my capacity",
    "my role as",
    "my function as",
    "vibe coding",
    "100 days of",
    "cursor ide",
    "claude code",
    "rtx 3050",
    "macbook air m1",
]

# Hardcoded clean responses for identity questions
IDENTITY_RESPONSES = {
    "who_are_you": (
        "I'm Jarvis, your personal AI assistant running locally on your Mac. "
        "How can I help you, Sonu?"
    ),
    "who_am_i": (
        "You're Swapnil Hazra, nickname Sonu. "
        "An aspiring AI engineer and student at Brainware University. "
        "Currently doing a 100 Days of Vibe Coding challenge. "
        "What can I do for you?"
    ),
}


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

    # ──────────────────────────────────────────────────────────
    # Layer 0: Identity shortcut (bypasses Phi-3 entirely)
    # ──────────────────────────────────────────────────────────

    def _check_identity_shortcut(self, user_input: str) -> str:
        """
        Intercept identity questions and return hardcoded responses.
        Phi-3-mini cannot reliably handle these — bypass it entirely.
        Returns None if not an identity question.
        """
        text_lower = user_input.lower().strip()

        # "Who are you?" / "What are you?"
        if any(q in text_lower for q in [
            "who are you", "what are you", "tell me about yourself",
            "introduce yourself", "what is your name", "what's your name",
            "are you jarvis", "are you an ai", "are you a robot",
            "what can you do", "what do you do"
        ]):
            return IDENTITY_RESPONSES["who_are_you"]

        # "Who am I?" / "What is my name?" / "Do you know me?"
        if any(q in text_lower for q in [
            "who am i", "what is my name", "what's my name",
            "do you know me", "do you know who i am",
            "tell me about me", "what do you know about me",
            "where do i study", "where do i work",
            "what is my goal", "what are my interests",
            "what do i do", "what am i doing",
            "what is my university", "what college",
            "my name", "about me"
        ]):
            return IDENTITY_RESPONSES["who_am_i"]

        return None

    # ──────────────────────────────────────────────────────────
    # Layer 1: System prompt + memory context firewall
    # ──────────────────────────────────────────────────────────

    def _build_system_prompt(self, memory_context: str = "") -> str:
        """
        Build the full system prompt with memory context.
        Rewrites memory facts to explicitly attribute them to the user.
        """
        if not memory_context:
            return self.system_prompt

        # Rewrite every memory line to explicitly say "The user"
        safe_lines = []
        for line in memory_context.strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            if line.startswith("=") or line.startswith("─") or line.startswith("---"):
                continue
            line_lower = line.lower()
            if not any(line_lower.startswith(p) for p in [
                "the user", "sonu", "swapnil", "user's", "- the user"
            ]):
                line = f"The user: {line}"
            safe_lines.append(f"  - {line}")

        safe_context = "\n".join(safe_lines)

        return (
            f"{self.system_prompt}\n\n"
            f"ABOUT YOUR USER (these facts describe the human you serve, NOT you):\n"
            f"{safe_context}\n\n"
            f"CRITICAL RULE: You are Jarvis, software. The above describes your user Sonu. "
            f"Never say 'I am an engineer' or 'my university' — those are about your USER."
        )

    # ──────────────────────────────────────────────────────────
    # Layer 2: Output post-processing
    # ──────────────────────────────────────────────────────────

    def _clean_response(self, raw_response: str) -> str:
        """
        Clean up Phi-3's response:
        1. Remove hallucinated instructions/markdown
        2. Detect and replace identity-confused responses
        3. Strip quotes and prefixes
        """
        if not raw_response:
            return ""

        # --- Hallucination cleanup ---
        for poison in ["---", "###", "REFERENCE:", "Instruction:", "```",
                        "Note:", "Disclaimer:", "As an AI language"]:
            if poison in raw_response:
                raw_response = raw_response[:raw_response.index(poison)].strip()

        # --- Identity confusion detection ---
        response_lower = raw_response.lower()
        for phrase in IDENTITY_POISON_PHRASES:
            if phrase in response_lower:
                logger.warning(
                    f"⚠️ Identity confusion detected: '{phrase}' — replacing response"
                )
                return IDENTITY_RESPONSES["who_am_i"]

        # --- Remove wrapping quotes ---
        if raw_response.startswith('"') and raw_response.endswith('"'):
            raw_response = raw_response[1:-1]
        if raw_response.startswith("'") and raw_response.endswith("'"):
            raw_response = raw_response[1:-1]

        # --- Remove filler prefixes ---
        for prefix in ["As Jarvis, ", "As your assistant, ", "Well, as Jarvis, ",
                        "As your AI assistant, ", "As an AI, ", "Certainly! ",
                        "Certainly, ", "Of course! "]:
            if raw_response.startswith(prefix):
                raw_response = raw_response[len(prefix):]
                if raw_response:
                    raw_response = raw_response[0].upper() + raw_response[1:]

        return raw_response.strip()

    # ──────────────────────────────────────────────────────────
    # Ollama API call
    # ──────────────────────────────────────────────────────────

    def _call_ollama(self, prompt: str, model: str, memory_context: str = "", override_tokens: int = None) -> str:
        """Make a request to Ollama's generate API. Returns raw response."""
        full_system = self._build_system_prompt(memory_context)

        # Use override if provided (for code gen), otherwise use config default
        predict_limit = override_tokens if override_tokens is not None else self.max_tokens

        payload = {
            "model": model,
            "prompt": prompt,
            "system": full_system,
            "stream": False,
            "options": {
                "num_ctx": self.context_window,
                "temperature": self.temperature,
                "num_predict": predict_limit,
            },
        }

        for attempt in range(2):
            try:
                response = requests.post(
                    f"{self.base_url}/api/generate",
                    json=payload,
                    timeout=60,
                )
                response.raise_for_status()

                data = response.json()
                return data.get("response", "").strip()

            except Exception as e:
                if attempt == 0:
                    logger.warning(f"Ollama request failed: {e}. Retrying in 3s...")
                    time.sleep(3)
                else:
                    raise e

    # ──────────────────────────────────────────────────────────
    # Main entry point
    # ──────────────────────────────────────────────────────────

    def think(self, user_input: str, memory_context: str = "", raw: bool = False) -> str:
        """
        Process user input and return Jarvis's response.

        Args:
            user_input: The user's query or prompt.
            memory_context: Optional memory context string.
            raw: If True, skip all post-processing (Layer 2).
                 Used for code generation where backticks/markdown
                 must be preserved.

        Pipeline:
            0. Identity shortcut (hardcoded, instant) — skipped if raw
            1. Phi-3 Mini (primary LLM)
            2. Hardcoded error (last resort)
        """
        if not user_input:
            return "I didn't catch that. Could you repeat?"

        logger.info(f"🧠 Thinking about: \"{user_input[:200]}\"")

        # Layer 0: Identity shortcut — skip in raw mode
        if not raw:
            identity_response = self._check_identity_shortcut(user_input)
            if identity_response:
                logger.info(f"💬 Identity shortcut: \"{identity_response[:60]}...\"")
                return identity_response

        if memory_context:
            logger.debug(f"  Memory context injected ({len(memory_context)} chars)")

        if not self._check_ollama_running():
            logger.error(
                "Ollama is not running! Start it with: open -a Ollama "
                "(or launch from Applications)"
            )
            return (
                "I can't reach my brain right now. "
                "Please make sure Ollama is running."
            )

        # Layer 1+2: Phi-3 with firewall + post-processing
        try:
            logger.info(f"Using primary model: {self.model}")
            
            # If raw mode (code gen), give J.A.R.V.I.S. a massive 1024 token limit.
            # Otherwise, use the standard chat limit.
            token_limit = 1024 if raw else self.max_tokens
            
            response = self._call_ollama(user_input, self.model, memory_context, override_tokens=token_limit)

            if response:
                # Apply cleaning only in normal mode, skip for code gen
                if not raw:
                    response = self._clean_response(response)

                if response:
                    logger.info(f"💬 Response: \"{response[:100]}...\"")
                    log_memory(logger)
                    return response

        except Exception as e:
            logger.warning(f"Primary model failed: {e}")

        return "I'm having trouble thinking right now. Please try again."