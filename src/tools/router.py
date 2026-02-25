"""
J.A.R.V.I.S. Tool Router — Phase 6
=====================================
The brain behind tool selection. Takes user input, classifies intent,
and routes to the appropriate tool.

Two-stage routing:
  Stage 1: Keyword pre-filter — catches obvious commands instantly (0ms).
  Stage 2: Phi-3 classification — handles complex/ambiguous commands (~2-5s).

If neither stage identifies a tool, returns None and main.py falls back
to the normal NLU chat path.
"""

import json
import re
import requests

from src.utils.config import load_config
from src.utils.logger import get_logger

logger = get_logger("tools.router")

# Classification prompt for Phi-3 (Stage 2 only — complex cases)
ROUTER_SYSTEM_PROMPT = """You are a command classifier for a voice assistant called Jarvis.
Given the user's spoken input, determine if they want to use a TOOL or just CHAT.

Available tools:
- system_info: time, date, day, battery level
- mac_control: open/close apps, volume up/down/mute, brightness up/down, screenshot, sleep, lock screen
- reminder: set timer, set reminder, countdown
- web_search: search the internet, look up information, weather, temperature, news, prices, scores
- whatsapp: send a WhatsApp message to someone
- vision: read screen text (ocr), describe what's on screen (describe_screen), describe webcam view (describe_webcam)
- code_executor: write and run Python scripts, automate tasks, system utilities

Respond with ONLY a JSON object, nothing else. No markdown, no explanation.

IMPORTANT RULES:
- "Open [any app name]" is ALWAYS mac_control with action open_app, never another tool.
- Only use the whatsapp tool when the user wants to SEND a message.
- Any question about weather, temperature, news, prices, scores, or current real-world information is ALWAYS web_search.
- "Read my screen" or "what text is on screen" is ALWAYS vision with action ocr.
- "What's on my screen" or "describe my screen" is ALWAYS vision with action describe_screen.
- "What do you see" (with webcam context) or "can you see me" is ALWAYS vision with action describe_webcam.
- "Write a script", "write code", "write a python", or any request to automate/check systems via programming is ALWAYS code_executor.

If it's a tool command:
{"tool": "tool_name", "action": "specific_action", "params": {"key": "value"}}

If it's just conversation:
{"tool": "none"}

Examples:
User: "What time is it?" -> {"tool": "system_info", "action": "time"}
User: "What is the current time?" -> {"tool": "system_info", "action": "time"}
User: "Tell me the time" -> {"tool": "system_info", "action": "time"}
User: "What's the current date?" -> {"tool": "system_info", "action": "date"}
User: "What day is today?" -> {"tool": "system_info", "action": "date"}
User: "What's the battery level?" -> {"tool": "system_info", "action": "battery"}
User: "How much battery is left?" -> {"tool": "system_info", "action": "battery"}
User: "Open Safari" -> {"tool": "mac_control", "action": "open_app", "params": {"app": "Safari"}}
User: "Open WhatsApp" -> {"tool": "mac_control", "action": "open_app", "params": {"app": "WhatsApp"}}
User: "Launch Spotify" -> {"tool": "mac_control", "action": "open_app", "params": {"app": "Spotify"}}
User: "Can you open Brave browser?" -> {"tool": "mac_control", "action": "open_app", "params": {"app": "Brave Browser"}}
User: "Set volume to 50 percent" -> {"tool": "mac_control", "action": "volume_set", "params": {"level": 50}}
User: "Set volume to 10 percent" -> {"tool": "mac_control", "action": "volume_set", "params": {"level": 10}}
User: "Mute the volume" -> {"tool": "mac_control", "action": "volume_mute"}
User: "Set brightness to maximum" -> {"tool": "mac_control", "action": "brightness_up"}
User: "Take a screenshot" -> {"tool": "mac_control", "action": "screenshot"}
User: "Lock the screen" -> {"tool": "mac_control", "action": "lock"}
User: "Set a timer for 5 minutes" -> {"tool": "reminder", "action": "timer", "params": {"minutes": 5}}
User: "Remind me in 10 minutes to call Mom" -> {"tool": "reminder", "action": "reminder", "params": {"minutes": 10, "message": "call Mom"}}
User: "Search for best restaurants nearby" -> {"tool": "web_search", "action": "search", "params": {"query": "best restaurants nearby"}}
User: "What's the temperature in Kolkata?" -> {"tool": "web_search", "action": "search", "params": {"query": "current temperature in Kolkata"}}
User: "How's the weather today?" -> {"tool": "web_search", "action": "search", "params": {"query": "weather today"}}
User: "What's the weather in Mumbai?" -> {"tool": "web_search", "action": "search", "params": {"query": "weather in Mumbai now"}}
User: "What's the latest news on AI?" -> {"tool": "web_search", "action": "search", "params": {"query": "latest AI news"}}
User: "What is the price of Bitcoin?" -> {"tool": "web_search", "action": "search", "params": {"query": "Bitcoin price today"}}
User: "Who won the cricket match?" -> {"tool": "web_search", "action": "search", "params": {"query": "cricket match result today"}}
User: "What is the current temperature?" -> {"tool": "web_search", "action": "search", "params": {"query": "current temperature"}}
User: "Send a WhatsApp message to Mom saying I'll be late" -> {"tool": "whatsapp", "action": "send", "params": {"contact": "Mom", "message": "I'll be late"}}
User: "Message Aditya on WhatsApp saying hello" -> {"tool": "whatsapp", "action": "send", "params": {"contact": "Aditya", "message": "hello"}}
User: "Read my screen" -> {"tool": "vision", "action": "ocr"}
User: "What text is on my screen?" -> {"tool": "vision", "action": "ocr"}
User: "What's on my screen?" -> {"tool": "vision", "action": "describe_screen"}
User: "Describe my screen" -> {"tool": "vision", "action": "describe_screen"}
User: "What app is open?" -> {"tool": "vision", "action": "describe_screen"}
User: "Can you see me?" -> {"tool": "vision", "action": "describe_webcam"}
User: "What do you see?" -> {"tool": "vision", "action": "describe_webcam"}
User: "How do I look?" -> {"tool": "vision", "action": "describe_webcam"}
User: "Write a script to sort my downloads" -> {"tool": "code_executor", "action": "run", "params": {"request": "sort my downloads"}}
User: "Check my disk usage" -> {"tool": "code_executor", "action": "run", "params": {"request": "check disk usage"}}
User: "Write a python code to add two numbers" -> {"tool": "code_executor", "action": "run", "params": {"request": "add two numbers"}}
User: "How are you doing?" -> {"tool": "none"}
User: "Tell me about yourself" -> {"tool": "none"}
User: "What can you do?" -> {"tool": "none"}"""


class ToolRouter:
    """
    Routes user input to the appropriate tool or back to chat.
    Two-stage: keyword pre-filter (instant) → Phi-3 classification (fallback).
    """

    def __init__(self):
        config = load_config()
        nlu_cfg = config["nlu"]

        self.base_url: str = nlu_cfg["base_url"]
        self.model: str = nlu_cfg["model"]

        # Registry of tool handlers
        self.tools: dict = {}
        self.last_route: dict = {}

        logger.info("Tool router initialized (keyword pre-filter + Phi-3)")

    def register_tool(self, name: str, handler):
        """
        Register a tool handler.

        Args:
            name: Tool name (must match names in ROUTER_SYSTEM_PROMPT).
            handler: Object with an execute(action, params) method.
        """
        self.tools[name] = handler
        logger.info(f"  🔧 Tool registered: {name}")

    # ══════════════════════════════════════════════════════════
    # Stage 1: Keyword Pre-Filter (instant, zero LLM calls)
    # ══════════════════════════════════════════════════════════

    def _keyword_route(self, user_text: str) -> dict:
        """
        Fast keyword-based routing for obvious commands.
        Returns a classification dict, or None if no keyword match.
        """
        text_lower = user_text.lower()

        # ── Time queries ──
        if any(kw in text_lower for kw in [
            "what time", "current time", "tell me the time", "what's the time",
            "what is the time", "time right now", "time please"
        ]):
            return {"tool": "system_info", "action": "time", "params": {}}

        # ── Date queries ──
        if any(kw in text_lower for kw in [
            "what date", "current date", "what day", "today's date",
            "what is today", "which day", "date today"
        ]):
            return {"tool": "system_info", "action": "date", "params": {}}

        # ── Battery queries ──
        if any(kw in text_lower for kw in [
            "battery", "charge level", "power left", "how much charge",
            "battery percentage", "battery life"
        ]):
            return {"tool": "system_info", "action": "battery", "params": {}}

        # ── Vision: OCR (read text from screen) ──
        if any(kw in text_lower for kw in [
            "read my screen", "read the screen", "read the text",
            "what text", "text on screen", "ocr", "extract text",
            "what does my screen say"
        ]):
            return {"tool": "vision", "action": "ocr", "params": {"question": user_text}}

        # ── Vision: Describe screen ──
        if any(kw in text_lower for kw in [
            "what's on my screen", "whats on my screen", "describe my screen",
            "look at my screen", "what do you see on my screen",
            "what app is open", "what am i looking at",
            "what am i working on", "describe screen"
        ]):
            return {"tool": "vision", "action": "describe_screen", "params": {"question": user_text}}

        # ── Vision: Webcam describe ──
        if any(kw in text_lower for kw in [
            "what do you see", "can you see me", "look at me",
            "webcam", "camera", "what am i wearing",
            "how do i look", "who do you see", "describe me"
        ]):
            return {"tool": "vision", "action": "describe_webcam", "params": {"question": user_text}}

        # ── Code writing / execution (Phase 6) ──
        if any(kw in text_lower for kw in [
            "write a script", "write a program", "write code", "run a script",
            "execute code", "python script", "write python", "code that",
            "make a script", "create a script", "run python", "write a python",
            "write me a", "automate", "sort my", "fetch api",
            "system check", "disk usage", "list files"
        ]):
            return {"tool": "code_executor", "action": "run", "params": {"request": user_text}}

        # ── Weather / temperature → web_search ──
        if any(kw in text_lower for kw in [
            "weather", "temperature", "how hot", "how cold", "forecast",
            "rain today", "humidity", "sunny", "cloudy", "wind speed"
        ]):
            return {"tool": "web_search", "action": "search", "params": {"query": user_text}}

        # ── Current events / prices / scores → web_search ──
        if any(kw in text_lower for kw in [
            "price of", "stock price", "bitcoin", "crypto", "score",
            "who won", "match result", "latest news", "current news",
            "headlines", "trending", "election", "ipl"
        ]):
            return {"tool": "web_search", "action": "search", "params": {"query": user_text}}

        # ── Explicit search intent → web_search ──
        if any(kw in text_lower for kw in [
            "search for", "look up", "google", "find out", "search about",
            "tell me about the latest", "what is happening"
        ]):
            return {"tool": "web_search", "action": "search", "params": {"query": user_text}}

        # ── WhatsApp messaging ──
        if any(kw in text_lower for kw in [
            "send a whatsapp", "whatsapp message", "message to",
            "text to", "send message to", "send a message"
        ]):
            # Phi-3 handles contact + message extraction
            return None

        # ── App launch / control ──
        if any(kw in text_lower for kw in [
            "open ", "launch ", "start "
        ]) and not any(kw in text_lower for kw in [
            "whatsapp message", "send a", "message to"
        ]):
            app = self._extract_app_name(text_lower)
            if app:
                return {"tool": "mac_control", "action": "open_app", "params": {"app": app}}

        # ── Close app ──
        if "close " in text_lower or "quit " in text_lower or "exit " in text_lower:
            app = text_lower
            for prefix in ["close the ", "close ", "quit ", "exit "]:
                if prefix in app:
                    app = app.split(prefix, 1)[1].strip()
                    break
            for suffix in [" app", " application", " please", " for me"]:
                app = app.replace(suffix, "").strip()
            app = app.title()
            if app:
                return {"tool": "mac_control", "action": "close_app", "params": {"app": app}}

        # ── Volume control ──
        if any(kw in text_lower for kw in ["volume", "mute", "unmute"]):
            if "mute" in text_lower and "unmute" not in text_lower:
                return {"tool": "mac_control", "action": "volume_mute", "params": {}}
            if "unmute" in text_lower:
                return {"tool": "mac_control", "action": "volume_set", "params": {"level": 50}}
            if "max" in text_lower or "full" in text_lower or "100" in text_lower:
                return {"tool": "mac_control", "action": "volume_set", "params": {"level": 100}}
            if "up" in text_lower or "increase" in text_lower or "raise" in text_lower or "higher" in text_lower:
                return {"tool": "mac_control", "action": "volume_up", "params": {}}
            if "down" in text_lower or "decrease" in text_lower or "lower" in text_lower or "reduce" in text_lower:
                return {"tool": "mac_control", "action": "volume_down", "params": {}}
            # Extract number
            numbers = re.findall(r'\d+', text_lower)
            if numbers:
                level = min(int(numbers[0]), 100)
                return {"tool": "mac_control", "action": "volume_set", "params": {"level": level}}
            # Default: just return volume_up
            return {"tool": "mac_control", "action": "volume_up", "params": {}}

        # ── Brightness control ──
        if "brightness" in text_lower:
            if "max" in text_lower or "full" in text_lower or "100" in text_lower:
                return {"tool": "mac_control", "action": "brightness_up", "params": {}}
            if "up" in text_lower or "increase" in text_lower or "raise" in text_lower or "higher" in text_lower:
                return {"tool": "mac_control", "action": "brightness_up", "params": {}}
            if "down" in text_lower or "decrease" in text_lower or "lower" in text_lower or "reduce" in text_lower or "dim" in text_lower:
                return {"tool": "mac_control", "action": "brightness_down", "params": {}}
            if "min" in text_lower or "low" in text_lower:
                return {"tool": "mac_control", "action": "brightness_down", "params": {}}
            return {"tool": "mac_control", "action": "brightness_up", "params": {}}

        # ── Screenshot ──
        if "screenshot" in text_lower or "screen shot" in text_lower:
            return {"tool": "mac_control", "action": "screenshot", "params": {}}

        # ── Lock screen ──
        if any(kw in text_lower for kw in ["lock screen", "lock the screen", "lock my mac"]):
            return {"tool": "mac_control", "action": "lock", "params": {}}

        # ── Timer / Reminder ──
        if any(kw in text_lower for kw in [
            "set a timer", "set timer", "remind me", "set a reminder",
            "countdown", "alarm", "timer for"
        ]):
            # Extract time and optional message
            numbers = re.findall(r'\d+', text_lower)
            minutes = int(numbers[0]) if numbers else 5

            # Check if it's a reminder with a message
            message = None
            for trigger in ["to ", "that ", "about "]:
                if trigger in text_lower:
                    parts = text_lower.split(trigger, 1)
                    if len(parts) > 1:
                        candidate = parts[1].strip()
                        # Don't treat "to 5 minutes" as a message
                        if candidate and not candidate[0].isdigit():
                            message = candidate
                            break

            if "remind" in text_lower and message:
                return {"tool": "reminder", "action": "reminder", "params": {"minutes": minutes, "message": message}}
            else:
                return {"tool": "reminder", "action": "timer", "params": {"minutes": minutes}}

        # No keyword match — fall through to Phi-3
        return None

    def _extract_app_name(self, text_lower: str) -> str:
        """
        Extract app name from 'open/launch/start' commands.
        E.g., "can you open brave browser?" → "Brave Browser"
        """
        app = text_lower
        for prefix in [
            "can you please open ", "can you open ", "could you open ",
            "please open the ", "please open ", "please launch ",
            "open the ", "open ", "launch the ", "launch ",
            "start the ", "start "
        ]:
            if prefix in app:
                app = app.split(prefix, 1)[1].strip()
                break

        # Remove trailing filler words
        for suffix in [" app", " application", " browser", " please",
                       " for me", " right now", " now"]:
            app = app.replace(suffix, "").strip()

        # Remove question marks / punctuation
        app = re.sub(r'[?!.,]', '', app).strip()

        # Title case the app name
        return app.title() if app else ""

    # ══════════════════════════════════════════════════════════
    # Stage 2: Phi-3 Classification (complex/ambiguous cases)
    # ══════════════════════════════════════════════════════════

    def classify(self, user_text: str) -> dict:
        """
        Ask Phi-3 to classify the user's intent.

        Args:
            user_text: Transcribed user speech.

        Returns:
            Dict with keys: tool, action, params.
            If tool is "none", it's just conversation.
        """
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": user_text,
                    "system": ROUTER_SYSTEM_PROMPT,
                    "stream": False,
                    "options": {
                        "num_ctx": 1024,
                        "temperature": 0.1,
                        "num_predict": 100,
                    },
                },
                timeout=15,
            )
            response.raise_for_status()

            raw = response.json().get("response", "").strip()

            # Remove markdown code blocks if present
            if "```" in raw:
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
                raw = raw.strip()

            # Find the first { and try to parse JSON
            start = raw.find("{")
            if start < 0:
                return {"tool": "none"}

            # Find matching closing brace
            json_str = raw[start:]
            result = None
            depth = 0

            for i, char in enumerate(json_str):
                if char == "{":
                    depth += 1
                elif char == "}":
                    depth -= 1
                    if depth == 0:
                        try:
                            result = json.loads(json_str[:i + 1])
                            break
                        except json.JSONDecodeError:
                            continue

            if result is None:
                return {"tool": "none"}

            tool_name = result.get("tool", "none")
            if tool_name != "none":
                logger.info(
                    f"🔧 Router (Phi-3): tool={tool_name}, "
                    f"action={result.get('action', 'unknown')}, "
                    f"params={result.get('params', {})}"
                )
            else:
                logger.debug("  Router: no tool needed (conversation)")

            return result

        except json.JSONDecodeError as e:
            logger.warning(f"Router JSON parse failed: {e} | raw: {raw[:100]}")
            return {"tool": "none"}
        except Exception as e:
            logger.warning(f"Router classification failed: {e}")
            return {"tool": "none"}

    # ══════════════════════════════════════════════════════════
    # Execute + Route
    # ══════════════════════════════════════════════════════════

    def execute(self, classification: dict) -> str:
        """
        Execute the classified tool action.

        Returns:
            Result string to be spoken by TTS.
            Returns None if tool is "none" or tool not found.
        """
        tool_name = classification.get("tool", "none")

        if tool_name == "none":
            return None

        if tool_name not in self.tools:
            logger.warning(f"Unknown tool: {tool_name}")
            return f"I don't have the {tool_name} tool available yet."

        action = classification.get("action", "unknown")
        params = classification.get("params", {})

        try:
            handler = self.tools[tool_name]
            result = handler.execute(action, params)
            logger.info(f"🔧 Tool result: {result}")
            return result
        except Exception as e:
            logger.error(f"Tool execution error: {e}")
            return "Sorry, something went wrong while running that command."

    def route(self, user_text: str) -> str:
        """
        Full routing pipeline:
          1. Try keyword pre-filter (instant)
          2. Fall back to Phi-3 classification (slower)
          3. Execute the matched tool

        Returns:
            Tool result string, or None if it's just conversation.
        """
        # Stage 1: Keyword pre-filter
        keyword_match = self._keyword_route(user_text)
        if keyword_match is not None:
            logger.info(
                f"⚡ Router (keyword): tool={keyword_match['tool']}, "
                f"action={keyword_match.get('action', '')}"
            )
            self.last_route = keyword_match
            return self.execute(keyword_match)

        # Stage 2: Phi-3 classification
        classification = self.classify(user_text)
        self.last_route = classification
        return self.execute(classification)