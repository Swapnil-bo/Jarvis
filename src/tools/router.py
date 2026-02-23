"""
J.A.R.V.I.S. Tool Router — Phase 3
=====================================
The brain behind tool selection. Takes user input, asks Phi-3 to classify
whether it's a tool command or just conversation, and routes accordingly.

How it works:
  1. User says something (already transcribed by STT).
  2. Router sends it to Phi-3 with a classification prompt.
  3. Phi-3 returns a JSON-like response: tool name + action + parameters.
  4. Router calls the appropriate tool module.
  5. Tool returns a result string.
  6. Result is spoken by TTS.

If Phi-3 says it's just conversation (no tool needed), router returns None
and main.py falls back to the normal NLU chat path.

Why Phi-3 for routing (not keywords)?
  - "What's the time?" and "Tell me the current time" both work
  - "Open Chrome" and "Launch Google Chrome" both work
  - No brittle regex or keyword lists to maintain
  - Phi-3 already loaded in Ollama — zero extra RAM
"""

import json
import requests

from src.utils.config import load_config
from src.utils.logger import get_logger

logger = get_logger("tools.router")

# Classification prompt — tells Phi-3 how to identify tool calls.
# This is sent as the system prompt for the routing call ONLY.
# Kept minimal to be fast (~50 tokens output max).
ROUTER_SYSTEM_PROMPT = """You are a command classifier for a voice assistant called Jarvis.
Given the user's spoken input, determine if they want to use a TOOL or just CHAT.

Available tools:
- system_info: time, date, day, battery level
- mac_control: open/close apps, volume up/down/mute, brightness up/down, screenshot, sleep, lock screen
- reminder: set timer, set reminder, countdown
- web_search: search the internet, look up information, find something online, weather, temperature, current news, latest updates, prices, scores
- whatsapp: send a WhatsApp message to someone

Respond with ONLY a JSON object, nothing else. No markdown, no explanation.

IMPORTANT RULES:
- "Open [any app name]" is ALWAYS mac_control with action open_app, never another tool.
- Only use the whatsapp tool when the user wants to SEND a message.
- Any question about weather, temperature, news, prices, scores, or current real-world information is ALWAYS web_search.

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
User: "Set volume to 50 percent" -> {"tool": "mac_control", "action": "volume_set", "params": {"level": 50}}
User: "Set volume to 10 percent" -> {"tool": "mac_control", "action": "volume_set", "params": {"level": 10}}
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
User: "How are you doing?" -> {"tool": "none"}
User: "Tell me about yourself" -> {"tool": "none"}
User: "What can you do?" -> {"tool": "none"}"""

class ToolRouter:
    """
    Routes user input to the appropriate tool or back to chat.
    """

    def __init__(self):
        config = load_config()
        nlu_cfg = config["nlu"]

        self.base_url: str = nlu_cfg["base_url"]
        self.model: str = nlu_cfg["model"]

        # Registry of tool handlers — each tool registers itself here
        self.tools: dict = {}
        self.last_route: dict = {}

        logger.info("Tool router initialized")

    def register_tool(self, name: str, handler):
        """
        Register a tool handler.

        Args:
            name: Tool name (must match names in ROUTER_SYSTEM_PROMPT).
            handler: Object with an execute(action, params) method.
        """
        self.tools[name] = handler
        logger.info(f"  🔧 Tool registered: {name}")

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
                        "num_ctx": 1024,        # Small context — classification is simple
                        "temperature": 0.1,     # Low temp — we want deterministic routing
                        "num_predict": 100,     # Short output — just JSON
                    },
                },
                timeout=15,
            )
            response.raise_for_status()

            raw = response.json().get("response", "").strip()

            # Try to extract JSON from the response
            # Phi-3 sometimes appends garbage text after valid JSON
            
            # Remove markdown code blocks if present
            if "```" in raw:
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
                raw = raw.strip()

            # Find the first { and try to parse from there
            start = raw.find("{")
            if start < 0:
                return {"tool": "none"}

            # Try parsing increasingly longer substrings from start
            # This handles cases where Phi-3 appends garbage after valid JSON
            json_str = raw[start:]
            result = None
            
            # Find matching closing brace by counting braces
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
                    f"🔧 Router: tool={tool_name}, "
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

    def execute(self, classification: dict) -> str:
        """
        Execute the classified tool action.

        Args:
            classification: Dict from classify() with tool, action, params.

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
        Full routing pipeline: classify → execute.

        Args:
            user_text: Transcribed user speech.

        Returns:
            Tool result string, or None if it's just conversation.
        """

        # ── Keyword pre-filter (bypasses Phi-3 for obvious cases) ──
        text_lower = user_text.lower()

        # Time queries
        if any(kw in text_lower for kw in ["what time", "current time", "tell me the time", "what's the time", "what is the time", "time right now"]):
            self.last_route = {"tool": "system_info", "action": "time", "params": {}}
            return self.execute(self.last_route)

        # Date queries
        if any(kw in text_lower for kw in ["what date", "current date", "what day", "today's date", "what is today", "which day"]):
            self.last_route = {"tool": "system_info", "action": "date", "params": {}}
            return self.execute(self.last_route)

        # Battery queries
        if any(kw in text_lower for kw in ["battery", "charge level", "power left", "how much charge"]):
            self.last_route = {"tool": "system_info", "action": "battery", "params": {}}
            return self.execute(self.last_route)

        # Weather/temperature → web_search
        if any(kw in text_lower for kw in ["weather", "temperature", "how hot", "how cold", "forecast", "rain today", "humidity"]):
            self.last_route = {"tool": "web_search", "action": "search", "params": {"query": user_text}}
            return self.execute(self.last_route)

        # Current events / prices / scores → web_search
        if any(kw in text_lower for kw in ["price of", "stock price", "bitcoin", "crypto", "score", "who won", "match result", "latest news", "current news", "headlines", "trending"]):
            self.last_route = {"tool": "web_search", "action": "search", "params": {"query": user_text}}
            return self.execute(self.last_route)

        # Explicit search intent → web_search
        if any(kw in text_lower for kw in ["search for", "look up", "google", "find out", "search about"]):
            self.last_route = {"tool": "web_search", "action": "search", "params": {"query": user_text}}
            return self.execute(self.last_route)

        # Volume control
        if any(kw in text_lower for kw in ["set volume", "volume to", "turn volume", "mute", "unmute"]):
            # Let Phi-3 handle param extraction for volume level
            pass

        # App launch
        if any(kw in text_lower for kw in ["open ", "launch ", "start ", "close "]) and not any(kw in text_lower for kw in ["whatsapp message", "send a", "message to"]):
            # Let Phi-3 handle app name extraction
            pass

        # WhatsApp — must check before falling through
        if any(kw in text_lower for kw in ["send a whatsapp", "whatsapp message", "message to", "text to", "send message"]):
            # Let Phi-3 handle contact + message extraction
            pass

        # Timer/reminder
        if any(kw in text_lower for kw in ["set a timer", "set timer", "remind me", "set a reminder", "countdown", "alarm"]):
            # Let Phi-3 handle time + message extraction
            pass

        classification = self.classify(user_text)
        self.last_route = classification
        return self.execute(classification)