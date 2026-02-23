"""
J.A.R.V.I.S. Tools — Phase 3
===============================
Action modules that let Jarvis DO things, not just talk.

Tool 1: SystemInfo — time, date, battery, weather
Tool 2: MacControl — open apps, volume, brightness, screenshots
Tool 3: Reminder  — timers, countdowns, alarms
Tool 4: WebSearch — DuckDuckGo search, read results aloud
Tool 5: WhatsApp  — send messages via AppleScript
"""

from src.tools.router import ToolRouter

__all__ = ["ToolRouter"]