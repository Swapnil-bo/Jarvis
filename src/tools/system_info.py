"""
J.A.R.V.I.S. Tool: System Info
=================================
Provides system information using native Python — zero extra dependencies.

Supported actions:
  - time: Current time in 12-hour format
  - date: Current date with day of week
  - battery: Battery percentage and charging status
  - weather: Current weather (uses wttr.in — free, no API key)

All responses are formatted for TTS (spoken aloud, no special characters).

RAM impact: 0MB — uses only Python stdlib + psutil (already installed).
"""

import subprocess
from datetime import datetime

import psutil

from src.utils.logger import get_logger

logger = get_logger("tools.system_info")


class SystemInfoTool:
    """
    System information tool — time, date, battery, weather.
    """

    def execute(self, action: str, params: dict = None) -> str:
        """
        Execute a system info action.

        Args:
            action: One of 'time', 'date', 'battery', 'weather'.
            params: Optional parameters (e.g., city for weather).

        Returns:
            Human-readable string for TTS.
        """
        params = params or {}

        actions = {
            "time": self._get_time,
            "date": self._get_date,
            "day": self._get_date,
            "battery": self._get_battery,
            "weather": self._get_weather,
        }

        handler = actions.get(action)
        if handler:
            return handler(params)

        return f"I can tell you the time, date, battery level, or weather. What would you like?"

    def _get_time(self, params: dict) -> str:
        """Current time in 12-hour spoken format."""
        now = datetime.now()
        # Format: "3:45 PM" spoken as "It's 3 45 PM"
        hour = now.strftime("%I").lstrip("0")  # Remove leading zero
        minute = now.strftime("%M")
        period = now.strftime("%p")

        if minute == "00":
            return f"It's {hour} {period} exactly."
        else:
            return f"It's {hour} {minute} {period}."

    def _get_date(self, params: dict) -> str:
        """Current date with day of week."""
        now = datetime.now()
        day_name = now.strftime("%A")          # "Saturday"
        month = now.strftime("%B")              # "February"
        day_num = now.day                       # 22 (no leading zero)
        year = now.year

        # Add ordinal suffix
        if 11 <= day_num <= 13:
            suffix = "th"
        else:
            suffix = {1: "st", 2: "nd", 3: "rd"}.get(day_num % 10, "th")

        return f"Today is {day_name}, {month} {day_num}{suffix}, {year}."

    def _get_battery(self, params: dict) -> str:
        """Battery percentage and charging status."""
        battery = psutil.sensors_battery()

        if battery is None:
            return "I couldn't read the battery status."

        percent = int(battery.percent)
        plugged = battery.power_plugged

        if plugged:
            if percent >= 100:
                return f"Battery is fully charged at {percent} percent."
            else:
                return f"Battery is at {percent} percent and charging."
        else:
            if percent <= 10:
                return f"Battery is critically low at {percent} percent. You should plug in soon."
            elif percent <= 20:
                return f"Battery is at {percent} percent. Getting a bit low."
            else:
                return f"Battery is at {percent} percent, running on battery power."

    def _get_weather(self, params: dict) -> str:
        """
        Current weather using wttr.in (free, no API key, no signup).
        Falls back to a helpful message if no internet.
        """
        city = params.get("city", params.get("location", ""))

        try:
            # wttr.in returns plain text weather with ?format parameter
            # %C = condition, %t = temperature, %h = humidity, %w = wind
            url = f"https://wttr.in/{city}?format=%C+%t"

            result = subprocess.run(
                ["curl", "-s", "--max-time", "5", url],
                capture_output=True,
                text=True,
                timeout=8,
            )

            if result.returncode == 0 and result.stdout.strip():
                weather = result.stdout.strip()
                # Clean up the response for TTS
                weather = weather.replace("+", " ").replace("°C", " degrees celsius")
                weather = weather.replace("°F", " degrees fahrenheit")

                if city:
                    return f"Weather in {city}: {weather}."
                else:
                    return f"Current weather: {weather}."
            else:
                return "I couldn't fetch the weather. Check your internet connection."

        except Exception as e:
            logger.warning(f"Weather fetch failed: {e}")
            return "I'm having trouble getting the weather right now."