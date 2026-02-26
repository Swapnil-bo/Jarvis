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

import requests
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

        # Upgrade 1: TTS "Oh" Fix for single-digit minutes
        if minute == "00":
            return f"It's {hour} {period} exactly."
        elif minute.startswith("0"):
            # Converts "05" to "oh 5" for better TTS flow
            return f"It's {hour} oh {minute[1]} {period}."
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
        Current weather using wttr.in.
        Falls back to web_search if wttr.in is down or times out.
        """
        # Fix: Safely handle empty strings passed by the router
        city = params.get("city", "") or params.get("location", "")
        if not city:
            city = "Kolkata"

        try:
            url = f"https://wttr.in/{city}?format=%C+%t"
            headers = {"User-Agent": "curl/8.4.0"}

            # Give wttr.in one quick 5-second attempt
            response = requests.get(url, headers=headers, timeout=5)
            
            if response.status_code == 200 and response.text.strip():
                weather = response.text.strip()
                weather = weather.replace("+", " ").replace("°C", " degrees celsius").replace("°F", " degrees fahrenheit")
                
                if city.lower() != "kolkata":
                    return f"Weather in {city}: {weather}."
                else:
                    return f"Current weather in Kolkata: {weather}."
            else:
                logger.warning(f"wttr.in returned status {response.status_code}, falling back to web search")
                raise ValueError("wttr.in failed")

        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError, ValueError) as e:
            logger.warning(f"Weather API failed ({e}), initiating fallback web search...")
            
            # The Magic Fallback: Call the web_search tool directly
            try:
                from src.tools.web_search import WebSearchTool
                search_tool = WebSearchTool()
                # Force DuckDuckGo to avoid news articles and look for exact current temps
                query = f"current temperature and weather conditions in {city} right now -news -forecast -IMD"
                return search_tool.execute("search", {"query": query})
            except Exception as e2:
                logger.error(f"Fallback web search also failed: {e2}")
                return "I'm having trouble getting the weather right now."
        except Exception as e:
            logger.warning(f"Unexpected weather fetch error: {e}")
            return "I'm having trouble getting the weather right now."