"""
J.A.R.V.I.S. Tool: Mac Control
=================================
Controls macOS using AppleScript and shell commands — zero extra dependencies.

Supported actions:
  - open_app: Open any application by name
  - close_app: Close/quit an application
  - volume_up / volume_down / volume_mute / volume_set: Volume control
  - brightness_up / brightness_down: Screen brightness
  - screenshot: Take a screenshot (saves to Desktop)
  - lock: Lock the screen
  - sleep: Put Mac to sleep

All commands use:
  - osascript (AppleScript) for app control and volume
  - brightness CLI for screen brightness (built into macOS)
  - screencapture for screenshots

RAM impact: 0MB — all commands are subprocess calls.
"""

import os
import subprocess
from datetime import datetime

from src.utils.logger import get_logger

logger = get_logger("tools.mac_control")


class MacControlTool:
    """
    macOS control tool — apps, volume, brightness, screenshots.
    """

    def execute(self, action: str, params: dict = None) -> str:
        """
        Execute a Mac control action.

        Args:
            action: The action to perform.
            params: Parameters like app name, volume level, etc.

        Returns:
            Human-readable result string for TTS.
        """
        params = params or {}

        actions = {
            "open_app": self._open_app,
            "close_app": self._close_app,
            "quit_app": self._close_app,
            "volume_up": self._volume_up,
            "volume_down": self._volume_down,
            "volume_mute": self._volume_mute,
            "volume_set": self._volume_set,
            "brightness_up": self._brightness_up,
            "brightness_down": self._brightness_down,
            "screenshot": self._screenshot,
            "lock": self._lock_screen,
            "lock_screen": self._lock_screen,
            "sleep": self._sleep,
        }

        handler = actions.get(action)
        if handler:
            return handler(params)

        return "I can open apps, control volume, adjust brightness, take screenshots, or lock the screen."

    # ── App Control ──────────────────────────────────────────

    def _open_app(self, params: dict) -> str:
        """Open an application by name."""
        app = params.get("app", params.get("application", ""))
        if not app:
            return "Which app would you like me to open?"

        # Normalize common app names
        app_aliases = {
            # Browsers
            "chrome": "Google Chrome",
            "google chrome": "Google Chrome",
            "google": "Google Chrome",
            "safari": "Safari",
            "firefox": "Firefox",
            "brave": "Brave Browser",
            # System apps
            "finder": "Finder",
            "terminal": "Terminal",
            "notes": "Notes",
            "note": "Notes",
            "calendar": "Calendar",
            "reminders": "Reminders",
            "reminder": "Reminders",
            "messages": "Messages",
            "message": "Messages",
            "imessage": "Messages",
            "facetime": "FaceTime",
            "face time": "FaceTime",
            "mail": "Mail",
            "email": "Mail",
            "photos": "Photos",
            "photo": "Photos",
            "music": "Music",
            "maps": "Maps",
            "map": "Maps",
            "calculator": "Calculator",
            "preview": "Preview",
            "activity monitor": "Activity Monitor",
            "app store": "App Store",
            "books": "Books",
            "clock": "Clock",
            "contacts": "Contacts",
            "weather": "Weather",
            # Settings
            "settings": "System Settings",
            "system preferences": "System Settings",
            "system settings": "System Settings",
            "preferences": "System Settings",
            # Dev tools
            "vscode": "Visual Studio Code",
            "vs code": "Visual Studio Code",
            "code": "Visual Studio Code",
            "xcode": "Xcode",
            # Communication
            "whatsapp": "WhatsApp",
            "whats app": "WhatsApp",
            "telegram": "Telegram",
            "discord": "Discord",
            "slack": "Slack",
            "zoom": "zoom.us",
            "teams": "Microsoft Teams",
            "microsoft teams": "Microsoft Teams",
            "skype": "Skype",
            # Entertainment
            "spotify": "Spotify",
            "netflix": "Google Chrome",
            "youtube": "Google Chrome",
            # Productivity
            "word": "Microsoft Word",
            "excel": "Microsoft Excel",
            "powerpoint": "Microsoft PowerPoint",
            "pages": "Pages",
            "numbers": "Numbers",
            "keynote": "Keynote",
        }

        app_name = app_aliases.get(app.lower().strip(), app)

        # If Phi-3 sent a properly capitalized name (like "Notes"), use it directly
        # but also check aliases for common variations
        if app_name == app and app.lower().strip() not in app_aliases:
            # Not in aliases — use as-is (Phi-3 might have sent the correct name)
            app_name = app.strip()

        try:
            subprocess.run(
                ["osascript", "-e", f'tell application "{app_name}" to activate'],
                capture_output=True,
                timeout=10,
            )
            return f"Opening {app_name}."
        except Exception as e:
            logger.warning(f"Failed to open {app_name}: {e}")
            return f"I couldn't open {app_name}. Make sure it's installed."

    def _close_app(self, params: dict) -> str:
        """Close/quit an application."""
        app = params.get("app", params.get("application", ""))
        if not app:
            return "Which app would you like me to close?"

        try:
            subprocess.run(
                ["osascript", "-e", f'tell application "{app}" to quit'],
                capture_output=True,
                timeout=10,
            )
            return f"Closing {app}."
        except Exception as e:
            logger.warning(f"Failed to close {app}: {e}")
            return f"I couldn't close {app}."

    # ── Volume Control ───────────────────────────────────────

    def _extract_volume_level(self, params: dict) -> int:
        """Extract volume level from any param key Phi-3 might use."""
        for key in ("level", "volume", "percentage", "percent", "value", "amount"):
            if key in params:
                try:
                    return int(params[key])
                except (ValueError, TypeError):
                    pass
        return None

    def _volume_up(self, params: dict) -> str:
        """Increase volume by ~15%, or set to specific level if provided."""
        level = self._extract_volume_level(params)
        if level is not None:
            return self._volume_set({"level": level})
        self._run_applescript("set volume output volume ((output volume of (get volume settings)) + 15)")
        return "Volume up."

    def _volume_down(self, params: dict) -> str:
        """Decrease volume by ~15%, or set to specific level if provided."""
        level = self._extract_volume_level(params)
        if level is not None:
            return self._volume_set({"level": level})
        self._run_applescript("set volume output volume ((output volume of (get volume settings)) - 15)")
        return "Volume down."

    def _volume_mute(self, params: dict) -> str:
        """Toggle mute."""
        self._run_applescript("set volume with output muted")
        return "Volume muted."

    def _volume_set(self, params: dict) -> str:
        """Set volume to a specific level (0-100)."""
        level = params.get("level", params.get("volume", 50))
        try:
            level = int(level)
            level = max(0, min(100, level))
        except (ValueError, TypeError):
            level = 50

        self._run_applescript(f"set volume output volume {level}")
        return f"Volume set to {level} percent."

    # ── Brightness Control ───────────────────────────────────

    def _brightness_up(self, params: dict) -> str:
        """Increase brightness using keyboard simulation."""
        try:
            # Use AppleScript to simulate brightness key press
            script = '''
            tell application "System Events"
                key code 144
                key code 144
                key code 144
            end tell
            '''
            self._run_applescript(script)
            return "Brightness increased."
        except Exception as e:
            logger.warning(f"Brightness up failed: {e}")
            return "I had trouble adjusting the brightness. You may need to grant accessibility permissions."

    def _brightness_down(self, params: dict) -> str:
        """Decrease brightness using keyboard simulation."""
        try:
            script = '''
            tell application "System Events"
                key code 145
                key code 145
                key code 145
            end tell
            '''
            self._run_applescript(script)
            return "Brightness decreased."
        except Exception as e:
            logger.warning(f"Brightness down failed: {e}")
            return "I had trouble adjusting the brightness. You may need to grant accessibility permissions."

    # ── Screenshot ───────────────────────────────────────────

    def _screenshot(self, params: dict) -> str:
        """Take a screenshot and save to Desktop."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"jarvis_screenshot_{timestamp}.png"
        filepath = f"~/Desktop/{filename}"

        try:
            subprocess.run(
                ["screencapture", "-x", os.path.expanduser(filepath)],
                capture_output=True,
                timeout=10,
            )
            return f"Screenshot saved to your Desktop as {filename}."
        except Exception as e:
            logger.warning(f"Screenshot failed: {e}")
            return "I couldn't take the screenshot."

    # ── Lock / Sleep ─────────────────────────────────────────

    def _lock_screen(self, params: dict) -> str:
        """Lock the screen."""
        try:
            # pmset displaysleepnow is the most reliable lock method on macOS
            subprocess.run(
                ["pmset", "displaysleepnow"],
                capture_output=True,
                timeout=5,
            )
            return "Locking the screen now."
        except Exception:
            # Fallback: use AppleScript keystroke
            try:
                self._run_applescript(
                    'tell application "System Events" to keystroke "q" using {command down, control down}'
                )
                return "Locking the screen."
            except Exception:
                return "I couldn't lock the screen. You may need to grant accessibility permissions."

    def _sleep(self, params: dict) -> str:
        """Put the Mac to sleep."""
        try:
            subprocess.run(
                ["osascript", "-e",
                 'tell application "System Events" to sleep'],
                capture_output=True,
                timeout=5,
            )
            return "Putting the Mac to sleep."
        except Exception:
            return "I couldn't put the Mac to sleep."

    # ── Utility ──────────────────────────────────────────────

    def _run_applescript(self, script: str) -> str:
        """Run an AppleScript command."""
        try:
            result = subprocess.run(
                ["osascript", "-e", script],
                capture_output=True,
                text=True,
                timeout=10,
            )
            return result.stdout.strip()
        except Exception as e:
            logger.warning(f"AppleScript failed: {e}")
            return ""