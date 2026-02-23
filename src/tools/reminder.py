"""
J.A.R.V.I.S. Tool: Reminders & Timers
========================================
Background timers that announce via TTS when they go off.

Supported actions:
  - timer: "Set a timer for 5 minutes"
  - countdown: "Start a countdown for 2 minutes"
  - reminder: "Remind me to take a break in 30 minutes"
  - list: "What timers are active?"
  - cancel: "Cancel all timers"

How it works:
  - Each timer runs in a background daemon thread.
  - When time is up, it uses macOS `say` to announce (same as Jarvis TTS).
  - Threads are daemon threads so they die when the main app exits.
  - Multiple timers can run simultaneously.

RAM impact: ~0MB per timer (just a sleeping thread).
"""

import subprocess
import threading
import time as time_module
from datetime import datetime, timedelta

from src.utils.logger import get_logger

logger = get_logger("tools.reminder")


class ReminderTool:
    """
    Timer and reminder tool with background thread execution.
    """

    def __init__(self):
        # Track active timers: {timer_id: {name, end_time, thread}}
        self.active_timers: dict = {}
        self.timer_counter: int = 0

    def execute(self, action: str, params: dict = None) -> str:
        """
        Execute a reminder/timer action.

        Args:
            action: One of 'timer', 'countdown', 'reminder', 'list', 'cancel'.
            params: Parameters like minutes, seconds, message.

        Returns:
            Human-readable result string for TTS.
        """
        params = params or {}

        actions = {
            "timer": self._set_timer,
            "countdown": self._set_timer,
            "set_timer": self._set_timer,
            "set": self._set_timer,
            "reminder": self._set_reminder,
            "set_reminder": self._set_reminder,
            "remind": self._set_reminder,
            "list": self._list_timers,
            "list_timers": self._list_timers,
            "active": self._list_timers,
            "cancel": self._cancel_all,
            "cancel_all": self._cancel_all,
            "stop": self._cancel_all,
        }

        handler = actions.get(action)
        if handler:
            return handler(params)

        # If action is unknown, try to figure it out from params
        if "minutes" in params or "seconds" in params:
            return self._set_timer(params)

        return "I can set timers, countdowns, and reminders. Just tell me how long."

    def _set_timer(self, params: dict) -> str:
        """Set a simple timer."""
        total_seconds = self._extract_duration(params)

        if total_seconds <= 0:
            return "How long should I set the timer for?"

        if total_seconds > 7200:  # 2 hour max
            return "I can set timers up to 2 hours. How long would you like?"

        self.timer_counter += 1
        timer_id = f"timer_{self.timer_counter}"

        # Format duration for speech
        duration_text = self._format_duration(total_seconds)

        # Start background thread
        thread = threading.Thread(
            target=self._timer_thread,
            args=(timer_id, total_seconds, f"Timer complete. {duration_text} is up."),
            daemon=True,
        )
        thread.start()

        self.active_timers[timer_id] = {
            "name": f"Timer ({duration_text})",
            "end_time": datetime.now() + timedelta(seconds=total_seconds),
            "thread": thread,
        }

        logger.info(f"⏱️ Timer set: {duration_text} (id={timer_id})")
        return f"Timer set for {duration_text}. I'll let you know when it's done."

    def _set_reminder(self, params: dict) -> str:
        """Set a reminder with a custom message."""
        total_seconds = self._extract_duration(params)
        message = params.get("message", params.get("text", params.get("reminder", "")))

        if total_seconds <= 0:
            return "When should I remind you?"

        if total_seconds > 7200:
            return "I can set reminders up to 2 hours. When would you like?"

        self.timer_counter += 1
        timer_id = f"reminder_{self.timer_counter}"

        duration_text = self._format_duration(total_seconds)

        # Build announcement
        if message:
            announcement = f"Reminder: {message}"
        else:
            announcement = f"Hey, this is your reminder. {duration_text} has passed."

        thread = threading.Thread(
            target=self._timer_thread,
            args=(timer_id, total_seconds, announcement),
            daemon=True,
        )
        thread.start()

        self.active_timers[timer_id] = {
            "name": f"Reminder: {message or duration_text}",
            "end_time": datetime.now() + timedelta(seconds=total_seconds),
            "thread": thread,
        }

        logger.info(f"🔔 Reminder set: {duration_text} — \"{message}\" (id={timer_id})")

        if message:
            return f"I'll remind you to {message} in {duration_text}."
        else:
            return f"Reminder set for {duration_text}."

    def _list_timers(self, params: dict) -> str:
        """List all active timers."""
        # Clean up finished timers
        self._cleanup_finished()

        if not self.active_timers:
            return "No active timers or reminders."

        count = len(self.active_timers)
        descriptions = []
        for timer_id, info in self.active_timers.items():
            remaining = (info["end_time"] - datetime.now()).total_seconds()
            if remaining > 0:
                remaining_text = self._format_duration(int(remaining))
                descriptions.append(f"{info['name']}, {remaining_text} left")

        if not descriptions:
            return "No active timers or reminders."

        if count == 1:
            return f"You have one active timer: {descriptions[0]}."
        else:
            items = " and ".join(descriptions)
            return f"You have {count} active timers: {items}."

    def _cancel_all(self, params: dict) -> str:
        """Cancel all active timers."""
        self._cleanup_finished()

        if not self.active_timers:
            return "There are no active timers to cancel."

        count = len(self.active_timers)
        # Mark all for cancellation
        for timer_id in list(self.active_timers.keys()):
            self.active_timers[timer_id]["cancelled"] = True

        self.active_timers.clear()
        return f"Cancelled {count} timer{'s' if count > 1 else ''}."

    def _timer_thread(self, timer_id: str, seconds: int, announcement: str):
        """
        Background thread that sleeps and then announces.
        Uses small sleep intervals so cancellation is responsive.
        """
        elapsed = 0
        while elapsed < seconds:
            time_module.sleep(1)
            elapsed += 1

            # Check if cancelled
            if timer_id not in self.active_timers:
                logger.debug(f"  Timer {timer_id} cancelled")
                return
            if self.active_timers.get(timer_id, {}).get("cancelled", False):
                logger.debug(f"  Timer {timer_id} cancelled")
                return

        # Timer complete — announce via macOS TTS
        logger.info(f"⏱️ Timer {timer_id} complete! Announcing: \"{announcement}\"")

        try:
            # Play a system sound first to get attention
            subprocess.run(
                ["afplay", "/System/Library/Sounds/Glass.aiff"],
                capture_output=True,
                timeout=5,
            )
            # Then speak the announcement
            subprocess.run(
                ["say", "-v", "Daniel", "-r", "190", announcement],
                capture_output=True,
                timeout=30,
            )
        except Exception as e:
            logger.warning(f"Timer announcement failed: {e}")

        # Remove from active list
        self.active_timers.pop(timer_id, None)

    def _extract_duration(self, params: dict) -> int:
        """Extract total seconds from params. Handles minutes, seconds, hours."""
        total = 0

        # Try various param keys Phi-3 might use
        for key in ("minutes", "minute", "mins", "min", "time"):
            if key in params:
                try:
                    total += int(float(params[key])) * 60
                except (ValueError, TypeError):
                    pass

        for key in ("seconds", "second", "secs", "sec"):
            if key in params:
                try:
                    total += int(float(params[key]))
                except (ValueError, TypeError):
                    pass

        for key in ("hours", "hour", "hrs", "hr"):
            if key in params:
                try:
                    total += int(float(params[key])) * 3600
                except (ValueError, TypeError):
                    pass

        # If params has a generic "duration" or "time" in seconds
        if total == 0:
            for key in ("duration", "duration_seconds", "total_seconds"):
                if key in params:
                    try:
                        total = int(float(params[key]))
                    except (ValueError, TypeError):
                        pass

        return total

    def _format_duration(self, seconds: int) -> str:
        """Format seconds into spoken text."""
        if seconds < 60:
            return f"{seconds} second{'s' if seconds != 1 else ''}"
        elif seconds < 3600:
            minutes = seconds // 60
            remaining_secs = seconds % 60
            text = f"{minutes} minute{'s' if minutes != 1 else ''}"
            if remaining_secs > 0:
                text += f" and {remaining_secs} second{'s' if remaining_secs != 1 else ''}"
            return text
        else:
            hours = seconds // 3600
            remaining_mins = (seconds % 3600) // 60
            text = f"{hours} hour{'s' if hours != 1 else ''}"
            if remaining_mins > 0:
                text += f" and {remaining_mins} minute{'s' if remaining_mins != 1 else ''}"
            return text

    def _cleanup_finished(self):
        """Remove finished timers from active list."""
        now = datetime.now()
        finished = [
            tid for tid, info in self.active_timers.items()
            if info["end_time"] <= now
        ]
        for tid in finished:
            self.active_timers.pop(tid, None)