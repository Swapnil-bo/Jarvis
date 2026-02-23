"""
J.A.R.V.I.S. Tool: WhatsApp Messaging (v6)
=============================================
Sends WhatsApp messages using cliclick for mouse clicks.

Key discovery: WhatsApp Desktop (Electron) reports 0 windows to macOS,
so we can't get window bounds. Instead we use fixed screen coordinates
determined by manual testing with `cliclick p`.

Screen: 1440x900 (MacBook Air M1)
Message input field: (647, 670) — confirmed by manual cursor placement.

Install: brew install cliclick
"""

import subprocess
import time

from src.utils.logger import get_logger

logger = get_logger("tools.whatsapp")

# Screen coordinates for WhatsApp message input field.
# Found by: hover mouse over "Type a message" box → run `cliclick p`
# Adjust these if your WhatsApp window is in a different position.
MSG_INPUT_X = 869
MSG_INPUT_Y = 877


class WhatsAppTool:

    def execute(self, action: str, params: dict = None) -> str:
        params = params or {}
        actions = {
            "send": self._send_message,
            "send_message": self._send_message,
            "message": self._send_message,
            "text": self._send_message,
        }
        handler = actions.get(action)
        if handler:
            return handler(params)
        return "I can send WhatsApp messages. Just tell me who to message and what to say."

    def _send_message(self, params: dict) -> str:
        # Extract contact
        contact = ""
        for key in ("contact", "to", "recipient", "name", "person"):
            if key in params:
                contact = str(params[key]).strip()
                if contact:
                    break

        # Extract message
        message = ""
        for key in ("message", "text", "content", "body", "msg"):
            if key in params:
                message = str(params[key]).strip()
                if message:
                    break

        if not contact:
            return "Who should I send the message to?"
        if not message:
            return f"What should I say to {contact}?"

        logger.info(f"📱 WhatsApp: Sending to '{contact}': \"{message}\"")

        try:
            # Step 1: Activate WhatsApp and bring to front
            self._applescript('tell application "WhatsApp" to activate')
            time.sleep(2.0)

            # Step 2: Click the search bar and type contact name
            self._click(150, 69)
            time.sleep(0.5)

            # Step 4: Clear any existing search text, then type contact name
            self._applescript('''
                tell application "System Events"
                    tell process "WhatsApp"
                        keystroke "a" using command down
                    end tell
                end tell
            ''')
            time.sleep(0.2)
            self._paste(contact)
            time.sleep(2.5)  # Wait for search results to appear

            # Step 5: Click the first search result to open the chat
            # First result appears below search bar, roughly at (229, 174)
            self._click(229, 174)
            time.sleep(2.5)  # Wait for chat to fully load

            # Step 6: Click directly on message input field
            self._click(MSG_INPUT_X, MSG_INPUT_Y)
            time.sleep(0.8)

            # Step 7: Paste the message
            self._paste(message)
            time.sleep(0.5)

            # Step 8: Press Enter to send
            self._key(36)  # Enter

            logger.info(f"📱 WhatsApp message sent to {contact}")
            return f"Message sent to {contact} on WhatsApp."

        except Exception as e:
            logger.error(f"WhatsApp automation failed: {e}")
            return f"I had trouble sending the message to {contact}. Make sure WhatsApp is open and logged in."

    def _click(self, x: int, y: int):
        """Click at screen coordinates using cliclick."""
        try:
            subprocess.run(
                ["/opt/homebrew/bin/cliclick", f"c:{x},{y}"],
                capture_output=True,
                timeout=5,
            )
            logger.debug(f"  cliclick at ({x}, {y})")
        except Exception as e:
            logger.warning(f"cliclick failed: {e}")

    def _paste(self, text: str):
        """Copy text to clipboard via pbcopy, then Cmd+V."""
        process = subprocess.Popen(["pbcopy"], stdin=subprocess.PIPE)
        process.communicate(text.encode("utf-8"))
        time.sleep(0.2)

        self._applescript('''
            tell application "System Events"
                tell process "WhatsApp"
                    keystroke "v" using command down
                end tell
            end tell
        ''')
        time.sleep(0.3)

    def _key(self, key_code: int):
        """Press a key in WhatsApp."""
        self._applescript(f'''
            tell application "System Events"
                tell process "WhatsApp"
                    key code {key_code}
                end tell
            end tell
        ''')

    def _applescript(self, script: str) -> str:
        """Run an AppleScript command."""
        try:
            result = subprocess.run(
                ["osascript", "-e", script],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode != 0 and result.stderr:
                logger.warning(f"AppleScript error: {result.stderr.strip()}")
            return result.stdout.strip()
        except Exception as e:
            logger.warning(f"AppleScript failed: {e}")
            return ""