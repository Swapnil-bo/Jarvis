"""
J.A.R.V.I.S. Phase 6.1 — Code Executor
========================================
Takes Python code as a string, saves to workspace/temp_script.py,
runs it via subprocess with a strict timeout, returns stdout or stderr.

Safety:
  - Runs in a dedicated workspace/ directory
  - Strict timeout (default 30s) prevents freezes
  - Captures both stdout and stderr
  - Auto-cleans temp script after execution

Usage:
  executor = CodeExecutor()
  result = executor.execute("run", {"code": "print('hello')"})
"""

import os
import subprocess
import sys
import time
import logging

logger = logging.getLogger(__name__)

# ── Config ──────────────────────────────────────────────────

WORKSPACE_DIR = os.path.join(os.path.dirname(__file__), "..", "workspace")
SCRIPT_NAME = "temp_script.py"
DEFAULT_TIMEOUT = 30  # seconds


class CodeExecutor:
    """
    Executes Python code strings in a local workspace directory.

    Actions:
      - "run": save code to file, execute, return output
    """

    def __init__(self, config: dict = None):
        config = config or {}
        self.timeout = config.get("timeout", DEFAULT_TIMEOUT)
        self.workspace = os.path.abspath(
            config.get("workspace_dir", WORKSPACE_DIR)
        )
        self.script_path = os.path.join(self.workspace, SCRIPT_NAME)
        self.python = sys.executable  # same Python as Jarvis (venv-aware)

        # Create workspace if missing
        os.makedirs(self.workspace, exist_ok=True)

        logger.info(
            f"⚙️ CodeExecutor initialized — workspace: {self.workspace}, "
            f"timeout: {self.timeout}s, python: {self.python}"
        )

    def execute(self, action: str, params: dict) -> str:
        """Called by the tool router."""
        if action == "run":
            code = params.get("code", "")
            if not code.strip():
                return "No code provided."
            return self._run_code(code)
        else:
            return f"Unknown code_executor action: {action}"

    def _run_code(self, code: str) -> str:
        """
        Save code → run → capture output → clean up.

        Returns:
          On success: stdout (or "Code ran successfully with no output.")
          On error: stderr with exit code
          On timeout: timeout message
        """
        # ── 1. Save to workspace ──
        try:
            with open(self.script_path, "w") as f:
                f.write(code)
            logger.info(f"📝 Saved {len(code)} chars to {self.script_path}")
        except OSError as e:
            return f"Failed to save script: {e}"

        # ── 2. Execute with timeout ──
        start = time.time()
        try:
            result = subprocess.run(
                [self.python, self.script_path],
                capture_output=True,
                text=True,
                timeout=self.timeout,
                cwd=self.workspace,
            )
            elapsed = time.time() - start
            logger.info(
                f"▶️ Script finished in {elapsed:.1f}s — "
                f"exit code: {result.returncode}"
            )

            # ── 3. Return output ──
            if result.returncode == 0:
                stdout = result.stdout.strip()
                if stdout:
                    return stdout
                return "Code ran successfully with no output."
            else:
                stderr = result.stderr.strip()
                return f"Error (exit code {result.returncode}):\n{stderr}"

        except subprocess.TimeoutExpired:
            elapsed = time.time() - start
            logger.warning(f"⏰ Script timed out after {elapsed:.1f}s")
            return (
                f"Script timed out after {self.timeout} seconds. "
                f"It may contain an infinite loop or heavy computation."
            )
        except Exception as e:
            return f"Execution failed: {e}"
        finally:
            self._cleanup()

    def _cleanup(self):
        """Remove temp script after execution."""
        try:
            if os.path.exists(self.script_path):
                os.remove(self.script_path)
                logger.debug(f"🧹 Cleaned up {self.script_path}")
        except OSError:
            pass