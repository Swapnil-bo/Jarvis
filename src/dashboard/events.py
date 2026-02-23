"""
J.A.R.V.I.S. Dashboard — Event Bus
====================================
Thread-safe event system that bridges the synchronous voice loop
with the async WebSocket dashboard server.

Usage in main.py:
    from src.dashboard.events import bus
    bus.emit({"type": "status", "state": "listening"})
"""

import queue
import time
from typing import Optional

# Singleton event queue — thread-safe
_event_queue: queue.Queue = queue.Queue(maxsize=500)

# Last known state for new dashboard connections
_state = {
    "status": "offline",
    "uptime_start": time.time(),
    "wake_count": 0,
    "tool_count": 0,
    "exchange_count": 0,
}


def emit(event: dict):
    """
    Emit an event from the voice loop (sync).
    Called from main.py — thread-safe.
    """
    event["ts"] = time.time()

    # Track cumulative state
    if event.get("type") == "status" and event.get("state") == "wake_detected":
        _state["wake_count"] += 1
    if event.get("type") == "routing":
        _state["tool_count"] += 1
    if event.get("type") == "response":
        _state["exchange_count"] += 1

    try:
        _event_queue.put_nowait(event)
    except queue.Full:
        # Drop oldest event if queue is full
        try:
            _event_queue.get_nowait()
            _event_queue.put_nowait(event)
        except queue.Empty:
            pass


def get_event(timeout: float = 0.1) -> Optional[dict]:
    """
    Get next event (async-side). Returns None if no event available.
    """
    try:
        return _event_queue.get(timeout=timeout)
    except queue.Empty:
        return None


def get_state() -> dict:
    """Get current cumulative state for new connections."""
    return {
        **_state,
        "uptime": int(time.time() - _state["uptime_start"]),
    }