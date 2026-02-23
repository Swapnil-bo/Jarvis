"""
J.A.R.V.I.S. Dashboard — Server
=================================
Lightweight FastAPI server serving the dashboard HTML
and pushing real-time events via WebSocket.

Runs in a daemon thread — dies when main app exits.
RAM overhead: ~15MB.
"""

import asyncio
import json
import threading
from pathlib import Path
from typing import Set

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

from src.dashboard import events
from src.utils.logger import get_logger

logger = get_logger("dashboard")

app = FastAPI(docs_url=None, redoc_url=None)  # No swagger UI — save RAM

# Connected WebSocket clients
_clients: Set[WebSocket] = set()

# Path to the dashboard HTML
_html_path = Path(__file__).parent / "static" / "index.html"


@app.get("/")
async def dashboard():
    """Serve the dashboard HTML."""
    return HTMLResponse(_html_path.read_text(encoding="utf-8"))


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    """WebSocket: push real-time events to the dashboard."""
    await ws.accept()
    _clients.add(ws)
    logger.info(f"📊 Dashboard connected ({len(_clients)} client{'s' if len(_clients) > 1 else ''})")

    # Send current state snapshot to new connection
    try:
        await ws.send_json({"type": "snapshot", **events.get_state()})
    except Exception:
        pass

    try:
        # Keep connection alive — client sends pings
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        _clients.discard(ws)
        logger.info(f"📊 Dashboard disconnected ({len(_clients)} client{'s' if len(_clients) > 1 else ''})")


async def _broadcast_loop():
    """
    Background task: reads events from the queue and broadcasts to all clients.
    Runs in the FastAPI async event loop.
    """
    while True:
        event = events.get_event(timeout=0.05)
        if event and _clients:
            dead = set()
            for ws in _clients.copy():
                try:
                    await ws.send_json(event)
                except Exception:
                    dead.add(ws)
            _clients.difference_update(dead)
        else:
            await asyncio.sleep(0.05)


@app.on_event("startup")
async def startup():
    """Start the broadcast loop when server starts."""
    asyncio.create_task(_broadcast_loop())


def start(host: str = "127.0.0.1", port: int = 8765):
    """
    Start the dashboard server in a daemon background thread.
    Call from main.py during initialization.
    """
    import uvicorn

    def _run():
        uvicorn.run(
            app,
            host=host,
            port=port,
            log_level="warning",
            access_log=False,
        )

    thread = threading.Thread(target=_run, daemon=True, name="dashboard")
    thread.start()
    logger.info(f"📊 Dashboard: http://{host}:{port}")
    return thread