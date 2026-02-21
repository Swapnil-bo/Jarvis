"""
J.A.R.V.I.S. Logger & Memory Profiler
=======================================
Provides:
  - Rich-formatted console logging (colored, timestamped)
  - RAM usage monitoring via psutil
  - Memory threshold warnings (configured in jarvis_config.yaml)

Usage:
    from src.utils.logger import get_logger, log_memory
    logger = get_logger("core.wake_word")
    logger.info("Wake word detected!")
    log_memory(logger)   # Logs current RAM usage
"""

import logging
import os
from datetime import datetime

import psutil
from rich.logging import RichHandler

from src.utils.config import load_config


def get_logger(name: str) -> logging.Logger:
    """
    Create a logger with Rich formatting.

    Args:
        name: Logger name (e.g., "core.stt", "core.nlu"). This appears in log output.

    Returns:
        Configured logging.Logger instance.
    """
    config = load_config()
    log_level = config["system"].get("log_level", "INFO").upper()
    log_dir = config["system"].get("log_dir", "logs")

    # Ensure log directory exists
    project_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    log_path = os.path.join(project_root, log_dir)
    os.makedirs(log_path, exist_ok=True)

    logger = logging.getLogger(f"jarvis.{name}")

    # Avoid adding duplicate handlers if logger already exists
    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, log_level, logging.INFO))

    # Rich console handler (pretty colored output)
    console_handler = RichHandler(
        rich_tracebacks=True,
        markup=True,
        show_path=False,
    )
    console_handler.setLevel(getattr(logging, log_level, logging.INFO))
    console_format = logging.Formatter("%(message)s")
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)

    # File handler (plain text for later analysis)
    today = datetime.now().strftime("%Y-%m-%d")
    file_handler = logging.FileHandler(
        os.path.join(log_path, f"jarvis_{today}.log"),
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)  # Always capture everything in file
    file_format = logging.Formatter(
        "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    file_handler.setFormatter(file_format)
    logger.addHandler(file_handler)

    return logger


def get_memory_usage_mb() -> float:
    """
    Get current process RAM usage in MB.

    Returns:
        RAM usage of the current Python process in megabytes.
    """
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)


def get_system_memory_mb() -> dict:
    """
    Get system-wide memory stats.

    Returns:
        Dict with total, available, used, and percent memory usage.
    """
    mem = psutil.virtual_memory()
    return {
        "total_mb": mem.total / (1024 * 1024),
        "available_mb": mem.available / (1024 * 1024),
        "used_mb": mem.used / (1024 * 1024),
        "percent": mem.percent,
    }


def log_memory(logger: logging.Logger) -> dict:
    """
    Log current memory usage and check against configured thresholds.

    Args:
        logger: The logger instance to write to.

    Returns:
        Dict with process_mb, system stats, and whether thresholds were breached.
    """
    config = load_config()
    warning_mb = config["system"].get("memory_warning_threshold_mb", 5500)
    critical_mb = config["system"].get("memory_critical_threshold_mb", 6500)

    process_mb = get_memory_usage_mb()
    system = get_system_memory_mb()

    status = "OK"
    if system["used_mb"] > critical_mb:
        status = "CRITICAL"
        logger.error(
            f"🔴 MEMORY CRITICAL: {system['used_mb']:.0f}MB used "
            f"(threshold: {critical_mb}MB) — consider unloading models!"
        )
    elif system["used_mb"] > warning_mb:
        status = "WARNING"
        logger.warning(
            f"🟡 MEMORY WARNING: {system['used_mb']:.0f}MB used "
            f"(threshold: {warning_mb}MB)"
        )
    else:
        logger.info(
            f"🟢 RAM: Process={process_mb:.0f}MB | "
            f"System={system['used_mb']:.0f}MB/{system['total_mb']:.0f}MB "
            f"({system['percent']}%)"
        )

    return {
        "process_mb": process_mb,
        "system": system,
        "status": status,
    }