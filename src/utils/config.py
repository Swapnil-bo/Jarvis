"""
J.A.R.V.I.S. Configuration Loader
==================================
Loads the YAML config file and provides easy dot-notation access.
Usage:
    from src.utils.config import load_config
    cfg = load_config()
    print(cfg["nlu"]["model"])  # "phi3:mini"
"""

import os
import yaml


def load_config(config_path: str = None) -> dict:
    """
    Load the YAML configuration file.

    Args:
        config_path: Optional path to config file. Defaults to config/jarvis_config.yaml
                     relative to the project root.

    Returns:
        Dictionary containing all configuration values.

    Raises:
        FileNotFoundError: If the config file doesn't exist.
    """
    if config_path is None:
        # Walk up from this file's location to find the project root
        # This file lives at: src/utils/config.py
        # Project root is 2 levels up
        project_root = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        config_path = os.path.join(project_root, "config", "jarvis_config.yaml")

    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"Config file not found at: {config_path}\n"
            f"Make sure you're running from the project root (~/jarvis)"
        )

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    return config