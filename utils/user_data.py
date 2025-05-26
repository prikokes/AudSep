import os
import sys
from pathlib import Path


def get_user_data_dir(app_name="AudioSeparator"):
    if sys.platform == "win32":
        base_dir = os.path.join(os.environ["LOCALAPPDATA"], app_name)
    elif sys.platform == "darwin":
        base_dir = os.path.expanduser(f"~/Library/Application Support/{app_name}")
    else:
        base_dir = os.path.expanduser(f"~/.local/share/{app_name}")

    return Path(base_dir)


def get_weights_dir():
    weights_dir = get_user_data_dir() / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    return str(weights_dir)