#
# Created by Gosha Ivanov on 08.02.2025.
#
import utils.file_patch

import sys

from pathlib import Path

from templates.audio_separator_app import AudioSeparatorApp
from PyQt5.QtWidgets import QApplication

import multiprocessing

from utils.user_data import get_user_data_dir, get_weights_dir


def initialize_app_directories():
    weights_dir = get_weights_dir()
    print(f"User weights directory: {weights_dir}")

    user_data_dir = Path(get_user_data_dir())
    (user_data_dir / "output").mkdir(exist_ok=True)
    (user_data_dir / "input").mkdir(exist_ok=True)


if __name__ == '__main__':
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        # Если уже установлен, игнорируем
        pass

    initialize_app_directories()
    app = QApplication(sys.argv)
    window = AudioSeparatorApp()
    window.run()
    sys.exit(app.exec_())
