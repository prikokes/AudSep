#
# Created by Gosha Ivanov on 08.02.2025.
#

import sys

from templates.audio_separator_app import AudioSeparatorApp
from PyQt5.QtWidgets import QApplication


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = AudioSeparatorApp()
    window.run()
    sys.exit(app.exec_())
