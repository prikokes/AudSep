#
# Created by Gosha Ivanov on 08.02.2025.
#

import sys
from PyQt5.QtWidgets import (QMainWindow, QWidget, QLabel, QPushButton,
                             QComboBox, QVBoxLayout, QHBoxLayout, QFrame, QProgressBar,
                             QFileDialog, QDialog, QScrollArea, QMessageBox, QApplication)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QSize
from PyQt5.QtGui import QFont, QIcon, QColor, QPalette
import torch
import torchaudio
import threading
import yaml
import os

from pathlib import Path

from ml_collections import ConfigDict

from model_loaders import htdemucs_loader
from omegaconf import OmegaConf
from utils.demix_track_demucs import demix_track_demucs

from .audio_player import (AudioPlayer)

from model_loaders.mel_band_roformer_loader import MelBandRoformerLoader
from utils.demix_track import demix_track
from model_loaders.bs_roformer_loader import BSRoformerLoader


STYLE = """
QMainWindow, QDialog {
    background-color: #1E1E1E;
    color: #FFFFFF;
}

QLabel {
    color: #FFFFFF;
    font-size: 14px;
    font-weight: normal;
}

QPushButton {
    background-color: #007BFF;
    color: white;
    border-radius: 6px;
    padding: 10px 15px;
    font-size: 14px;
    font-weight: bold;
    min-height: 40px;
}

QPushButton:hover {
    background-color: #0069D9;
}

QPushButton:pressed {
    background-color: #0062CC;
}

QPushButton:disabled {
    background-color: #6C757D;
    color: #C0C0C0;
}

QComboBox {
    background-color: #343A40;
    color: white;
    padding: 10px;
    border-radius: 6px;
    min-height: 40px;
    font-size: 14px;
}

QComboBox::drop-down {
    border: none;
    width: 30px;
}

QComboBox::down-arrow {
    image: url('data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="white" viewBox="0 0 16 16"><path d="M8 9.5l4-4 .7.7L8 10.9 3.3 6.2l.7-.7z"/></svg>');
    width: 16px;
    height: 16px;
}

QComboBox QAbstractItemView {
    background-color: #343A40;
    color: white;
    selection-background-color: #007BFF;
    selection-color: white;
    border-radius: 6px;
}

QProgressBar {
    border: none;
    background-color: #343A40;
    border-radius: 6px;
    text-align: center;
    color: white;
    font-weight: bold;
    min-height: 25px;
}

QProgressBar::chunk {
    background-color: #007BFF;
    border-radius: 6px;
}

QScrollArea, QScrollBar {
    background-color: #1E1E1E;
    border: none;
}

QScrollBar:vertical {
    width: 12px;
    margin: 0px;
}

QScrollBar::handle:vertical {
    background-color: #5C5C5C;
    min-height: 20px;
    border-radius: 6px;
}

QScrollBar::handle:vertical:hover {
    background-color: #007BFF;
}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0px;
}

QFrame#modelFrame, QFrame#infoFrame {
    background-color: #2D2D30;
    border-radius: 10px;
    padding: 5px;
}

QWidget#centralWidget {
    background-color: #1E1E1E;
}
"""


class ProcessingThread(QThread):
    update_status = pyqtSignal(str)
    update_progress = pyqtSignal(int)
    processing_finished = pyqtSignal(dict)
    processing_error = pyqtSignal(str)

    def __init__(self, audio_file, model_info, device):
        super().__init__()
        self.audio_file = audio_file
        self.model_info = model_info
        self.device = device

    def run(self):
        try:
            self.update_status.emit("Загрузка аудио...")
            self.update_progress.emit(10)
            mix, sample_rate = torchaudio.load(self.audio_file)

            self.update_status.emit(f"Обработка с помощью {self.model_info['description']}...")
            self.update_progress.emit(30)

            device = self.model_info["device"]
            processor = self.model_info["processor"]

            if str(device) == "mps":
                torch.mps.empty_cache()

            mix = mix.to(device)
            separated_tracks = processor(mix, sample_rate, device, self.model_info)

            self.update_progress.emit(100)
            self.update_status.emit("Готово!")
            self.processing_finished.emit(separated_tracks)
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.processing_error.emit(str(e))


class ModelsInfoDialog(QDialog):
    def __init__(self, parent, available_models):
        super().__init__(parent)
        self.setWindowTitle("Информация о моделях")
        self.setMinimumSize(700, 500)
        self.setStyleSheet(STYLE)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(30, 30, 30, 30)
        layout.setSpacing(20)

        title_label = QLabel("Доступные модели для разделения аудио:")
        title_label.setFont(QFont("Arial", 16, QFont.Bold))
        title_label.setStyleSheet("color: #007BFF; margin-bottom: 15px;")
        layout.addWidget(title_label)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QFrame.NoFrame)
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)
        scroll_layout.setSpacing(20)

        for model_name, model_info in available_models.items():
            model_frame = QFrame()
            model_frame.setObjectName("infoFrame")
            model_frame.setStyleSheet("QFrame#infoFrame { padding: 20px; margin: 10px 0; }")
            model_layout = QVBoxLayout(model_frame)
            model_layout.setSpacing(15)

            name_label = QLabel(model_name)
            name_label.setFont(QFont("Arial", 14, QFont.Bold))
            name_label.setStyleSheet("color: #17A2B8;")
            model_layout.addWidget(name_label)

            desc_label = QLabel(model_info.get("description", "Нет описания"))
            desc_label.setWordWrap(True)
            desc_label.setStyleSheet("font-size: 13px; line-height: 1.5;")
            model_layout.addWidget(desc_label)

            scroll_layout.addWidget(model_frame)

        scroll_area.setWidget(scroll_content)
        layout.addWidget(scroll_area)

        close_button = QPushButton("Закрыть")
        close_button.setStyleSheet("background-color: #17A2B8;")
        close_button.clicked.connect(self.accept)
        layout.addWidget(close_button)


class AudioSeparatorApp(QMainWindow):
    def __init__(self):
        super().__init__()
        os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
        os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'
        os.environ['PYTORCH_MPS_LOW_WATERMARK_RATIO'] = '0.0'
        self.setWindowTitle("AudSep")
        self.setMinimumSize(1000, 700)
        self.setStyleSheet(STYLE)

        central_widget = QWidget()
        central_widget.setObjectName("centralWidget")
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(40, 40, 40, 40)
        main_layout.setSpacing(25)

        self.is_processing = False

        app_title = QLabel("AudSep")
        app_title.setFont(QFont("Arial", 24, QFont.Bold))
        app_title.setStyleSheet("color: #007BFF; margin-bottom: 20px; text-align: center;")
        app_title.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(app_title)

        model_frame = QFrame()
        model_frame.setObjectName("modelFrame")
        model_layout = QHBoxLayout(model_frame)
        model_layout.setContentsMargins(20, 20, 20, 20)

        model_label = QLabel("Выбери модель:")
        model_label.setFont(QFont("Arial", 14))
        model_layout.addWidget(model_label)

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif hasattr(torch, 'mps') and hasattr(torch.mps, 'is_available') and torch.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        print(self.device)

        self.device = 'cpu'
        print(f"Доступная память MPS: {torch.mps.current_allocated_memory() / (1024 ** 3):.2f}GB")

        self.available_models = {
            "HTDemucs (6 стемов)": {
                "loader": htdemucs_loader.HTDemucsLoader(),
                "config": "./configs/config_htdemucs_6stems.yaml",
                "model_id": "6s",
                "processor": self._process_htdemucs,
                "description": "HTDemucs - базовая модель разделения аудио на отдельные компоненты.",
                "device": "cpu"
            },
            "MelBand RoFormer": {
                "loader": MelBandRoformerLoader,
                "config": "./configs/config_vocals_mel_band_roformer_kj.yaml",
                "model_id": "base",
                "processor": self._process_melband_roformer,
                "description": "MelBandRoformer - модель для выделения вокала из аудио.",
                "device": self.device
            },
            "BS RoFormer": {
                "loader": BSRoformerLoader,
                "config": "./configs/config_bs_roformer.yaml",
                "model_id": "bs",
                "processor": self._process_bs_roformer,
                "description": "Band-Split RoFormer модель для более точного разделения аудио на отдельные компоненты.",
                "device": self.device
            }
        }

        self.model_dropdown = QComboBox()
        self.model_dropdown.addItems(list(self.available_models.keys()))
        model_layout.addWidget(self.model_dropdown, 1)

        models_info_button = QPushButton("?")
        models_info_button.setFixedSize(QSize(40, 40))
        models_info_button.setFont(QFont("Arial", 14, QFont.Bold))
        models_info_button.setStyleSheet("border-radius: 20px;")
        models_info_button.clicked.connect(self.show_models_info)
        model_layout.addWidget(models_info_button)

        main_layout.addWidget(model_frame)

        self.select_button = QPushButton("Выбрать аудио файл")
        self.select_button.setIcon(QIcon.fromTheme("document-open"))
        self.select_button.setIconSize(QSize(24, 24))
        self.select_button.setFont(QFont("Arial", 14))
        self.select_button.clicked.connect(self.select_file)
        main_layout.addWidget(self.select_button)

        self.file_label = QLabel("Файл не выбран")
        self.file_label.setStyleSheet("background-color: #2D2D30; padding: 15px; border-radius: 6px; font-size: 14px;")
        self.file_label.setWordWrap(True)
        main_layout.addWidget(self.file_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("%p%")
        self.progress_bar.setFont(QFont("Arial", 12))
        main_layout.addWidget(self.progress_bar)

        self.process_button = QPushButton("Разделить")
        self.process_button.setIcon(QIcon.fromTheme("media-playback-start"))
        self.process_button.setIconSize(QSize(24, 24))
        self.process_button.setFont(QFont("Arial", 14, QFont.Bold))
        self.process_button.setStyleSheet("background-color: #28A745;")
        self.process_button.clicked.connect(self.process_audio)
        self.process_button.setEnabled(False)
        main_layout.addWidget(self.process_button)

        self.status_label = QLabel("")
        self.status_label.setStyleSheet("font-size: 14px; color: #17A2B8; min-height: 30px; margin-top: 10px;")
        self.status_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(self.status_label)

        footer_label = QLabel("©️ Gosha Ivanov, 2025")
        footer_label.setStyleSheet("color: #6C757D; font-size: 12px;")
        footer_label.setAlignment(Qt.AlignRight | Qt.AlignBottom)
        main_layout.addWidget(footer_label)

        self.selected_file = None
        self.separated_tracks = {}

    def select_file(self):
        if self.is_processing:
            return

        try:
            file_path, _ = QFileDialog.getOpenFileName(
                self,
                "Выбрать аудио файл",
                "",
                "Audio Files (*.mp3 *.wav)"
            )
            if file_path:
                self.selected_file = Path(file_path)
                self.file_label.setText(str(self.selected_file))
                self.process_button.setEnabled(True)

                original_style = self.file_label.styleSheet()
                self.file_label.setStyleSheet(
                    "background-color: #28A745; padding: 15px; border-radius: 6px; font-size: 14px;")
                QTimer.singleShot(500, lambda: self.file_label.setStyleSheet(original_style))
        except Exception as e:
            self.status_label.setText(f"Ошибка при выборе файла: {str(e)}")
            QMessageBox.critical(self, "Ошибка", f"Не удалось выбрать файл: {str(e)}")

    def show_models_info(self):
        info_dialog = ModelsInfoDialog(self, self.available_models)
        info_dialog.exec_()

    def process_audio(self):
        if self.is_processing:
            return

        self.is_processing = True
        self.select_button.setEnabled(False)
        self.process_button.setEnabled(False)
        self.process_button.setText("Обработка...")
        self.process_button.setStyleSheet("background-color: #FFC107;")

        self.status_label.setText("Подготовка...")
        self.progress_bar.setValue(0)

        selected_model_name = self.model_dropdown.currentText()
        model_info = self.available_models[selected_model_name]

        self.processing_thread = ProcessingThread(self.selected_file, model_info, self.device)
        self.processing_thread.update_status.connect(self.update_status)
        self.processing_thread.update_progress.connect(self.progress_bar.setValue)
        self.processing_thread.processing_finished.connect(self.processing_complete)
        self.processing_thread.processing_error.connect(self.processing_error)
        self.processing_thread.start()

    def update_status(self, message):
        self.status_label.setText(message)

    def processing_complete(self, tracks):
        self.separated_tracks = tracks
        self.is_processing = False
        self.select_button.setEnabled(True)
        self.process_button.setEnabled(True)
        self.process_button.setText("Разделить")
        self.process_button.setStyleSheet("background-color: #28A745;")
        self.status_label.setText("Разделение завершено успешно!")
        self.status_label.setStyleSheet("font-size: 14px; color: #28A745; min-height: 30px; margin-top: 10px;")

        QTimer.singleShot(3000, lambda: self.status_label.setStyleSheet(
            "font-size: 14px; color: #17A2B8; min-height: 30px; margin-top: 10px;"))

        self.open_player()

    def processing_error(self, error_message):
        self.status_label.setText(f"Ошибка: {error_message}")
        self.status_label.setStyleSheet("font-size: 14px; color: #DC3545; min-height: 30px; margin-top: 10px;")
        self.is_processing = False
        self.select_button.setEnabled(True)
        self.process_button.setEnabled(True)
        self.process_button.setText("Разделить")
        self.process_button.setStyleSheet("background-color: #28A745;")

        QMessageBox.critical(self, "Ошибка обработки", f"Произошла ошибка при обработке аудио:\n{error_message}")

    def _process_htdemucs(self, mix, sample_rate, device, model_info):
        loader = model_info["loader"]

        config = OmegaConf.load(model_info["config"])

        model = loader.load(model_info["model_id"], model_info["device"], config)

        mix = mix.to(model_info["device"])
        waveform = demix_track_demucs(config, model, mix, model_info["device"], pbar=False)

        tracks = {}
        for stem in config.training.instruments:
            tracks[stem] = {'data': torch.tensor(waveform[stem]).float(), 'sr': sample_rate}

        return tracks

    def _process_melband_roformer(self, mix, sample_rate, device, model_info):
        with open(model_info['config'], 'r') as f:
            config_dict = yaml.load(f, Loader=yaml.SafeLoader)

        config = ConfigDict(config_dict)

        loader = model_info["loader"]()
        model = loader.load(model_info["model_id"], model_info["device"], config)

        mix = mix.to(model_info["device"])
        waveform = demix_track(config, model, mix, model_info["device"], pbar=False)

        tracks = {}
        for stem in waveform.keys():
            tracks[stem] = {'data': torch.tensor(waveform[stem]).float(), 'sr': sample_rate}

        return tracks

    def _process_bs_roformer(self, mix, sample_rate, device, model_info):
        with open(model_info['config'], 'r') as f:
            config_dict = yaml.load(f, Loader=yaml.SafeLoader)

        config = ConfigDict(config_dict)

        loader = model_info["loader"]()
        model = loader.load(model_info["model_id"], model_info["device"], config)

        mix = mix.to(model_info["device"])
        waveform = demix_track(config, model, mix, model_info["device"], pbar=False)

        tracks = {}
        for stem in waveform.keys():
            tracks[stem] = {'data': torch.tensor(waveform[stem]).float(), 'sr': sample_rate}

        return tracks

    def open_player(self):
        player_dialog = QDialog(self)
        player_dialog.setWindowTitle("Аудио плеер")
        player_dialog.setMinimumSize(1000, 700)

        player = AudioPlayer(player_dialog, self.separated_tracks, self.selected_file)
        player_dialog.exec_()

    def run(self):
        self.show()

