#
# Created by Gosha Ivanov on 08.02.2025.
#

import sys
from PyQt5.QtWidgets import (QMainWindow, QWidget, QLabel, QPushButton,
                             QComboBox, QVBoxLayout, QHBoxLayout, QFrame, QProgressBar,
                             QFileDialog, QDialog, QScrollArea, QMessageBox, QApplication)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QSize, QUrl, QMutex, QWaitCondition
from PyQt5.QtGui import QFont, QIcon, QColor, QPalette, QDragEnterEvent, QDropEvent
import torch
import torchaudio
import threading
import yaml
import os
import mimetypes
import time
import gc
import signal
import multiprocessing
import pickle

from pathlib import Path

from ml_collections import ConfigDict

from model_loaders import htdemucs_loader
from omegaconf import OmegaConf
from utils.demix_track_demucs import demix_track_demucs

from .audio_player import (AudioPlayer)

from model_loaders.mel_band_roformer_loader import MelBandRoformerLoader
from utils.demix_track import demix_track
from utils.path_utils import get_resource_path
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

QPushButton#cancelButton {
    background-color: #DC3545;
}

QPushButton#cancelButton:hover {
    background-color: #C82333;
}

QPushButton#cancelButton:pressed {
    background-color: #BD2130;
}

QPushButton#cancelButton:disabled {
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

QFrame#dropZone {
    background-color: #2D2D30;
    border: 2px dashed #6C757D;
    border-radius: 10px;
    padding: 20px;
}

QFrame#dropZone[active="true"] {
    border: 2px dashed #007BFF;
    background-color: #003D7A;
}

QWidget#centralWidget {
    background-color: #1E1E1E;
}
"""


class ProgressReporter:
    def __init__(self, progress_queue, cancel_event):
        self.progress_queue = progress_queue
        self.cancel_event = cancel_event

    def update_progress(self, progress):
        try:
            self.progress_queue.put(("progress", min(100, max(0, int(progress)))))
        except:
            pass

    def update_status(self, status):
        try:
            self.progress_queue.put(("status", str(status)))
        except:
            pass

    def is_cancelled(self):
        return self.cancel_event.is_set()

    def emit(self, value):
        self.update_progress(value)


def processing_worker(audio_file, model_info, progress_queue, result_queue, cancel_event):
    try:
        def signal_handler(signum, frame):
            print(f"Процесс получил сигнал {signum}, завершаемся...")
            try:
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                if hasattr(torch, 'mps') and torch.backends.mps.is_available():
                    torch.mps.empty_cache()
            except:
                pass
            os._exit(0)

        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)

        progress_reporter = ProgressReporter(progress_queue, cancel_event)

        if cancel_event.is_set():
            return

        progress_reporter.update_status("Загрузка аудио...")
        # progress_reporter.update_progress(5)

        mix, sample_rate = torchaudio.load(audio_file)
        print(f"Аудио загружено: {mix.shape}")

        if cancel_event.is_set():
            return

        progress_reporter.update_status("Подготовка модели...")
        # progress_reporter.update_progress(10)

        if model_info["processor"] == "_process_htdemucs":
            if cancel_event.is_set():
                return

            progress_reporter.update_status("Загрузка модели HTDemucs...")
            # progress_reporter.update_progress(15)

            loader = htdemucs_loader.HTDemucsLoader()
            config = OmegaConf.load(get_resource_path(model_info["config"]))
            model = loader.load(model_info["model_id"], model_info["device"], config)

            if cancel_event.is_set():
                return

            progress_reporter.update_status("Обработка аудио...")
            # progress_reporter.update_progress(25)

            mix = mix.to(model_info["device"])

            if cancel_event.is_set():
                return

            waveform = demix_track_demucs(
                config,
                model,
                mix,
                model_info["device"],
                pbar=False,
                progress_bar=progress_reporter
            )

            if cancel_event.is_set():
                return

            progress_reporter.update_status("Формирование результата...")
            # progress_reporter.update_progress(95)

            tracks = {}
            for stem in config.training.instruments:
                if cancel_event.is_set():
                    return
                tracks[stem] = {'data': torch.tensor(waveform[stem]).float(), 'sr': sample_rate}

            result_queue.put(("success", tracks))

        elif model_info["processor"] == "_process_melband_roformer":
            if cancel_event.is_set():
                return

            progress_reporter.update_status("Загрузка модели MelBand RoFormer...")
            # progress_reporter.update_progress(15)

            with open(get_resource_path(model_info["config"]), 'r') as f:
                config_dict = yaml.load(f, Loader=yaml.SafeLoader)
            config = ConfigDict(config_dict)

            loader = MelBandRoformerLoader()
            model = loader.load(model_info["model_id"], model_info["device"], config)

            if cancel_event.is_set():
                return

            progress_reporter.update_status("Обработка аудио...")
            # progress_reporter.update_progress(25)

            mix = mix.to(model_info["device"])

            if cancel_event.is_set():
                return

            waveform = demix_track(
                config,
                model,
                mix,
                model_info["device"],
                pbar=False,
                progress_bar=progress_reporter
            )

            if cancel_event.is_set():
                return

            progress_reporter.update_status("Формирование результата...")
            # progress_reporter.update_progress(95)

            tracks = {}
            for stem in waveform.keys():
                if cancel_event.is_set():
                    return
                tracks[stem] = {'data': torch.tensor(waveform[stem]).float(), 'sr': sample_rate}

            result_queue.put(("success", tracks))

        elif model_info["processor"] == "_process_bs_roformer":
            if cancel_event.is_set():
                return

            progress_reporter.update_status("Загрузка модели BS RoFormer...")
            # progress_reporter.update_progress(15)

            with open(get_resource_path(model_info["config"]), 'r') as f:
                config_dict = yaml.load(f, Loader=yaml.SafeLoader)
            config = ConfigDict(config_dict)

            loader = BSRoformerLoader()
            model = loader.load(model_info["model_id"], model_info["device"], config)

            if cancel_event.is_set():
                return

            progress_reporter.update_status("Обработка аудио...")
            # progress_reporter.update_progress(25)

            mix = mix.to(model_info["device"])

            if cancel_event.is_set():
                return

            waveform = demix_track(
                config,
                model,
                mix,
                model_info["device"],
                pbar=False,
                progress_bar=progress_reporter
            )

            if cancel_event.is_set():
                return

            progress_reporter.update_status("Формирование результата...")
            # progress_reporter.update_progress(95)

            tracks = {}
            for stem in waveform.keys():
                if cancel_event.is_set():
                    return
                tracks[stem] = {'data': torch.tensor(waveform[stem]).float(), 'sr': sample_rate}

            result_queue.put(("success", tracks))

        progress_reporter.update_progress(100)
        progress_reporter.update_status("Готово!")

    except Exception as e:
        if not cancel_event.is_set():
            result_queue.put(("error", str(e)))
        else:
            print("Процесс был отменен во время исключения")


class ProcessMonitoringThread(QThread):
    update_status = pyqtSignal(str)
    update_progress = pyqtSignal(int)
    processing_finished = pyqtSignal(dict)
    processing_error = pyqtSignal(str)
    processing_cancelled = pyqtSignal()

    def __init__(self, audio_file, model_info, device):
        super().__init__()
        self.audio_file = audio_file
        self.model_info = model_info
        self.device = device
        self._is_cancelled = False
        self._process = None
        self._cancel_event = None

        self._force_kill_timer = QTimer()
        self._force_kill_timer.timeout.connect(self._force_kill_process)
        self._force_kill_timer.setSingleShot(True)

    def cancel(self):
        print("Запрос отмены - завершаем процесс")
        self._is_cancelled = True
        self.update_status.emit("Отменяется...")

        if self._cancel_event:
            self._cancel_event.set()

        self._force_kill_timer.start(2000)

        if self._process and self._process.is_alive():
            print("Отправляем сигнал SIGTERM")
            try:
                self._process.terminate()
            except:
                pass

    def _force_kill_process(self):
        if self._process and self._process.is_alive():
            print("Принудительное уничтожение процесса")
            try:
                self._process.kill()
                self._process.join(timeout=1)
            except:
                pass

        if not self._is_cancelled:
            self._is_cancelled = True
            self.processing_cancelled.emit()

    def run(self):
        try:
            progress_queue = multiprocessing.Queue()
            result_queue = multiprocessing.Queue()
            self._cancel_event = multiprocessing.Event()

            model_info_serializable = {
                "config": self.model_info["config"],
                "model_id": self.model_info["model_id"],
                "device": self.model_info["device"],
                "processor": self.model_info["processor"].__name__ if hasattr(self.model_info["processor"], '__name__') else str(self.model_info["processor"])
            }

            self._process = multiprocessing.Process(
                target=processing_worker,
                args=(str(self.audio_file), model_info_serializable, progress_queue, result_queue, self._cancel_event)
            )

            print("Запускаем процесс обработки")
            self._process.start()

            while self._process.is_alive() and not self._is_cancelled:
                try:
                    while not progress_queue.empty():
                        msg_type, data = progress_queue.get_nowait()
                        if msg_type == "status":
                            self.update_status.emit(data)
                        elif msg_type == "progress":
                            # self.update_progress.emit(data)
                            pass
                except:
                    pass

                try:
                    if not result_queue.empty():
                        result_type, data = result_queue.get_nowait()
                        if result_type == "success":
                            print("Процесс завершился успешно")
                            self._force_kill_timer.stop()
                            self.processing_finished.emit(data)
                            return
                        elif result_type == "error":
                            print(f"Ошибка в процессе: {data}")
                            self._force_kill_timer.stop()
                            self.processing_error.emit(data)
                            return
                except:
                    pass

                self.msleep(100)

            if self._is_cancelled:
                print("Обработка отменена")
                self.processing_cancelled.emit()
            else:
                if self._process.exitcode != 0:
                    self.processing_error.emit("Процесс завершился с ошибкой")
                else:
                    self.processing_cancelled.emit()

        except Exception as e:
            print(f"Ошибка в мониторинге: {e}")
            if not self._is_cancelled:
                self.processing_error.emit(str(e))
            else:
                self.processing_cancelled.emit()
        finally:
            if self._process and self._process.is_alive():
                self._process.terminate()
                self._process.join(timeout=2)
                if self._process.is_alive():
                    self._process.kill()

            self._force_kill_timer.stop()

    def stop_thread(self):
        print("Останавливаем мониторинг...")
        self.cancel()
        self.wait(5000)
        if self.isRunning():
            self.terminate()


class DropZone(QFrame):
    file_dropped = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("dropZone")
        self.setAcceptDrops(True)
        self.setMinimumHeight(150)

        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignCenter)

        self.drop_label = QLabel("🎵\n\nПеретащите аудиофайл сюда\n\nПоддерживаемые форматы: MP3, WAV, FLAC")
        self.drop_label.setAlignment(Qt.AlignCenter)
        self.drop_label.setStyleSheet("""
            QLabel {
                color: #6C757D;
                font-size: 16px;
                font-weight: normal;
                line-height: 1.5;
            }
        """)
        layout.addWidget(self.drop_label)

        self.select_file_btn = QPushButton("Или выберите файл")
        self.select_file_btn.setStyleSheet("""
            QPushButton {
                background-color: transparent;
                color: #007BFF;
                border: 2px solid #007BFF;
                border-radius: 6px;
                padding: 8px 16px;
                font-size: 13px;
                margin-top: 10px;
            }
            QPushButton:hover {
                background-color: #007BFF;
                color: white;
            }
        """)
        layout.addWidget(self.select_file_btn)

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                file_path = url.toLocalFile()
                if self.is_audio_file(file_path):
                    event.acceptProposedAction()
                    self.setProperty("active", "true")
                    self.style().unpolish(self)
                    self.style().polish(self)
                    self.drop_label.setText("🎵\n\nОтпустите файл для загрузки")
                    self.drop_label.setStyleSheet("""
                        QLabel {
                            color: #007BFF;
                            font-size: 16px;
                            font-weight: bold;
                        }
                    """)
                    return
        event.ignore()

    def dragLeaveEvent(self, event):
        self.setProperty("active", "false")
        self.style().unpolish(self)
        self.style().polish(self)
        self.drop_label.setText("🎵\n\nПеретащите аудиофайл сюда\n\nПоддерживаемые форматы: MP3, WAV, FLAC")
        self.drop_label.setStyleSheet("""
            QLabel {
                color: #6C757D;
                font-size: 16px;
                font-weight: normal;
                line-height: 1.5;
            }
        """)

    def dropEvent(self, event: QDropEvent):
        files = [url.toLocalFile() for url in event.mimeData().urls()]

        for file_path in files:
            if self.is_audio_file(file_path):
                self.file_dropped.emit(file_path)
                self.setProperty("active", "false")
                self.style().unpolish(self)
                self.style().polish(self)
                self.drop_label.setText("\n\nФайл загружен успешно!")
                self.drop_label.setStyleSheet("""
                    QLabel {
                        color: #28A745;
                        font-size: 16px;
                        font-weight: bold;
                    }
                """)
                QTimer.singleShot(2000, self.reset_drop_zone)
                break

        event.acceptProposedAction()

    def reset_drop_zone(self):
        self.drop_label.setText("🎵\n\nПеретащите аудиофайл сюда\n\nПоддерживаемые форматы: MP3, WAV, FLAC")
        self.drop_label.setStyleSheet("""
            QLabel {
                color: #6C757D;
                font-size: 16px;
                font-weight: normal;
                line-height: 1.5;
            }
        """)

    def is_audio_file(self, file_path):
        if not os.path.isfile(file_path):
            return False

        _, ext = os.path.splitext(file_path.lower())
        supported_extensions = ['.mp3', '.wav', '.flac']

        if ext in supported_extensions:
            return True

        mime_type, _ = mimetypes.guess_type(file_path)
        if mime_type and mime_type.startswith('audio/'):
            return True

        return False


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

        self.setAcceptDrops(True)

        central_widget = QWidget()
        central_widget.setObjectName("centralWidget")
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(40, 40, 40, 40)
        main_layout.setSpacing(25)

        self.is_processing = False
        self.processing_thread = None

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

        print(f"Используется устройство: {self.device}")

        if str(self.device) == "mps":
            try:
                print(f"Доступная память MPS: {torch.mps.current_allocated_memory() / (1024 ** 3):.2f}GB")
            except:
                print("Не удалось получить информацию о памяти MPS")

        self.available_models = {
            "HTDemucs (6 стемов)": {
                "loader": htdemucs_loader.HTDemucsLoader(),
                "config": "./configs/config_htdemucs_6stems.yaml",
                "model_id": "6s",
                "processor": self._process_htdemucs,
                "description": "HTDemucs - базовая модель разделения аудио на отдельные компоненты.",
                "device": 'cpu'
            },
            "MelBand RoFormer": {
                "loader": MelBandRoformerLoader,
                "config": "./configs/config_vocals_mel_band_roformer_kj.yaml",
                "model_id": "base",
                "processor": self._process_melband_roformer,
                "description": "MelBandRoformer - модель для выделения вокала из аудио.",
                "device": 'cpu'
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

        self.drop_zone = DropZone()
        self.drop_zone.file_dropped.connect(self.handle_dropped_file)
        self.drop_zone.select_file_btn.clicked.connect(self.select_file)
        main_layout.addWidget(self.drop_zone)

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

        buttons_layout = QHBoxLayout()

        self.process_button = QPushButton("Разделить")
        self.process_button.setIcon(QIcon.fromTheme("media-playback-start"))
        self.process_button.setIconSize(QSize(24, 24))
        self.process_button.setFont(QFont("Arial", 14, QFont.Bold))
        self.process_button.setStyleSheet("background-color: #28A745;")
        self.process_button.clicked.connect(self.process_audio)
        self.process_button.setEnabled(False)
        buttons_layout.addWidget(self.process_button)

        self.cancel_button = QPushButton("Отмена")
        self.cancel_button.setObjectName("cancelButton")
        self.cancel_button.setIcon(QIcon.fromTheme("process-stop"))
        self.cancel_button.setIconSize(QSize(24, 24))
        self.cancel_button.setFont(QFont("Arial", 14, QFont.Bold))
        self.cancel_button.clicked.connect(self.cancel_processing)
        self.cancel_button.setVisible(False)
        buttons_layout.addWidget(self.cancel_button)

        main_layout.addLayout(buttons_layout)

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

    def closeEvent(self, event):
        if self.processing_thread and self.processing_thread.isRunning():
            reply = QMessageBox.question(
                self,
                "Закрытие приложения",
                "Обработка еще не завершена. Вы уверены, что хотите закрыть приложение?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )

            if reply == QMessageBox.No:
                event.ignore()
                return

            print("Принудительно завершаем обработку при закрытии")
            self.processing_thread.stop_thread()

        event.accept()

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                file_path = url.toLocalFile()
                if self.is_audio_file(file_path):
                    event.acceptProposedAction()
                    return
        event.ignore()

    def dropEvent(self, event: QDropEvent):
        files = [url.toLocalFile() for url in event.mimeData().urls()]

        for file_path in files:
            if self.is_audio_file(file_path):
                self.handle_dropped_file(file_path)
                break

        event.acceptProposedAction()

    def is_audio_file(self, file_path):
        if not os.path.isfile(file_path):
            return False

        _, ext = os.path.splitext(file_path.lower())
        supported_extensions = ['.mp3', '.wav', '.flac']

        if ext in supported_extensions:
            return True

        mime_type, _ = mimetypes.guess_type(file_path)
        if mime_type and mime_type.startswith('audio/'):
            return True

        return False

    def handle_dropped_file(self, file_path):
        if self.is_processing:
            QMessageBox.warning(self, "Внимание", "Дождитесь завершения текущей обработки или отмените её")
            return

        try:
            self.selected_file = Path(file_path)
            self.file_label.setText(f"Выбран файл: {self.selected_file.name}")
            self.process_button.setEnabled(True)

            original_style = self.file_label.styleSheet()
            self.file_label.setStyleSheet(
                "background-color: #28A745; padding: 15px; border-radius: 6px; font-size: 14px; color: white;")
            QTimer.singleShot(1500, lambda: self.file_label.setStyleSheet(original_style))

            self.status_label.setText(f"Файл загружен: {self.selected_file.name}")

        except Exception as e:
            self.status_label.setText(f"Ошибка при загрузке файла: {str(e)}")
            QMessageBox.critical(self, "Ошибка", f"Не удалось загрузить файл: {str(e)}")

    def select_file(self):
        if self.is_processing:
            return

        try:
            file_path, _ = QFileDialog.getOpenFileName(
                self,
                "Выбрать аудио файл",
                "",
                "Audio Files (*.mp3 *.wav *.flac);;MP3 Files (*.mp3);;WAV Files (*.wav);;FLAC Files (*.flac)"
            )
            if file_path:
                self.handle_dropped_file(file_path)
        except Exception as e:
            self.status_label.setText(f"Ошибка при выборе файла: {str(e)}")
            QMessageBox.critical(self, "Ошибка", f"Не удалось выбрать файл: {str(e)}")

    def show_models_info(self):
        info_dialog = ModelsInfoDialog(self, self.available_models)
        info_dialog.exec_()

    def process_audio(self):
        if self.is_processing:
            return

        if not self.selected_file or not self.selected_file.exists():
            QMessageBox.warning(self, "Внимание", "Выберите аудиофайл для обработки")
            return

        self.is_processing = True
        self.drop_zone.setEnabled(False)
        self.process_button.setVisible(False)
        self.cancel_button.setVisible(True)

        self.status_label.setText("Подготовка...")
        self.progress_bar.setValue(0)

        selected_model_name = self.model_dropdown.currentText()
        model_info = self.available_models[selected_model_name]

        self.processing_thread = ProcessMonitoringThread(self.selected_file, model_info, self.device)
        self.processing_thread.update_status.connect(self.update_status)
        self.processing_thread.update_progress.connect(self.progress_bar.setValue)
        self.processing_thread.processing_finished.connect(self.processing_complete)
        self.processing_thread.processing_error.connect(self.processing_error)
        self.processing_thread.processing_cancelled.connect(self.processing_cancelled)
        self.processing_thread.finished.connect(self.thread_finished)

        self.processing_thread.start()

    def thread_finished(self):
        if self.processing_thread:
            self.processing_thread.deleteLater()
            self.processing_thread = None

    def cancel_processing(self):
        if self.processing_thread and self.processing_thread.isRunning():
            print("Немедленная отмена обработки")

            self.cancel_button.setEnabled(False)
            self.cancel_button.setText("Отменяется...")
            self.status_label.setText("Принудительная остановка...")

            self.processing_thread.cancel()

            QTimer.singleShot(1000, self.force_reset_ui)

    def force_reset_ui(self):
        if self.is_processing:
            print("Принудительный сброс UI")
            self.processing_cancelled()

    def update_status(self, message):
        self.status_label.setText(message)

    def processing_complete(self, tracks):
        self.separated_tracks = tracks
        self.reset_ui_after_processing()
        self.status_label.setText("Разделение завершено успешно!")
        self.status_label.setStyleSheet("font-size: 14px; color: #28A745; min-height: 30px; margin-top: 10px;")

        QTimer.singleShot(3000, lambda: self.status_label.setStyleSheet(
            "font-size: 14px; color: #17A2B8; min-height: 30px; margin-top: 10px;"))

        self.open_player()

    def processing_error(self, error_message):
        self.status_label.setText(f"Ошибка: {error_message}")
        self.status_label.setStyleSheet("font-size: 14px; color: #DC3545; min-height: 30px; margin-top: 10px;")
        self.reset_ui_after_processing()
        QMessageBox.critical(self, "Ошибка обработки", f"Произошла ошибка при обработке аудио:\n{error_message}")

    def processing_cancelled(self):
        self.status_label.setText("Обработка отменена пользователем")
        self.status_label.setStyleSheet("font-size: 14px; color: #FFC107; min-height: 30px; margin-top: 10px;")
        self.reset_ui_after_processing()
        self.progress_bar.setValue(0)

        QTimer.singleShot(3000, lambda: self.status_label.setStyleSheet(
            "font-size: 14px; color: #17A2B8; min-height: 30px; margin-top: 10px;"))

    def reset_ui_after_processing(self):
        self.is_processing = False
        self.drop_zone.setEnabled(True)
        self.process_button.setVisible(True)
        self.process_button.setEnabled(True)
        self.cancel_button.setVisible(False)
        self.cancel_button.setEnabled(True)
        self.cancel_button.setText("Отмена")

    def _process_htdemucs(self, mix, sample_rate, device, model_info, thread=None):
        pass

    def _process_melband_roformer(self, mix, sample_rate, device, model_info, thread=None):
        pass

    def _process_bs_roformer(self, mix, sample_rate, device, model_info, thread=None):
        pass

    def open_player(self):
        player_dialog = QDialog(self)
        player_dialog.setWindowTitle("Аудио плеер")
        player_dialog.setMinimumSize(1000, 700)

        player = AudioPlayer(player_dialog, self.separated_tracks, self.selected_file)
        player_dialog.exec_()

    def run(self):
        self.show()
