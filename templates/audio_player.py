import numpy as np
import torch
import torchaudio
import vlc
import tempfile
import os
import soundfile as sf
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from PIL import Image
import io
import time
from PyQt5.QtWidgets import (QWidget, QLabel, QPushButton, QSlider, QVBoxLayout,
                             QHBoxLayout, QFrame, QScrollArea, QFileDialog, QGraphicsView,
                             QGraphicsScene, QGraphicsPixmapItem, QSizePolicy, QApplication, QDialog)
from PyQt5.QtCore import Qt, QSize, QTimer, QThread, pyqtSignal, QPropertyAnimation, QEasingCurve
from PyQt5.QtGui import QPixmap, QImage, QFont, QPainter, QColor, QIcon, QPalette, QBrush, QLinearGradient, QPen


class AudioPlayer:
    def __init__(self, root, tracks_data, original_file=None):
        self.root = root
        self.tracks = {}
        self.audio_data = tracks_data
        self.playing = False
        self.active_players = {}
        self.temp_files = []
        self.original_file = original_file

        self.waveform_views = {}
        self.waveform_images = {}
        self.waveform_originals = {}
        self.position_markers = {}

        self.base_waveform_width = 1500
        self.base_waveform_height = 300

        if self.root.layout() is not None:
            QWidget().setLayout(self.root.layout())

        self.root.setStyleSheet("""
            QDialog, QWidget {
                background-color: #222222;
                color: white;
            }

            QLabel {
                font-size: 14px;
                color: white;
            }

            QPushButton {
                background-color: #444444;
                color: white;
                border-radius: 5px;
                padding: 8px;
                font-weight: bold;
                min-height: 30px;
            }

            QPushButton:hover {
                background-color: #666666;
            }

            QSlider::groove:horizontal {
                border: none;
                height: 8px;
                background: #444444;
                border-radius: 4px;
            }

            QSlider::handle:horizontal {
                background: #007BFF;
                width: 16px;
                height: 16px;
                margin: -4px 0;
                border-radius: 8px;
            }

            QFrame#trackRow {
                background-color: #333333;
                border-radius: 8px;
                margin: 5px;
                padding: 5px;
            }
        """)

        main_layout = QVBoxLayout(self.root)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(15)

        title = QLabel("Аудио Плеер")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size: 20px; font-weight: bold; color: #007BFF;")
        main_layout.addWidget(title)

        if original_file:
            file_name = os.path.basename(str(original_file))
            file_label = QLabel(f"Файл: {file_name}")
            file_label.setAlignment(Qt.AlignCenter)
            main_layout.addWidget(file_label)

        controls_frame = QFrame()
        controls_layout = QHBoxLayout(controls_frame)
        controls_layout.setContentsMargins(10, 10, 10, 10)
        controls_layout.setSpacing(15)

        button_size = QSize(50, 50)
        button_style = """
            QPushButton {
                background-color: #007BFF;
                color: white;
                border-radius: 25px;
                font-size: 20px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #0069D9;
            }
            QPushButton:pressed {
                background-color: #0062CC;
            }
        """

        self.play_button = QPushButton("▶")
        self.play_button.setFixedSize(button_size)
        self.play_button.setStyleSheet(button_style)
        self.play_button.clicked.connect(self.play_all)
        controls_layout.addWidget(self.play_button)

        self.stop_button = QPushButton("⏹")
        self.stop_button.setFixedSize(button_size)
        self.stop_button.setStyleSheet(button_style)
        self.stop_button.clicked.connect(self.stop_all)
        controls_layout.addWidget(self.stop_button)

        separator = QFrame()
        separator.setFrameShape(QFrame.VLine)
        separator.setStyleSheet("background-color: #555555;")
        controls_layout.addWidget(separator)

        position_widget = QWidget()
        position_layout = QVBoxLayout(position_widget)
        position_layout.setContentsMargins(0, 0, 0, 0)
        position_layout.setSpacing(2)

        position_label = QLabel("Позиция:")
        position_label.setAlignment(Qt.AlignCenter)
        position_layout.addWidget(position_label)

        self.position_slider = QSlider(Qt.Horizontal)
        self.position_slider.setRange(0, 1000)
        self.position_slider.setValue(0)
        self.position_slider.setStyleSheet("""
            QSlider::handle:horizontal {
                background: #28A745;
                width: 16px;
                height: 16px;
                margin: -5px 0;
                border-radius: 8px;
            }
            QSlider::groove:horizontal {
                height: 8px;
                background: #444444;
                border-radius: 4px;
            }
        """)
        position_layout.addWidget(self.position_slider)

        self.time_layout = QHBoxLayout()
        self.current_time_label = QLabel("0:00")
        self.total_time_label = QLabel("0:00")
        self.time_layout.addWidget(self.current_time_label)
        self.time_layout.addStretch()
        self.time_layout.addWidget(self.total_time_label)
        position_layout.addLayout(self.time_layout)

        controls_layout.addWidget(position_widget, 3)

        separator2 = QFrame()
        separator2.setFrameShape(QFrame.VLine)
        separator2.setStyleSheet("background-color: #555555;")
        controls_layout.addWidget(separator2)

        volume_widget = QWidget()
        volume_layout = QHBoxLayout(volume_widget)
        volume_layout.setContentsMargins(0, 0, 0, 0)
        volume_layout.setSpacing(5)

        volume_icon = QLabel("🔊")
        volume_icon.setFont(QFont("Arial", 16))
        volume_layout.addWidget(volume_icon)

        self.master_volume = QSlider(Qt.Horizontal)
        self.master_volume.setRange(0, 100)
        self.master_volume.setValue(100)
        self.master_volume.setFixedWidth(80)
        self.master_volume.setStyleSheet("""
            QSlider::handle:horizontal {
                background: #007BFF;
                width: 14px;
                height: 14px;
                margin: -4px 0;
                border-radius: 7px;
            }
            QSlider::groove:horizontal {
                height: 6px;
                background: #444444;
                border-radius: 3px;
            }
        """)
        self.master_volume.valueChanged.connect(lambda val: self.update_master_volume(val / 100))
        volume_layout.addWidget(self.master_volume)

        self.volume_label = QLabel("100%")
        self.volume_label.setFixedWidth(40)
        self.volume_label.setAlignment(Qt.AlignCenter)
        volume_layout.addWidget(self.volume_label)
        self.master_volume.valueChanged.connect(lambda val: self.volume_label.setText(f"{val}%"))

        controls_layout.addWidget(volume_widget)

        self.save_button = QPushButton("💾")
        self.save_button.setFixedSize(button_size)
        self.save_button.setStyleSheet(button_style)
        self.save_button.clicked.connect(self.save_results)
        controls_layout.addWidget(self.save_button)

        self.position_slider.sliderPressed.connect(self.on_position_slider_pressed)
        self.position_slider.sliderReleased.connect(self.on_position_slider_released)
        self.position_slider.valueChanged.connect(self.on_position_slider_value_changed)
        self.slider_being_dragged = False

        main_layout.addWidget(controls_frame)

        self.track_duration = 0
        self.current_position = 0

        tracks_title = QLabel("Доступные треки:")
        tracks_title.setStyleSheet("font-size: 16px; color: #007BFF; margin-top: 10px;")
        main_layout.addWidget(tracks_title)

        tracks_container = QScrollArea()
        tracks_container.setWidgetResizable(True)
        tracks_container.setFrameShape(QFrame.NoFrame)

        tracks_content = QWidget()
        self.tracks_layout = QVBoxLayout(tracks_content)
        self.tracks_layout.setSpacing(10)
        tracks_container.setWidget(tracks_content)

        main_layout.addWidget(tracks_container, 1)

        self.track_colors = {
            'vocals': {"main": "#FF5555", "bg": "#4D1919", "plot": [1.0, 0.33, 0.33]},
            'bass': {"main": "#5555FF", "bg": "#19194D", "plot": [0.33, 0.33, 1.0]},
            'drums': {"main": "#55FF55", "bg": "#194D19", "plot": [0.33, 1.0, 0.33]},
            'guitar': {"main": "#FFFF55", "bg": "#4D4D19", "plot": [1.0, 1.0, 0.33]},
            'piano': {"main": "#FF55FF", "bg": "#4D194D", "plot": [1.0, 0.33, 1.0]},
            'other': {"main": "#55FFFF", "bg": "#194D4D", "plot": [0.33, 1.0, 1.0]}
        }

        self.default_color = {"main": "#FFFFFF", "bg": "#4D4D4D", "plot": [1.0, 1.0, 1.0]}

        for name, data in self.audio_data.items():
            self.prepare_track(name, data)
            waveform_data = data['waveform_data']
            color = self.track_colors.get(name.lower(), self.default_color)["plot"]
            self.waveform_originals[name.lower()] = self.create_original_waveform(waveform_data, color)

        for name, data in self.audio_data.items():
            track_color = self.track_colors.get(name.lower(), self.default_color)
            self.create_track_row(name.capitalize(), track_color, data)

        info_text = QLabel("Используйте кнопки Solo и Mute для управления треками")
        info_text.setAlignment(Qt.AlignCenter)
        info_text.setStyleSheet("color: #AAAAAA; font-size: 12px;")
        main_layout.addWidget(info_text)

        QTimer.singleShot(200, self.init_waveforms)

        self.vlc_instance = vlc.Instance()
        self.master_volume_value = 1.0

    def init_waveforms(self):
        for name in self.audio_data.keys():
            self.force_draw_waveform(name.lower())

    def create_track_row(self, name, color, data):
        track_row = QFrame()
        track_row.setObjectName("trackRow")
        track_layout = QHBoxLayout(track_row)
        track_layout.setContentsMargins(10, 10, 10, 10)
        track_layout.setSpacing(15)

        track_name = QLabel(name)
        track_name.setStyleSheet(f"color: {color['main']}; font-size: 16px; font-weight: bold; min-width: 100px;")
        track_name.setAlignment(Qt.AlignCenter)
        track_layout.addWidget(track_name)

        buttons_frame = QFrame()
        buttons_layout = QHBoxLayout(buttons_frame)
        buttons_layout.setContentsMargins(0, 0, 0, 0)
        buttons_layout.setSpacing(10)

        solo_btn = QPushButton("S")
        solo_btn.setStyleSheet(f"background-color: {color['main']}; color: white;")
        solo_btn.setMinimumWidth(80)
        solo_btn.clicked.connect(lambda: self.solo_track(name))
        buttons_layout.addWidget(solo_btn)

        mute_btn = QPushButton("M")
        mute_btn.setMinimumWidth(80)
        mute_btn.clicked.connect(lambda: self.mute_track(name))
        buttons_layout.addWidget(mute_btn)

        track_layout.addWidget(buttons_frame)

        volume_control = QFrame()
        volume_layout = QVBoxLayout(volume_control)
        volume_layout.setContentsMargins(0, 0, 0, 0)

        volume_header = QLabel("Volume: ")
        volume_header.setAlignment(Qt.AlignCenter)
        volume_layout.addWidget(volume_header)

        volume_slider_layout = QHBoxLayout()

        volume = QSlider(Qt.Horizontal)
        volume.setRange(0, 100)
        volume.setValue(80)
        volume.valueChanged.connect(lambda val, n=name: self.update_volume(n, val / 100))
        volume_slider_layout.addWidget(volume)

        volume_value = QLabel("80%")
        volume_value.setFixedWidth(40)
        volume_value.setAlignment(Qt.AlignCenter)
        volume_slider_layout.addWidget(volume_value)

        volume_layout.addLayout(volume_slider_layout)

        track_layout.addWidget(volume_control)

        volume.valueChanged.connect(lambda val: volume_value.setText(f"{val}%"))

        waveform_view = QGraphicsView()
        waveform_view.setMinimumHeight(80)
        waveform_view.setStyleSheet(
            f"background-color: {color['bg']}; border: 1px solid {color['main']}; border-radius: 5px;")
        waveform_view.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        waveform_view.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        waveform_view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        track_layout.addWidget(waveform_view, 1)

        name_lower = name.lower()
        self.waveform_views[name_lower] = waveform_view

        self.tracks[name_lower] = {
            'frame': track_row,
            'data': data,
            'volume': volume,
            'volume_label': volume_value,
            'waveform_view': waveform_view,
            'mute': mute_btn,
            'solo': solo_btn,
            'is_muted': False,
            'is_solo': False
        }

        waveform_view.resizeEvent = lambda event, n=name_lower: self.on_view_resize(
            n, waveform_view.width(), waveform_view.height()
        )

        self.tracks_layout.addWidget(track_row)

        QTimer.singleShot(100, lambda: self.draw_waveform(name_lower))

    def force_draw_waveform(self, name):
        try:
            view = self.waveform_views[name]
            scene = view.scene()
            if scene:
                scene.clear()

            width = view.width() or 400
            height = view.height() or 60

            if width < 10:
                width = 400

            if name in self.waveform_originals:
                pixmap = self.create_waveform_pixmap(
                    self.waveform_originals[name],
                    width,
                    height,
                    high_quality=True
                )

                self.waveform_images[name] = pixmap

                if not scene:
                    scene = QGraphicsScene()
                    view.setScene(scene)

                pixmap_item = QGraphicsPixmapItem(pixmap)
                scene.addItem(pixmap_item)

                pen = QPen(Qt.white)
                pen.setWidth(2)
                marker = scene.addLine(0, 0, 0, height, pen)
                self.position_markers[name] = marker

                print(f"Принудительно отрисован waveform для {name}, размер: {width}x{height}")
        except Exception as e:
            print(f"Ошибка при принудительной отрисовке {name}: {e}")
            import traceback
            traceback.print_exc()

    def play_all(self):
        if not self.playing:
            self.playing = True
            self.play_button.setText("⏸")

            tracks_to_play = self.get_tracks_to_play()

            for name in tracks_to_play:
                if name in self.active_players:
                    self.active_players[name].play()
                else:
                    self.play_track(name)

            self.start_position_updater()
        else:
            for name, player in self.active_players.items():
                player.pause()
            self.playing = False
            self.play_button.setText("▶")

    def start_position_updater(self):
        if not hasattr(self, 'position_timer') or self.position_timer is None:
            self.position_timer = QTimer()
            self.position_timer.timeout.connect(self.update_position_slider)
            self.position_timer.start(30)

    def update_position_slider(self):
        if self.slider_being_dragged or not self.active_players:
            return

        player = next(iter(self.active_players.values()))

        current_time = player.get_time() / 1000
        duration = player.get_length() / 1000

        if duration > 0 and abs(duration - self.track_duration) > 0.1:
            self.track_duration = duration
            self.total_time_label.setText(self.format_time(duration))

        if current_time >= 0 and duration > 0:
            self.current_position = current_time
            self.current_time_label.setText(self.format_time(current_time))
            position_value = int((current_time / duration) * 1000)
            self.position_slider.setValue(position_value)

            for name in self.active_players:
                if name in self.position_markers and name in self.waveform_views:
                    position_ratio = current_time / duration
                    view_width = self.waveform_views[name].width()
                    x_pos = int(position_ratio * view_width)

                    marker = self.position_markers[name]
                    marker.setLine(x_pos, 0, x_pos, self.waveform_views[name].height())

    def update_position(self, name):
        if name in self.active_players:
            player = self.active_players[name]

            duration = player.get_length() / 1000
            if duration > 0:
                current_time = player.get_time() / 1000
                position_ratio = current_time / duration
                view_width = self.waveform_views[name].width()
                x_pos = int(position_ratio * view_width)

                scene = self.waveform_views[name].scene()
                if scene and name in self.position_markers:
                    marker = self.position_markers[name]
                    marker.setLine(x_pos, 0, x_pos, self.waveform_views[name].height())

            QTimer.singleShot(30, lambda: self.update_position(name))

    def stop_all(self):
        for name, player in self.active_players.items():
            player.stop()
            if self.tracks[name]['data']['temp_file']:
                media = self.vlc_instance.media_new(self.tracks[name]['data']['temp_file'])
                player.set_media(media)

        self.playing = False
        self.play_button.setText("▶")
        self.position_slider.setValue(0)
        self.current_time_label.setText("0:00")

    def get_tracks_to_play(self):
        solo_tracks = [name for name, track in self.tracks.items() if track['is_solo']]

        if solo_tracks:
            return solo_tracks
        else:
            return [name for name, track in self.tracks.items() if not track['is_muted']]

    def stop_track(self, name):
        if name.lower() in self.active_players:
            self.active_players[name.lower()].stop()
            del self.active_players[name.lower()]

    def play_track(self, name):
        try:
            track = self.tracks[name.lower()]
            temp_file = track['data']['temp_file']
            volume_val = track['volume'].value() / 100

            adjusted_volume = volume_val * self.master_volume_value

            media = self.vlc_instance.media_new(temp_file)
            player = self.vlc_instance.media_player_new()
            player.set_media(media)

            player.audio_set_volume(int(adjusted_volume * 100))

            player.play()
            self.active_players[name.lower()] = player

            QTimer.singleShot(100, lambda: self.update_position(name.lower()))

        except Exception as e:
            print(f"Ошибка при воспроизведении трека {name}: {e}")
            import traceback
            traceback.print_exc()

    def update_master_volume(self, value):
        self.master_volume_value = value

        for name, player in self.active_players.items():
            track_volume = self.tracks[name]['volume'].value() / 100
            adjusted_volume = int(track_volume * value * 100)
            player.audio_set_volume(adjusted_volume)

    def update_volume(self, name, value):
        if name.lower() in self.active_players:
            adjusted_volume = value * self.master_volume_value
            self.active_players[name.lower()].audio_set_volume(int(adjusted_volume * 100))

    def save_results(self):
        if self.playing:
            self.stop_all()

        tracks_to_mix = self.get_tracks_to_play()

        if not tracks_to_mix:
            from PyQt5.QtWidgets import QMessageBox
            QMessageBox.warning(self.root, "Внимание",
                                "Нечего сохранять: все треки заглушены. Включите хотя бы один трек.")
            return

        first_track = self.audio_data[tracks_to_mix[0].lower()]
        sample_rate = first_track['sr']

        mixed_data = None

        for track_name in tracks_to_mix:
            track_data = self.audio_data[track_name.lower()]['data']
            volume = self.tracks[track_name.lower()]['volume'].value() / 100

            if isinstance(track_data, torch.Tensor):
                track_data = track_data.cpu().numpy()

            scaled_data = track_data * volume

            if mixed_data is None:
                mixed_data = scaled_data
            else:
                if mixed_data.shape != scaled_data.shape:
                    if len(mixed_data.shape) == 1 and len(scaled_data.shape) == 1:
                        mixed_data = mixed_data + scaled_data
                    elif len(mixed_data.shape) == 1:
                        mixed_data = np.vstack([mixed_data, mixed_data]) + scaled_data
                    elif len(scaled_data.shape) == 1:
                        scaled_data = np.vstack([scaled_data, scaled_data])
                        mixed_data = mixed_data + scaled_data
                else:
                    mixed_data = mixed_data + scaled_data

        if mixed_data is not None:
            max_val = np.max(np.abs(mixed_data))
            if max_val > 0:
                mixed_data = mixed_data / max_val

            file_path, _ = QFileDialog.getSaveFileName(
                self.root,
                "Сохранить микс",
                "",
                "WAV файлы (*.wav);;Все файлы (*.*)"
            )

            if file_path:
                try:
                    if len(mixed_data.shape) == 1:
                        mixed_data_sf = mixed_data.reshape(-1, 1)
                    else:
                        if mixed_data.shape[0] == 2:
                            mixed_data_sf = mixed_data.T
                        else:
                            mixed_data_sf = mixed_data

                    sf.write(file_path, mixed_data_sf, sample_rate)

                    from PyQt5.QtWidgets import QMessageBox
                    QMessageBox.information(self.root, "Сохранение",
                                            f"Микс успешно сохранен в:\n{file_path}")

                except Exception as e:
                    from PyQt5.QtWidgets import QMessageBox
                    QMessageBox.critical(self.root, "Ошибка",
                                         f"Не удалось сохранить файл:\n{str(e)}")

                    print(f"Ошибка при сохранении файла: {e}")
                    import traceback
                    traceback.print_exc()

    def create_waveform_pixmap(self, original_img, width, height, high_quality=True):
        if width < 1:
            width = 400
        if height < 1:
            height = 60

        try:
            img_copy = original_img.copy()
            method = Image.BILINEAR if high_quality else Image.NEAREST
            if width < 50:
                width = 50

            resized_img = img_copy.resize((width, height), method)
            img_data = resized_img.convert("RGBA").tobytes("raw", "RGBA")
            qimg = QImage(img_data, width, height, QImage.Format_RGBA8888)
            return QPixmap.fromImage(qimg)
        except Exception as e:
            print(f"Ошибка при ресайзе изображения: {e}")
            return QPixmap(width, height)

    def prepare_track(self, name, track_data):
        data = track_data['data']
        sr = track_data['sr']

        if isinstance(data, torch.Tensor):
            data = data.cpu().numpy()
            waveform_data = data.copy()
        else:
            waveform_data = data.copy()

        if waveform_data.ndim > 1:
            if waveform_data.shape[0] == 2:
                waveform_data = np.mean(waveform_data, axis=0)
            else:
                waveform_data = waveform_data[0]

        downsample_factor = max(1, len(waveform_data) // 2000)
        waveform_data = waveform_data[::downsample_factor]

        fd, temp_path = tempfile.mkstemp(suffix='.wav')
        os.close(fd)
        self.temp_files.append(temp_path)

        data = data.astype(np.float32)
        data = data / np.max(np.abs(data) + 1e-7)

        if data.ndim == 1:
            data = data.reshape(1, -1)
        elif data.shape[0] == 2:
            pass
        else:
            data = data.T

        tensor_data = torch.tensor(data)
        torchaudio.save(temp_path, tensor_data, sr)

        track_data['temp_file'] = temp_path
        track_data['waveform_data'] = waveform_data

    def create_original_waveform(self, waveform_data, color):
        if len(waveform_data) == 0:
            waveform_data = np.zeros(100)

        data = waveform_data.copy()

        if data.ndim > 1:
            data = data[0]

        data = np.nan_to_num(data)

        max_val = np.max(np.abs(data)) or 1.0
        data = data / max_val

        print(f"Создание waveform. Форма данных: {data.shape}, макс: {np.max(data)}, мин: {np.min(data)}")

        fig = Figure(figsize=(self.base_waveform_width / 100, self.base_waveform_height / 100), dpi=100)
        ax = fig.add_subplot(111)

        x = np.arange(len(data))
        ax.plot(x, data, color=color, linewidth=1.2)
        ax.fill_between(x, data, alpha=0.4, color=color)

        ax.set_ylim(-1, 1)
        ax.axis('off')
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
        fig.patch.set_alpha(0.0)

        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, transparent=True)
        buf.seek(0)
        original_img = Image.open(buf).copy()
        plt.close(fig)

        if original_img:
            print(f"Создано изображение размером {original_img.size}")
        else:
            print("ОШИБКА: Не удалось создать изображение!")

        return original_img

    def on_view_resize(self, name, width, height):
        if width > 10 and height > 10:
            timer_name = f"timer_{name}"
            if hasattr(self, timer_name) and getattr(self, timer_name) is not None:
                getattr(self, timer_name).stop()

            timer = QTimer()
            timer.setSingleShot(True)
            timer.timeout.connect(lambda: self.resize_waveform(name, width, height))
            timer.start(50)
            setattr(self, timer_name, timer)

    def resize_waveform(self, name, width, height):
        try:
            view = self.waveform_views[name]
            scene = view.scene()
            if scene:
                scene.clear()

            if name in self.waveform_originals:
                pixmap = self.create_waveform_pixmap(
                    self.waveform_originals[name],
                    width,
                    height,
                    high_quality=True
                )

                self.waveform_images[name] = pixmap

                if not scene:
                    scene = QGraphicsScene()
                    view.setScene(scene)

                pixmap_item = QGraphicsPixmapItem(pixmap)
                scene.addItem(pixmap_item)

                pen = QPen(Qt.white)
                pen.setWidth(2)
                marker = scene.addLine(0, 0, 0, height, pen)
                self.position_markers[name] = marker

                print(f"Ресайз waveform для {name}: {width}x{height}")
        except Exception as e:
            print(f"Ошибка при ресайзе {name}: {e}")
            import traceback
            traceback.print_exc()

    def on_position_slider_pressed(self):
        self.slider_being_dragged = True

    def on_position_slider_released(self):
        self.slider_being_dragged = False
        position_percent = self.position_slider.value() / 1000.0
        self.seek_all_tracks(position_percent)

    def on_position_slider_value_changed(self, value):
        if self.slider_being_dragged:
            position_percent = value / 1000.0
            current_time = int(position_percent * self.track_duration)
            self.current_time_label.setText(self.format_time(current_time))

    def seek_all_tracks(self, position_percent):
        if not self.active_players:
            return

        for name, player in self.active_players.items():
            duration = player.get_length()
            if duration > 0:
                new_position = int(position_percent * duration)
                player.set_time(new_position)

    def format_time(self, seconds):
        minutes = int(seconds // 60)
        seconds = int(seconds % 60)
        return f"{minutes}:{seconds:02d}"

    def update_position(self, name):
        if name in self.active_players and self.playing:
            player = self.active_players[name]

            current_time = player.get_time() / 1000
            duration = player.get_length() / 1000

            if duration > 0 and abs(duration - self.track_duration) > 0.1:
                self.track_duration = duration
                self.total_time_label.setText(self.format_time(duration))

            if current_time >= 0:
                self.current_position = current_time
                if not self.slider_being_dragged:
                    self.current_time_label.setText(self.format_time(current_time))
                    if duration > 0:
                        position_value = int((current_time / duration) * 1000)
                        self.position_slider.setValue(position_value)

            if duration > 0:
                position_ratio = current_time / duration
                view_width = self.waveform_views[name].width()

                x_pos = int(position_ratio * view_width)

                scene = self.waveform_views[name].scene()
                if scene and name in self.position_markers:
                    marker = self.position_markers[name]
                    marker.setLine(x_pos, 0, x_pos, self.waveform_views[name].height())

            QTimer.singleShot(30, lambda: self.update_position(name))

    def solo_track(self, name):
        track = self.tracks[name.lower()]
        track['is_solo'] = not track['is_solo']

        if track['is_solo']:
            track['solo'].setStyleSheet("background-color: #28A745; color: white;")

            if track['is_muted']:
                track['is_muted'] = False
                track['mute'].setStyleSheet("")
        else:
            color = self.track_colors.get(name.lower(), self.default_color)['main']
            track['solo'].setStyleSheet(f"background-color: {color}; color: white;")

        self.update_tracks_volume()

    def mute_track(self, name):
        track = self.tracks[name.lower()]
        track['is_muted'] = not track['is_muted']

        if track['is_muted']:
            track['mute'].setStyleSheet("background-color: #DC3545; color: white;")

            if track['is_solo']:
                track['is_solo'] = False
                color = self.track_colors.get(name.lower(), self.default_color)['main']
                track['solo'].setStyleSheet(f"background-color: {color}; color: white;")
        else:
            track['mute'].setStyleSheet("")

        self.update_tracks_volume()

    def update_tracks_volume(self):
        solo_tracks = [name for name, track in self.tracks.items() if track['is_solo']]

        for name, player in self.active_players.items():
            track = self.tracks[name]

            if (solo_tracks and name not in solo_tracks) or track['is_muted']:
                player.audio_set_volume(0)
            else:
                volume_val = track['volume'].value() / 100
                adjusted_volume = int(volume_val * self.master_volume_value * 100)
                player.audio_set_volume(adjusted_volume)

    def draw_waveform(self, name):
        view = self.waveform_views[name]
        scene = view.scene()
        if scene:
            scene.clear()
        else:
            scene = QGraphicsScene()
            view.setScene(scene)

        try:
            width = view.width() or 400
            height = view.height() or 60

            if name.lower() in self.waveform_originals:
                pixmap = self.create_waveform_pixmap(
                    self.waveform_originals[name.lower()],
                    width,
                    height
                )
                self.waveform_images[name] = pixmap

            if name in self.waveform_images and self.waveform_images[name]:
                pixmap_item = QGraphicsPixmapItem(self.waveform_images[name])
                scene.addItem(pixmap_item)

            pen = QPen(Qt.white)
            pen.setWidth(2)
            position_marker = scene.addLine(0, 0, 0, height, pen)
            self.position_markers[name] = position_marker
        except Exception as e:
            print(f"Ошибка при отрисовке waveform для {name}: {e}")
            import traceback
            traceback.print_exc()
