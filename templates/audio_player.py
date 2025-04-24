import customtkinter as ctk
import numpy as np
import torch
import torchaudio
import vlc
import tempfile
import os
import soundfile as sf
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from PIL import Image, ImageTk
import io
import time


class AudioPlayer:
    def __init__(self, root, tracks_data, original_file=None):
        self.root = root
        self.tracks = {}
        self.audio_data = tracks_data
        self.playing = False
        self.active_players = {}
        self.temp_files = []
        self.original_file = original_file

        self.waveform_canvases = {}
        self.waveform_images = {}
        self.waveform_originals = {}
        self.position_markers = {}

        self.resize_timer = None
        self.last_resize_time = 0
        self.resize_throttle_ms = 200
        self.resize_in_progress = False

        self.resize_method = Image.BILINEAR

        self.base_waveform_width = 1500
        self.base_waveform_height = 300

        self.player_frame = ctk.CTkFrame(root)
        self.player_frame.pack(pady=20, padx=20, fill="both", expand=True)

        self.controls_frame = ctk.CTkFrame(self.player_frame)
        self.controls_frame.pack(pady=10)

        self.play_button = ctk.CTkButton(
            self.controls_frame,
            text="▶️",
            width=40,
            command=self.play_all
        )
        self.play_button.pack(side="left", padx=5)

        self.stop_button = ctk.CTkButton(
            self.controls_frame,
            text="⏹️",
            width=40,
            command=self.stop_all
        )
        self.stop_button.pack(side="left", padx=5)

        self.volume_label = ctk.CTkLabel(
            self.controls_frame,
            text="Громкость:",
            width=80
        )
        self.volume_label.pack(side="left", padx=(15, 5))

        self.master_volume = ctk.CTkSlider(
            self.controls_frame,
            from_=0,
            to=1,
            width=120,
            command=self.update_master_volume
        )
        self.master_volume.set(1.0)
        self.master_volume.pack(side="left", padx=5)

        self.master_volume_value = 1.0

        self.save_button = ctk.CTkButton(
            self.controls_frame,
            text="💾",
            width=40,
            command=self.save_results
        )
        self.save_button.pack(side="left", padx=5)

        self.tracks_container = ctk.CTkScrollableFrame(self.player_frame)
        self.tracks_container.pack(fill="both", expand=True)

        self.track_colors = {
            'vocals': {
                'main': "#FF5555",
                'bg': "#4D1919",  # Темная версия для фона
                'plot': [1.0, 0.33, 0.33]  # RGB для matplotlib (0-1)
            },
            'bass': {
                'main': "#5555FF",
                'bg': "#19194D",
                'plot': [0.33, 0.33, 1.0]
            },
            'drums': {
                'main': "#55FF55",
                'bg': "#194D19",
                'plot': [0.33, 1.0, 0.33]
            },
            'guitar': {
                'main': "#FFFF55",
                'bg': "#4D4D19",
                'plot': [1.0, 1.0, 0.33]
            },
            'piano': {
                'main': "#FF55FF",
                'bg': "#4D194D",
                'plot': [1.0, 0.33, 1.0]
            },
            'other': {
                'main': "#55FFFF",
                'bg': "#194D4D",
                'plot': [0.33, 1.0, 1.0]
            }
        }

        self.default_color = {
            'main': "#FFFFFF",
            'bg': "#4D4D4D",
            'plot': [1.0, 1.0, 1.0]
        }

        for name, data in self.audio_data.items():
            self.prepare_track(name, data)
            waveform_data = data['waveform_data']
            color = self.track_colors.get(name.lower(), self.default_color)['plot']
            self.waveform_originals[name.lower()] = self.create_original_waveform(waveform_data, color)

        for name, data in self.audio_data.items():
            track_color = self.track_colors.get(name.lower(), self.default_color)
            self.create_track_row(name.capitalize(), track_color, data)

        self.root.update_idletasks()
        for name in self.audio_data.keys():
            self.root.after(200, lambda n=name.lower(): self.force_draw_waveform(n))

        self.vlc_instance = vlc.Instance()

    def __del__(self):
        for temp_file in self.temp_files:
            try:
                os.remove(temp_file)
            except:
                pass

    def update_master_volume(self, value):
        self.master_volume_value = value

        for name, player in self.active_players.items():
            track_volume = self.tracks[name]['volume'].get()
            adjusted_volume = int(track_volume * value * 100)
            player.audio_set_volume(adjusted_volume)

    def force_draw_waveform(self, name):
        try:
            canvas = self.waveform_canvases[name]
            canvas.delete("all")

            width = canvas.winfo_width() or 400
            height = canvas.winfo_height() or 60

            if width < 10:
                width = 400

            if name in self.waveform_originals:
                new_image = self.create_waveform_image(
                    self.waveform_originals[name],
                    width,
                    height,
                    high_quality=True
                )

                self.waveform_images[name] = new_image

                # Рисуем изображение на канвасе
                canvas.create_image(
                    width // 2,
                    height // 2,
                    image=new_image,
                    anchor="center"
                )

                # Создаем маркер позиции
                marker = canvas.create_line(0, 0, 0, height, fill="white", width=1)
                self.position_markers[name] = marker

                print(f"Принудительно отрисован waveform для {name}, размер: {width}x{height}")
        except Exception as e:
            print(f"Ошибка при принудительной отрисовке {name}: {e}")
            import traceback
            traceback.print_exc()

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

    def save_results(self):
        if self.playing:
            self.stop_all()

        tracks_to_mix = self.get_tracks_to_play()

        if not tracks_to_mix:
            print("Нечего сохранять: все треки заглушены")
            return

        first_track = self.audio_data[tracks_to_mix[0].lower()]
        sample_rate = first_track['sr']

        mixed_data = None

        for track_name in tracks_to_mix:
            track_data = self.audio_data[track_name.lower()]['data']
            volume = self.tracks[track_name.lower()]['volume'].get()

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

                    file_path = ctk.filedialog.asksaveasfilename(
                        defaultextension=".wav",
                        filetypes=[("WAV файлы", "*.wav"), ("Все файлы", "*.*")]
                    )

                    if file_path:
                        base_path, ext = os.path.splitext(file_path)
                        counter = 1
                        new_file_path = file_path

                        while os.path.exists(new_file_path):
                            new_file_path = f"{base_path}_{counter}{ext}"
                            counter += 1

                        try:
                            if len(mixed_data.shape) == 1:
                                mixed_data_sf = mixed_data.reshape(-1, 1)
                            else:
                                if mixed_data.shape[0] == 2:
                                    mixed_data_sf = mixed_data.T
                                else:
                                    mixed_data_sf = mixed_data

                            sf.write(new_file_path, mixed_data_sf, sample_rate)
                            print(f"Результаты сохранены в {new_file_path}")
                        except Exception as e:
                            print(f"Ошибка при сохранении файла: {e}")
                            import traceback
                            traceback.print_exc()
                    else:
                        print("Ошибка при смешивании треков")

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
        ax.plot(x, data, color=color, linewidth=0.8)
        ax.fill_between(x, data, alpha=0.3, color=color)

        ax.set_ylim(-1, 1)
        ax.axis('off')
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)

        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        buf.seek(0)
        original_img = Image.open(buf).copy()
        plt.close(fig)

        if original_img:
            print(f"Создано изображение размером {original_img.size}")
        else:
            print("ОШИБКА: Не удалось создать изображение!")

        return original_img

    def create_waveform_image(self, original_img, width, height, high_quality=True):
        if width < 1:
            width = 400
        if height < 1:
            height = 60

        try:
            img_copy = original_img.copy()
            method = self.resize_method if high_quality else Image.NEAREST
            if width < 50:
                width = 50

            resized_img = img_copy.resize((width, height), method)
            return ImageTk.PhotoImage(resized_img)
        except Exception as e:
            print(f"Ошибка при ресайзе изображения: {e}")
            img = Image.new('RGB', (width, height), (100, 100, 100))
            return ImageTk.PhotoImage(img)

    def delayed_resize(self, event, name):
        name_lower = name.lower()
        if name_lower in self.waveform_originals:
            canvas = self.waveform_canvases[name_lower]
            width = canvas.winfo_width()
            height = canvas.winfo_height()

            if width > 10 and height > 10:
                try:
                    new_image = self.create_waveform_image(
                        self.waveform_originals[name_lower],
                        width,
                        height,
                        high_quality=True
                    )

                    self.waveform_images[name_lower] = new_image
                    print(f"Обновлено изображение для {name_lower}, размер: {width}x{height}")
                    self.draw_waveform(name_lower)
                except Exception as e:
                    print(f"Ошибка при ресайзе {name_lower}: {e}")

        self.resize_in_progress = False

    def on_canvas_configure(self, event, name):
        width = event.width
        height = event.height

        if width > 10 and height > 10:
            if hasattr(self, f"timer_{name}"):
                timer = getattr(self, f"timer_{name}")
                if timer:
                    self.root.after_cancel(timer)

            timer = self.root.after(
                50,
                lambda w=width, h=height, n=name: self.resize_waveform(n, w, h)
            )
            setattr(self, f"timer_{name}", timer)

    def resize_waveform(self, name, width, height):
        try:
            canvas = self.waveform_canvases[name]
            canvas.delete("all")

            if name in self.waveform_originals:
                new_image = self.create_waveform_image(
                    self.waveform_originals[name],
                    width,
                    height,
                    high_quality=True
                )

                # Сохраняем и отображаем
                self.waveform_images[name] = new_image
                canvas.create_image(
                    width // 2,
                    height // 2,
                    image=new_image,
                    anchor="center",
                    tags="waveform_img"
                )

                marker = canvas.create_line(0, 0, 0, height, fill="white", width=1)
                self.position_markers[name] = marker

                print(f"Ресайз waveform для {name}: {width}x{height}")
        except Exception as e:
            print(f"Ошибка при ресайзе {name}: {e}")
            import traceback
            traceback.print_exc()

    def create_track_row(self, name, color, data):
        track_frame = ctk.CTkFrame(self.tracks_container)
        track_frame.pack(fill="x", pady=5)

        label = ctk.CTkLabel(track_frame, text=name, width=100)
        label.pack(side="left", padx=5)

        solo_btn = ctk.CTkButton(track_frame, text="S", width=30, fg_color=color['main'],
                                 command=lambda n=name: self.solo_track(n))
        solo_btn.pack(side="left", padx=2)

        mute_btn = ctk.CTkButton(track_frame, text="M", width=30,
                                 command=lambda n=name: self.mute_track(n))
        mute_btn.pack(side="left", padx=2)

        volume = ctk.CTkSlider(track_frame, from_=0, to=1, width=100,
                               command=lambda val, n=name: self.update_volume(n, val))
        volume.set(0.8)
        volume.pack(side="left", padx=10)

        waveform_frame = ctk.CTkFrame(track_frame, height=60)
        waveform_frame.pack(side="left", padx=10, fill="x", expand=True)

        waveform_canvas = ctk.CTkCanvas(waveform_frame, height=60,
                                        bg=color['bg'],
                                        highlightthickness=0,
                                        takefocus=0)
        waveform_canvas.pack(fill="both", expand=True)

        name_lower = name.lower()
        self.waveform_canvases[name_lower] = waveform_canvas

        self.tracks[name_lower] = {
            'frame': track_frame,
            'data': data,
            'volume': volume,
            'waveform_canvas': waveform_canvas,
            'mute': mute_btn,
            'solo': solo_btn,
            'is_muted': False,
            'is_solo': False
        }

        waveform_canvas.bind("<Configure>", lambda event, n=name.lower(): self.on_canvas_configure(event, n))

        self.root.after(100, lambda: self.draw_waveform(name_lower))

    def draw_waveform(self, name):
        canvas = self.waveform_canvases[name]
        canvas.delete("all")

        try:
            width = canvas.winfo_width() or 400
            height = canvas.winfo_height() or 60

            if name.lower() in self.waveform_originals:
                self.waveform_images[name] = self.create_waveform_image(
                    self.waveform_originals[name.lower()],
                    width,
                    height
                )

            if name in self.waveform_images and self.waveform_images[name]:
                image = self.waveform_images[name]
                canvas.create_image(
                    width // 2,
                    height // 2,
                    image=image,
                    anchor="center"
                )

            position_marker = canvas.create_line(
                0, 0, 0, height,
                fill="white", width=1
            )
            self.position_markers[name] = position_marker
        except Exception as e:
            print(f"Ошибка при отрисовке waveform для {name}: {e}")
            import traceback
            traceback.print_exc()

    def stop_track(self, name):
        if name.lower() in self.active_players:
            self.active_players[name.lower()].stop()
            del self.active_players[name.lower()]

    def play_track(self, name: str):
        try:
            track = self.tracks[name.lower()]
            temp_file = track['data']['temp_file']
            volume_val = track['volume'].get()

            # Учитываем общую громкость
            adjusted_volume = volume_val * self.master_volume_value

            media = self.vlc_instance.media_new(temp_file)
            player = self.vlc_instance.media_player_new()
            player.set_media(media)

            player.audio_set_volume(int(adjusted_volume * 100))

            player.play()
            self.active_players[name.lower()] = player

            self.root.after(100, lambda: self.update_position(name.lower()))

            print(f"Воспроизводим трек {name} из {temp_file}")

        except Exception as e:
            print(f"Ошибка при воспроизведении трека {name}: {e}")
            import traceback
            traceback.print_exc()

    def update_volume(self, name, value):
        if name.lower() in self.active_players:
            adjusted_volume = value * self.master_volume_value
            self.active_players[name.lower()].audio_set_volume(int(adjusted_volume * 100))

    def update_position(self, name):
        if name in self.active_players and self.playing:
            player = self.active_players[name]

            current_time = player.get_time() / 1000
            duration = player.get_length() / 1000

            if duration > 0:
                position_ratio = current_time / duration
                canvas_width = self.waveform_canvases[name].winfo_width()

                x_pos = int(position_ratio * canvas_width)
                self.waveform_canvases[name].coords(self.position_markers[name],
                                                    x_pos, 0, x_pos, self.waveform_canvases[name].winfo_height())

            self.root.after(50, lambda: self.update_position(name))

    def play_all(self):
        if not self.playing:
            self.playing = True
            self.play_button.configure(text="⏸️")

            tracks_to_play = self.get_tracks_to_play()

            for name in tracks_to_play:
                self.play_track(name)
        else:
            self.stop_all()

    def get_tracks_to_play(self):
        solo_tracks = [name for name, track in self.tracks.items() if track['is_solo']]

        if solo_tracks:
            return solo_tracks
        else:
            return [name for name, track in self.tracks.items() if not track['is_muted']]

    def mute_track(self, name):
        track = self.tracks[name.lower()]
        track['is_muted'] = not track['is_muted']

        if track['is_muted']:
            track['mute'].configure(fg_color="red")
        else:
            track['mute'].configure(fg_color="#1F6AA5")

        if self.playing:
            self.stop_all()
            self.playing = False
            self.play_all()

    def solo_track(self, name):
        track = self.tracks[name.lower()]
        track['is_solo'] = not track['is_solo']

        if track['is_solo']:
            track['solo'].configure(fg_color="green")
        else:
            track['solo'].configure(fg_color="#1F6AA5")

        if self.playing:
            self.stop_all()
            self.playing = False
            self.play_all()

    def stop_all(self):
        for name in list(self.active_players.keys()):
            self.stop_track(name)
        self.playing = False
        self.play_button.configure(text="▶️")