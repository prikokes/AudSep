import customtkinter as ctk
import numpy as np
import torch
import torchaudio
import vlc
import tempfile
import os


class AudioPlayer:
    def __init__(self, root, tracks_data, original_file=None):
        self.root = root
        self.tracks = {}
        self.audio_data = tracks_data
        self.playing = False
        self.active_players = {}
        self.temp_files = []
        self.original_file = original_file

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

        self.save_button = ctk.CTkButton(
            self.controls_frame,
            text="💾",
            width=40,
            command=self.save_results
        )
        self.save_button.pack(side="left", padx=5)

        self.tracks_container = ctk.CTkScrollableFrame(self.player_frame)
        self.tracks_container.pack(fill="both", expand=True)

        colors = {
            'vocals': "#FF5555",
            'bass': "#5555FF",
            'drums': "#55FF55",
            'guitar': "#FFFF55",
            'piano': "#FF55FF",
            'other': "#55FFFF"
        }

        for name, data in self.audio_data.items():
            self.prepare_track(name, data)
            self.create_track_row(name.capitalize(), colors.get(name, "#FFFFFF"), data)

        self.vlc_instance = vlc.Instance()

    def __del__(self):
        for temp_file in self.temp_files:
            try:
                os.remove(temp_file)
            except:
                pass

    def prepare_track(self, name, track_data):
        data = track_data['data']
        sr = track_data['sr']

        if isinstance(data, torch.Tensor):
            data = data.cpu().numpy()

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

                    mixed_tensor = torch.tensor(mixed_data)

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
                            torchaudio.save(new_file_path, mixed_tensor, sample_rate)
                            print(f"Результаты сохранены в {new_file_path}")
                        except Exception as e:
                            print(f"Ошибка при сохранении файла: {e}")
                    else:
                        print("Ошибка при смешивании треков")

    def create_track_row(self, name, color, data):
        track_frame = ctk.CTkFrame(self.tracks_container)
        track_frame.pack(fill="x", pady=5)

        label = ctk.CTkLabel(track_frame, text=name, width=100)
        label.pack(side="left", padx=5)

        solo_btn = ctk.CTkButton(track_frame, text="S", width=30, fg_color=color,
                                 command=lambda n=name: self.solo_track(n))
        solo_btn.pack(side="left", padx=2)

        mute_btn = ctk.CTkButton(track_frame, text="M", width=30,
                                 command=lambda n=name: self.mute_track(n))
        mute_btn.pack(side="left", padx=2)

        volume = ctk.CTkSlider(track_frame, from_=0, to=1, width=100,
                               command=lambda val, n=name: self.update_volume(n, val))
        volume.set(0.8)
        volume.pack(side="left", padx=10)

        waveform = ctk.CTkProgressBar(track_frame, width=400, progress_color=color)
        waveform.set(0.7)
        waveform.pack(side="left", padx=10, fill="x", expand=True)

        self.tracks[name.lower()] = {
            'frame': track_frame,
            'data': data,
            'volume': volume,
            'waveform': waveform,
            'mute': mute_btn,
            'solo': solo_btn,
            'is_muted': False,
            'is_solo': False
        }

    def stop_track(self, name):
        if name.lower() in self.active_players:
            self.active_players[name.lower()].stop()
            del self.active_players[name.lower()]

    def play_track(self, name: str):
        try:
            track = self.tracks[name.lower()]
            temp_file = track['data']['temp_file']
            volume_val = track['volume'].get()

            media = self.vlc_instance.media_new(temp_file)
            player = self.vlc_instance.media_player_new()
            player.set_media(media)

            player.audio_set_volume(int(volume_val * 100))

            player.play()
            self.active_players[name.lower()] = player

            print(f"Воспроизводим трек {name} из {temp_file}")

        except Exception as e:
            print(f"Ошибка при воспроизведении трека {name}: {e}")
            import traceback
            traceback.print_exc()

    def update_volume(self, name, value):
        if name.lower() in self.active_players:
            self.active_players[name.lower()].audio_set_volume(int(value * 100))

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