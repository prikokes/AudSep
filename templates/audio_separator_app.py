#
# Created by Gosha Ivanov on 08.02.2025.
#

import customtkinter as ctk
import torch
import torchaudio
import threading
import yaml

from pathlib import Path

from ml_collections import ConfigDict

from model_loaders import htdemucs_loader
from omegaconf import OmegaConf
from utils.demix_track_demucs import demix_track_demucs

from .audio_player import (AudioPlayer)

from model_loaders.mel_band_roformer_loader import MelBandRoformerLoader
from utils.demix_track import demix_track


class AudioSeparatorApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("AudSep")
        self.geometry("800x600")

        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.main_frame = ctk.CTkFrame(master=self)
        self.main_frame.pack(pady=20, padx=20, fill="both", expand=True)

        self.is_processing = False

        self.model_frame = ctk.CTkFrame(master=self.main_frame)
        self.model_frame.pack(pady=10, padx=20, fill="x")

        self.model_label = ctk.CTkLabel(
            self.model_frame,
            text="Выбери модель:"
        )
        self.model_label.pack(side="left", padx=(0, 10))

        self.available_models = {
            "HTDemucs (6 стемов)": {
                "loader": htdemucs_loader.HTDemucsLoader(),
                "config": "./configs/config_htdemucs_6stems.yaml",
                "model_id": "6s",
                "processor": self._process_htdemucs
            },
            "MelBand RoFormer": {
                "loader": MelBandRoformerLoader,
                "config": "./configs/config_vocals_mel_band_roformer_kj.yaml",
                "model_id": "base",
                "processor": self._process_melband_roformer
            }
        }

        self.model_var = ctk.StringVar(value="HTDemucs (6 стемов)")
        self.model_dropdown = ctk.CTkOptionMenu(
            self.model_frame,
            values=list(self.available_models.keys()),
            variable=self.model_var
        )
        self.model_dropdown.pack(side="left", fill="x", expand=True)

        self.select_button = ctk.CTkButton(
            self.main_frame,
            text="Выбрать аудио файл",
            command=self.select_file
        )
        self.select_button.pack(pady=10, padx=20)

        self.file_label = ctk.CTkLabel(
            self.main_frame,
            text="Файл не выбран"
        )
        self.file_label.pack(pady=10, padx=20)

        self.progress_bar = ctk.CTkProgressBar(self.main_frame)
        self.progress_bar.pack(pady=10, padx=20)
        self.progress_bar.set(0)

        self.process_button = ctk.CTkButton(
            self.main_frame,
            text="Разделить",
            command=self._separate_audio,
            state="disabled"
        )
        self.process_button.pack(pady=10, padx=20)

        self.status_label = ctk.CTkLabel(
            self.main_frame,
            text=""
        )
        self.status_label.pack(pady=10)

        self.selected_file = None
        self.separated_tracks = {}

    def select_file(self):
        if self.is_processing:
            return

        self.update()

        try:
            print("Cringe!")
            file_path = ctk.filedialog.askopenfilename(
                filetypes=[("Audio Files", "*.mp3 *.wav")]
            )
            if file_path:
                self.selected_file = Path(file_path)
                self.file_label.configure(text=str(self.selected_file))
                self.process_button.configure(state="normal")
        except Exception as e:
            self.status_label.configure(text=f"Ошибка при выборе файла: {str(e)}")

    def process_audio(self):
        if self.is_processing:
            return

        self.is_processing = True
        self.select_button.configure(state="disabled")
        self.process_button.configure(state="disabled")

        self.update()

        thread = threading.Thread(target=self._process_audio_thread)
        thread.daemon = True
        thread.start()

    def _process_audio_thread(self):
        try:
            self.after(0, lambda: self.status_label.configure(text="Разделяем"))
            self.progress_bar.set(0)

            self._separate_audio()

            self.progress_bar.set(1)
            self.after(0, lambda: self.status_label.configure(text="Готово"))
        except Exception as e:
            print(e)
            self.after(0, lambda: self.status_label.configure(text=f"Ошибка: {e}"))
        finally:
            self.is_processing = False
            self.after(0, lambda: self.select_button.configure(state="normal"))
            self.after(0, lambda: self.process_button.configure(state="normal"))

    def _separate_audio(self):
        mix, sample_rate = torchaudio.load(self.selected_file)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        selected_model_name = self.model_var.get()
        model_info = self.available_models[selected_model_name]

        processor = model_info["processor"]
        self.separated_tracks = processor(mix, sample_rate, device, model_info)

        self.open_player()

    def _process_htdemucs(self, mix, sample_rate, device, model_info):
        loader = model_info["loader"]

        config = OmegaConf.load(model_info["config"])

        model = loader.load(model_info["model_id"], device, config)

        waveform = demix_track_demucs(config, model, mix, device, pbar=False, progress_bar=self.progress_bar)

        tracks = {}
        for stem in config.training.instruments:
            tracks[stem] = {'data': torch.tensor(waveform[stem]).float(), 'sr': sample_rate}

        return tracks

    def _process_melband_roformer(self, mix, sample_rate, device, model_info):
        self.status_label.configure(text="Загрузка MelBand RoFormer...")

        with open(model_info['config'], 'r') as f:
            config_dict = yaml.load(f, Loader=yaml.SafeLoader)

        print("I am here")

        config = ConfigDict(config_dict)

        loader = model_info["loader"]()
        model = loader.load(model_info["model_id"], device, config)

        waveform = demix_track(config, model, mix, device, pbar=False)

        tracks = {}
        for stem in waveform.keys():
            tracks[stem] = {'data': torch.tensor(waveform[stem]).float(), 'sr': sample_rate}

        return tracks

    def open_player(self):
        def create_player():
            player_window = ctk.CTkToplevel(self)
            player_window.title("Аудио плеер")
            player_window.geometry("800x600")
            player_window.grab_set()

            player = AudioPlayer(player_window, self.separated_tracks, self.selected_file)

        self.after_idle(create_player)

    def run(self):
        self.mainloop()

