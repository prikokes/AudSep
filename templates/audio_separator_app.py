#
# Created by Gosha Ivanov on 08.02.2025.
#

import customtkinter as ctk
import torch
import torchaudio
import threading
from pathlib import Path

from model_loaders import htdemucs_loader
from omegaconf import OmegaConf
from utils.demix_track_demucs import demix_track_demucs

from .audio_player import (AudioPlayer)


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
            command=self.process_audio,
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

        device = torch.device("cpu")
        config = OmegaConf.load("./configs/config_htdemucs_6stems.yaml")

        loader = htdemucs_loader.HTDemucsLoader()

        model = loader.load('6s', device, config)

        waveform = demix_track_demucs(config, model, mix, device, pbar=False, progress_bar=self.progress_bar)

        vocals = torch.tensor(waveform['vocals']).float()
        bass = torch.tensor(waveform['bass']).float()
        drums = torch.tensor(waveform['drums']).float()
        other = torch.tensor(waveform['other']).float()
        guitar = torch.tensor(waveform['guitar']).float()
        piano = torch.tensor(waveform['piano']).float()

        output_dir = self.selected_file.parent / "separated"
        output_dir.mkdir(exist_ok=True)

        self.separated_tracks = {
            'vocals': {'data': vocals, 'sr': sample_rate},
            'bass': {'data': bass, 'sr': sample_rate},
            'drums': {'data': drums, 'sr': sample_rate},
            'other': {'data': other, 'sr': sample_rate},
            'guitar': {'data': guitar, 'sr': sample_rate},
            'piano': {'data': piano, 'sr': sample_rate}
        }

        # print(guitar.numpy().dtype)
        # sf.write("piano.wav", piano.numpy(), 44100)

        self.open_player()

    def open_player(self):
        def create_player():
            player_window = ctk.CTkToplevel(self)
            player_window.title("Аудио плеер")
            player_window.geometry("800x600")
            player_window.grab_set()

            player = AudioPlayer(player_window, self.separated_tracks)

        self.after_idle(create_player)

    def run(self):
        self.mainloop()

