#
# Created by Gosha Ivanov on 08.02.2025.
#

import customtkinter as ctk
import torch
import torchaudio
import threading
from pathlib import Path
import numpy as np

from model_loaders import htdemucs_loader
from omegaconf import OmegaConf
from utils.demix_track_demucs import demix_track_demucs


class AudioSeparatorApp:
    def __init__(self):

        self.window = ctk.CTk()
        self.window.title("AudSep")
        self.window.geometry("800x600")

        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.main_frame = ctk.CTkFrame(self.window)
        self.main_frame.pack(pady=20, padx=20, fill="both", expand=True)

        self.select_button = ctk.CTkButton(
            self.main_frame,
            text="Выбрать аудио файл",
            command=self.select_file
        )
        self.select_button.pack(pady=10)

        self.file_label = ctk.CTkLabel(
            self.main_frame,
            text="Файл не выбран"
        )
        self.file_label.pack(pady=10)

        self.progress_bar = ctk.CTkProgressBar(self.main_frame)
        self.progress_bar.pack(pady=10)
        self.progress_bar.set(0)

        self.process_button = ctk.CTkButton(
            self.main_frame,
            text="Разделить",
            command=self.process_audio,
            state="disabled"
        )
        self.process_button.pack(pady=10)

        self.status_label = ctk.CTkLabel(
            self.main_frame,
            text=""
        )
        self.status_label.pack(pady=10)

        self.selected_file = None

    def select_file(self):
        file_path = ctk.filedialog.askopenfilename(
            filetypes=[("Audio Files", "*.mp3 *.wav")]
        )
        if file_path:
            self.selected_file = Path(file_path)
            self.file_label.configure(text=str(self.selected_file))
            self.process_button.configure(state="normal")

    def process_audio(self):
        thread = threading.Thread(target=self._process_audio_thread)
        thread.start()

    def _process_audio_thread(self):
        self.status_label.configure(text="Разделяем")
        self.process_button.configure(state="disabled")

        try:
            self._separate_audio()

            self.status_label.configure(text="Готово")
        except Exception as e:
            self.status_label.configure(text=f"Ошибка: {str(e)}")
            raise e
        finally:
            self.process_button.configure(state="normal")

    def _separate_audio(self):
        mix, sample_rate = torchaudio.load(self.selected_file)

        device = torch.device("cpu")
        config = OmegaConf.load("./configs/config_htdemucs_6stems.yaml")

        model = htdemucs_loader.HTDemucsLoader.load('6s', device, config)

        waveform = demix_track_demucs(config, model, mix, device, pbar=False)

        vocals = torch.tensor(waveform['vocals']).float()
        bass = torch.tensor(waveform['bass']).float()
        drums = torch.tensor(waveform['drums']).float()
        other = torch.tensor(waveform['other'] ).float()

        output_dir = self.selected_file.parent / "separated"
        output_dir.mkdir(exist_ok=True)

        torchaudio.save(
            output_dir / "vocals.wav",
            vocals,
            sample_rate
        )
        torchaudio.save(
            output_dir / "bass.wav",
            bass,
            sample_rate
        )
        torchaudio.save(
            output_dir / "drums.wav",
            drums,
            sample_rate
        )
        torchaudio.save(
            output_dir / "other.wav",
            other,
            sample_rate
        )

    def run(self):
        self.window.mainloop()
