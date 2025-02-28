import customtkinter as ctk
import sounddevice as sd

import numpy as np

import torch
import torchaudio


class AudioPlayer:
    def __init__(self, root, tracks_data):
        self.root = root
        self.tracks = {}
        self.audio_data = tracks_data
        self.playing = False
        self.active_streams = {}

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
            self.create_track_row(name.capitalize(), colors.get(name, "#FFFFFF"), data)

    def create_track_row(self, name, color, data):
        track_frame = ctk.CTkFrame(self.tracks_container)
        track_frame.pack(fill="x", pady=5)

        label = ctk.CTkLabel(track_frame, text=name, width=100)
        label.pack(side="left", padx=5)

        solo_btn = ctk.CTkButton(track_frame, text="S", width=30, fg_color=color)
        solo_btn.pack(side="left", padx=2)

        mute_btn = ctk.CTkButton(track_frame, text="M", width=30)
        mute_btn.pack(side="left", padx=2)

        volume = ctk.CTkSlider(track_frame, from_=0, to=1, width=100)
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
            'solo': solo_btn
        }

    def make_callback(self, data):
        frame_index = 0

        def callback(outdata, frames, time, status):
            nonlocal frame_index

            if status:
                print(f'Status: {status}')

            try:
                if frame_index + frames > len(data):
                    remaining = len(data) - frame_index
                    if remaining > 0:
                        outdata[:remaining] = data[frame_index:frame_index + remaining]
                        outdata[remaining:] = 0
                    else:
                        outdata.fill(0)
                    raise sd.CallbackStop()
                else:
                    outdata[:] = data[frame_index:frame_index + frames]
                    frame_index += frames

            except Exception as e:
                print(f"Error in callback: {e}")
                raise sd.CallbackStop()

        return callback

    def stop_track(self, name):
        if name.lower() in self.active_streams:
            self.active_streams[name.lower()].stop()
            self.active_streams[name.lower()].close()
            del self.active_streams[name.lower()]

    def play_track(self, name: str):
        try:
            track = self.tracks[name.lower()]
            data = track['data']['data']
            sr = track['data']['sr']
            volume = track['volume'].get()

            if isinstance(data, torch.Tensor):
                data = data.cpu().numpy()

            print(f"\nDebug info for track {name}:")
            print(f"Shape: {data.shape}")
            print(f"Sample rate: {sr}")
            print(f"Data type: {data.dtype}")
            print(f"Min value: {np.min(data)}")
            print(f"Max value: {np.max(data)}")

            data = data.astype(np.float32)
            data = data / np.max(np.abs(data)) * volume

            if data.shape[0] == 2:
                data = data.T
            elif data.ndim == 1:
                data = data.reshape(-1, 1)

            stream = sd.OutputStream(
                samplerate=sr,
                channels=data.shape[1] if data.ndim > 1 else 1,
                dtype=np.float32,
                callback=self.make_callback(data)
            )
            self.active_streams[name.lower()] = stream
            stream.start()
        except Exception as e:
            print(f"Error playing track {name}: {e}")
            import traceback
            traceback.print_exc()

    def play_all(self):
        if not self.playing:
            for name in self.audio_data:
                self.play_track(name)
            self.playing = True
            self.play_button.configure(text="⏸️")
        else:
            self.stop_all()

    def stop_all(self):
        for name in list(self.active_streams.keys()):
            self.stop_track(name)
        self.playing = False
        self.play_button.configure(text="▶️")
