#
# Created by Gosha Ivanov on 08.02.2025.
#
import torch
import os
import requests
import sys
from pathlib import Path

from omegaconf import OmegaConf
from models.htdemucs import HTDemucs
from utils.user_data import get_weights_dir
from utils.path_utils import get_resource_path


class HTDemucsLoader:
    WEIGHTS_FILENAME = 'ht_demucs_v4.th'
    WEIGHTS_URL = 'https://dl.fbaipublicfiles.com/demucs/hybrid_transformer/5c90dfd2-34c22ccb.th'

    def __init__(self):
        self._initialize_paths()

    def _initialize_paths(self):
        user_weights_dir = get_weights_dir()
        self.weights_path = os.path.join(user_weights_dir, self.WEIGHTS_FILENAME)

    @staticmethod
    def download_weights():
        user_weights_dir = get_weights_dir()
        weights_path = os.path.join(user_weights_dir, HTDemucsLoader.WEIGHTS_FILENAME)

        print(f"Downloading weights to: {weights_path}")

        os.makedirs(user_weights_dir, exist_ok=True)

        response = requests.get(HTDemucsLoader.WEIGHTS_URL, stream=True)
        total_size = int(response.headers.get('content-length', 0))

        with open(weights_path, 'wb') as file:
            file.write(response.content)

        print(f"Downloaded weights to: {weights_path}")
        return weights_path

    def load(self, type_, device, config):
        if type_ == '4s':
            pass
        elif type_ == '6s':
            if not os.path.exists(self.weights_path):
                self.weights_path = HTDemucsLoader.download_weights()

            extra = {
                'sources': list(config.training.instruments),
                'audio_channels': config.training.channels,
                'samplerate': config.training.samplerate,
                'segment': config.training.segment,
            }

            kw = OmegaConf.to_container(getattr(config, config.model), resolve=True)

            model = HTDemucs(**extra, **kw)

            print(f"Loading weights from: {self.weights_path}")
            state_dict = torch.load(self.weights_path, map_location=device, weights_only=False)
            if 'state' in state_dict:
                state_dict = state_dict['state']
            if 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']

            model = model.to(device)
            model.load_state_dict(state_dict)

            return model
        else:
            raise NotImplementedError("Error! HTDemucs supports only 4s and 6s versions in our app")