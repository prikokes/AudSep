#
# Created by Gosha Ivanov on 08.02.2025.
#

import torch
import os
import requests
import yaml

from models.mel_band_roformer import MelBandRoformer
from utils.user_data import get_weights_dir
from utils.path_utils import get_resource_path


class MelBandRoformerLoader:
    WEIGHTS_FILENAME = 'melband_roformer.ckpt'
    WEIGHTS_URL = 'https://huggingface.co/KimberleyJSN/melbandroformer/resolve/main/MelBandRoformer.ckpt'

    def __init__(self):
        pass

    def _initialize_paths(self):
        user_weights_dir = get_weights_dir()
        self.weights_path = os.path.join(user_weights_dir, self.WEIGHTS_FILENAME)

    @staticmethod
    def download_weights():
        user_weights_dir = get_weights_dir()
        weights_path = os.path.join(user_weights_dir, MelBandRoformerLoader.WEIGHTS_FILENAME)

        print(f"Downloading weights to: {weights_path}")

        response = requests.get(MelBandRoformerLoader.WEIGHTS_URL, stream=True)
        total_size = int(response.headers.get('content-length', 0))

        with open(weights_path, 'wb') as file:
            file.write(response.content)

        print(f"Downloaded weights to: {weights_path}")
        return weights_path

    def load(self, type_, device, config):
        if type_ == 'base':
            if not hasattr(self, 'weights_path'):
                self._initialize_paths()

            if not os.path.exists(self.weights_path):
                MelBandRoformerLoader.download_weights()

            model = MelBandRoformer(
                **dict(config.model)
            )

            state_dict = torch.load(self.weights_path, map_location=device, weights_only=False)
            if 'state' in state_dict:
                state_dict = state_dict['state']
            if 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']

            model = model.to(device)
            model.load_state_dict(state_dict)

            return model
        else:
            raise NotImplementedError("Error! MelBand RoFormer supports only 'base' version in our app")