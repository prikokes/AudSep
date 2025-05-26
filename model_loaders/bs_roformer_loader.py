#
# Created by Gosha Ivanov on 18.04.2025.
#

import torch
import os
import requests
import yaml

from models.bs_roformer import BSRoformer
from utils.user_data import get_weights_dir
from utils.path_utils import get_resource_path


class BSRoformerLoader:
    WEIGHTS_FILENAME = 'bs_roformer.ckpt'
    WEIGHTS_URL = 'https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.12/model_bs_roformer_ep_17_sdr_9.6568.ckpt'

    def __init__(self):
        self._initialize_paths()

    def _initialize_paths(self):
        user_weights_dir = get_weights_dir()
        self.weights_path = os.path.join(user_weights_dir, self.WEIGHTS_FILENAME)

    @staticmethod
    def download_weights():
        user_weights_dir = get_weights_dir()
        weights_path = os.path.join(user_weights_dir, BSRoformerLoader.WEIGHTS_FILENAME)

        print(f"Downloading weights to: {weights_path}")

        os.makedirs(user_weights_dir, exist_ok=True)

        response = requests.get(BSRoformerLoader.WEIGHTS_URL, stream=True)
        total_size = int(response.headers.get('content-length', 0))

        with open(weights_path, 'wb') as file:
            file.write(response.content)

        print(f"Downloaded weights to: {weights_path}")
        return weights_path

    def load(self, type_, device, config):
        if type_ == 'bs':
            if not hasattr(self, 'weights_path'):
                self._initialize_paths()

            if not os.path.exists(self.weights_path):
                BSRoformerLoader.download_weights()

            model = BSRoformer(
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
            raise NotImplementedError("Error! BS RoFormer supports only 'bs' version in our app")