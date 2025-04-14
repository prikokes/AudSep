#
# Created by Gosha Ivanov on 08.02.2025.
#

import torch
import os
import requests
import yaml

from models.mel_band_roformer import MelBandRoformer


class MelBandRoformerLoader:
    WEIGHTS_PATH = './weights/melband_roformer.ckpt'
    WEIGHTS_URL = 'https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/model_mel_band_roformer_ep_3005_sdr_11.4360.ckpt'

    def __init__(self):
        pass

    @staticmethod
    def download_weights():
        os.makedirs('./weights', exist_ok=True)

        response = requests.get(MelBandRoformerLoader.WEIGHTS_URL, stream=True)
        total_size = int(response.headers.get('content-length', 0))

        with open(MelBandRoformerLoader.WEIGHTS_PATH, 'wb') as file:
            file.write(response.content)

    def load(self, type_, device, config):
        if type_ == 'base':
            if not os.path.exists(MelBandRoformerLoader.WEIGHTS_PATH):
                MelBandRoformerLoader.download_weights()

            model = MelBandRoformer(
                **dict(config.model)
            )

            state_dict = torch.load(MelBandRoformerLoader.WEIGHTS_PATH, map_location=device, weights_only=False)
            if 'state' in state_dict:
                state_dict = state_dict['state']
            if 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']

            model.load_state_dict(state_dict)

            return model
        else:
            raise NotImplementedError("Error! MelBand RoFormer supports only 'base' version in our app")