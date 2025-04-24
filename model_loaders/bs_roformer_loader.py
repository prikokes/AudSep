#
# Created by Gosha Ivanov on 18.04.2025.
#

import torch
import os
import requests
import yaml

from models.bs_roformer import BSRoformer


class BSRoformerLoader:
    WEIGHTS_PATH = './weights/bs_roformer.ckpt'
    WEIGHTS_URL = 'https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.12/model_bs_roformer_ep_17_sdr_9.6568.ckpt'

    def __init__(self):
        pass

    @staticmethod
    def download_weights():
        os.makedirs('./weights', exist_ok=True)

        response = requests.get(BSRoformerLoader.WEIGHTS_URL, stream=True)
        total_size = int(response.headers.get('content-length', 0))

        with open(BSRoformerLoader.WEIGHTS_PATH, 'wb') as file:
            file.write(response.content)

    def load(self, type_, device, config):
        if type_ == 'bs':
            if not os.path.exists(BSRoformerLoader.WEIGHTS_PATH):
                BSRoformerLoader.download_weights()

            model = BSRoformer(
                **dict(config.model)
            )

            state_dict = torch.load(BSRoformerLoader.WEIGHTS_PATH, map_location=device, weights_only=False)
            if 'state' in state_dict:
                state_dict = state_dict['state']
            if 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']

            model = model.to(device)
            model.load_state_dict(state_dict)

            return model
        else:
            raise NotImplementedError("Error! BS RoFormer supports only 'bs' version in our app")