#
# Created by Gosha Ivanov on 08.02.2025.
#

import torch
import os
import requests

from omegaconf import OmegaConf
# from demucs.htdemucs import HTDemucs

from models.htdemucs import HTDemucs


class HTDemucsLoader:
    WEIGHTS_PATH = './weights/ht_demucs_v4.th'
    WEIGHTS_URL = 'https://dl.fbaipublicfiles.com/demucs/hybrid_transformer/5c90dfd2-34c22ccb.th'

    def __init__(self):
        pass

    @staticmethod
    def download_weights():
        os.makedirs('./weights', exist_ok=True)

        response = requests.get(HTDemucsLoader.WEIGHTS_URL, stream=True)
        total_size = int(response.headers.get('content-length', 0))

        with open(HTDemucsLoader.WEIGHTS_PATH, 'wb') as file:
            file.write(response.content)

    def load(self, type_, device, config):
        if type_ == '4s':
            pass
        elif type_ == '6s':
            if not os.path.exists(HTDemucsLoader.WEIGHTS_PATH):
                HTDemucsLoader.download_weights()

            extra = {
                'sources': list(config.training.instruments),
                'audio_channels': config.training.channels,
                'samplerate': config.training.samplerate,
                'segment': config.training.segment,
            }

            kw = OmegaConf.to_container(getattr(config, config.model), resolve=True)

            model = HTDemucs(**extra, **kw)

            state_dict = torch.load(HTDemucsLoader.WEIGHTS_PATH, map_location=device, weights_only=False)
            if 'state' in state_dict:
                state_dict = state_dict['state']
            if 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']

            model.load_state_dict(state_dict)

            return model
        else:
            raise NotImplementedError("Error! HTDemucs supports only 4s and 6s versions in our app")