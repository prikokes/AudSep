#
# Created by Gosha Ivanov on 08.02.2025.
#

import torch
from omegaconf import OmegaConf
# from demucs.htdemucs import HTDemucs

from models.htdemucs import HTDemucs


class HTDemucsLoader:
    def __init__(self):
        pass

    def load(type_, device, config):
        if type_ == '4s':
            pass
        elif type_ == '6s':

            extra = {
                'sources': list(config.training.instruments),
                'audio_channels': config.training.channels,
                'samplerate': config.training.samplerate,
                # 'segment': args.model_segment or 4 * args.dset.segment,
                'segment': config.training.segment,
            }

            kw = OmegaConf.to_container(getattr(config, config.model), resolve=True)

            model = HTDemucs(**extra, **kw)

            state_dict = torch.load('./weights/ht_demucs_v4.th', map_location=device, weights_only=False)
            # Fix for htdemucs pretrained models
            if 'state' in state_dict:
                state_dict = state_dict['state']
            # Fix for apollo pretrained models
            if 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']

            model.load_state_dict(state_dict)

            return model
        else:
            raise NotImplementedError("Error! HTDemucs supports only 4s and 6s versions in our app")