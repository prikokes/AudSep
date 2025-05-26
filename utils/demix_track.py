import torch
import torch.nn as nn
import numpy as np

from ml_collections import ConfigDict
from typing import List

from tqdm.auto import tqdm


def demix_track(config, model, mix, device, pbar=False, progress_bar=None):
    C = config.audio.chunk_size
    N = config.inference.num_overlap
    fade_size = C // 10
    step = int(C // N)
    border = C - step
    batch_size = config.inference.batch_size

    length_init = mix.shape[-1]

    # Do pad from the beginning and end to account floating window results better
    if length_init > 2 * border and (border > 0):
        mix = nn.functional.pad(mix, (border, border), mode='reflect')

    # windowingArray crossfades at segment boundaries to mitigate clicking artifacts
    windowingArray = _getWindowingArray(C, fade_size)

    with torch.cuda.amp.autocast(enabled=config.training.use_amp):
        with torch.inference_mode():
            req_shape = (len(prefer_target_instrument(config)),) + tuple(mix.shape)

            result = torch.zeros(req_shape, dtype=torch.float32)
            counter = torch.zeros(req_shape, dtype=torch.float32)
            i = 0
            batch_data = []
            batch_locations = []

            total_iterations = (mix.shape[1] + step - 1) // step
            current_iteration = 0

            while i < mix.shape[1]:
                # print(i, i + C, mix.shape[1])
                if progress_bar:
                    progress_bar.update_progress(min(100.0, int((current_iteration / total_iterations) * 100)))
                    current_iteration += 1

                part = mix[:, i:i + C].to(device)
                length = part.shape[-1]
                if length < C:
                    if length > C // 2 + 1:
                        part = nn.functional.pad(input=part, pad=(0, C - length), mode='reflect')
                    else:
                        part = nn.functional.pad(input=part, pad=(0, C - length, 0, 0), mode='constant', value=0)
                batch_data.append(part)
                batch_locations.append((i, length))
                i += step

                print(i)

                if len(batch_data) >= batch_size or (i >= mix.shape[1]):
                    arr = torch.stack(batch_data, dim=0)
                    x = model(arr)

                    window = windowingArray
                    if i - step == 0:  # First audio chunk, no fadein
                        window[:fade_size] = 1
                    elif i >= mix.shape[1]:  # Last audio chunk, no fadeout
                        window[-fade_size:] = 1

                    for j in range(len(batch_locations)):
                        start, l = batch_locations[j]
                        result[..., start:start+l] += x[j][..., :l].cpu() * window[..., :l]
                        counter[..., start:start+l] += window[..., :l]

                    batch_data = []
                    batch_locations = []

            estimated_sources = result / counter
            estimated_sources = estimated_sources.cpu().numpy()
            np.nan_to_num(estimated_sources, copy=False, nan=0.0)

            if length_init > 2 * border and (border > 0):
                # Remove pad
                estimated_sources = estimated_sources[..., border:-border]

    return {k: v for k, v in zip(prefer_target_instrument(config), estimated_sources)}


def _getWindowingArray(window_size, fade_size):
    fadein = torch.linspace(0, 1, fade_size)
    fadeout = torch.linspace(1, 0, fade_size)
    window = torch.ones(window_size)
    window[-fade_size:] *= fadeout
    window[:fade_size] *= fadein
    return window

def prefer_target_instrument(config: ConfigDict) -> List[str]:
    if config.training.get('target_instrument'):
        return [config.training.target_instrument]
    else:
        return config.training.instruments
