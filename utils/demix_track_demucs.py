import torch
import torch.nn as nn
import numpy as np

from tqdm.auto import tqdm


def demix_track_demucs(config, model, mix, device, pbar=False, progress_bar=None):
    S = len(config.training.instruments)
    C = config.training.samplerate * config.training.segment
    N = config.inference.num_overlap
    batch_size = config.inference.batch_size
    step = C // N

    with torch.cuda.amp.autocast(enabled=config.training.use_amp):
        with torch.inference_mode():
            req_shape = (S,) + tuple(mix.shape)
            result = torch.zeros(req_shape, dtype=torch.float32, device=device)
            counter = torch.zeros(req_shape, dtype=torch.float32, device=device)
            i = 0
            batch_data = []
            batch_locations = []

            total_iterations = (mix.shape[1] + step - 1) // step
            current_iteration = 0

            # Перемещаем mix целиком на устройство один раз
            mix = mix.to(device)

            while i < mix.shape[1]:
                part = mix[:, i:i + C]
                length = part.shape[-1]

                if progress_bar:
                    progress_bar.set(min(1.0, current_iteration / total_iterations))
                    current_iteration += 1

                if length < C:
                    part = nn.functional.pad(input=part, pad=(0, C - length, 0, 0), mode='constant', value=0)
                batch_data.append(part)
                batch_locations.append((i, length))
                i += step

                if len(batch_data) >= batch_size or (i >= mix.shape[1]):
                    arr = torch.stack(batch_data, dim=0)
                    x = model(arr)

                    for j in range(len(batch_locations)):
                        start, l = batch_locations[j]
                        result[..., start:start + l] += x[j][..., :l]
                        counter[..., start:start + l] += 1.

                    # Очищаем переменные для экономии памяти
                    del arr, x, batch_data, batch_locations
                    batch_data = []
                    batch_locations = []

                    # Явно освобождаем кеш
                    if str(device).startswith('mps'):
                        torch.mps.empty_cache()
                    elif str(device).startswith('cuda'):
                        torch.cuda.empty_cache()

            # Выполняем расчеты на том же устройстве
            result = result / counter

            # Переносим на CPU только в самом конце
            estimated_sources = result.cpu().numpy()
            np.nan_to_num(estimated_sources, copy=False, nan=0.0)

    if S > 1:
        return {k: v for k, v in zip(config.training.instruments, estimated_sources)}
    else:
        return estimated_sources