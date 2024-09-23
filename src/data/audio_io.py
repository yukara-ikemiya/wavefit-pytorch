"""
Copyright (C) 2024 Yukara Ikemiya
"""

import math
import json

from torch.nn import functional as F
import torchaudio
from torchaudio import transforms as T


def get_audio_metadata(filepath, cache=True):
    try:
        with open(filepath + '.json', 'r') as f:
            info = json.load(f)
        return info
    except Exception:
        try:
            info_ = torchaudio.info(filepath)
            sample_rate = info_.sample_rate
            num_channels = info_.num_channels
            num_frames = info_.num_frames

            info = {
                'sample_rate': sample_rate,
                'num_frames': num_frames,
                'num_channels': num_channels
            }
        except Exception:
            # error : cannot open an audio file
            info = {'sample_rate': 0, 'num_frames': 0, 'num_channels': 0}

        if cache:
            with open(filepath + '.json', 'w') as f:
                json.dump(info, f, indent=2)

        return info


def load_audio_with_pad(filepath, info: dict, sr: int, n_samples: int, offset: int):
    sr_in, num_frames = info['sample_rate'], info['num_frames']
    n_samples_in = int(math.ceil(n_samples * (sr_in / sr)))

    # load audio
    ext = filepath.split(".")[-1]
    out_frames = min(n_samples_in, num_frames - offset)

    audio, _ = torchaudio.load(
        filepath, frame_offset=offset, num_frames=out_frames,
        format=ext, backend='soundfile')

    # resample
    if sr_in != sr:
        resample_tf = T.Resample(sr_in, sr)
        audio = resample_tf(audio)[..., :n_samples]

    # zero pad
    L = audio.shape[-1]
    if L < n_samples:
        audio = F.pad(audio, (0, n_samples - L), value=0.)

    return audio
