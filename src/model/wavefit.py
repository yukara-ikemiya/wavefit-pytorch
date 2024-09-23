"""
Copyright (C) 2024 Yukara Ikemiya
"""

import torch
import torch.nn as nn

from utils.audio_utils import MelSpectrogram, get_amplitude_spec
from model import Generator


class WaveFit(nn.Module):
    def __init__(
        self,
        num_iteration: int,
        args_mel: dict = {
            'sr': 24000,
            'n_fft': 2048,
            'win_size': 1200,
            'hop_size': 300,
            'n_mels': 128,
            'fmin': 20.,
            'fmax': 12000.
        }
    ):
        super().__init__()

        self.T = num_iteration
        self.args_mel = args_mel
        self.mel = MelSpectrogram(**args_mel)
        self.generator = Generator(num_iteration)
        self.EPS = 1e-8

    def forward(
        self,
        initial_noise: torch.Tensor,
        log_mel_spec: torch.Tensor,
        pstft_spec: torch.Tensor,
        # You can use this option at inference time
        return_only_last: bool = False
    ):
        """
        Args:
            initial_noise: Initial noise, (bs, 1, L).
            log_mel_spec: Log Mel spectrogram, (bs, n_mels, L//hop_size).
            pstft_spec: Pseudo spectrogram used for gain adjustment.
            return_only_last: If true, only the last output (y_0) is returned.
        Returns:
            preds: List of predictions (y_t)
        """
        assert initial_noise.dim() == log_mel_spec.dim() == 3
        assert initial_noise.shape[-1] == log_mel_spec.shape[-1] * self.args_mel['hop_size']

        preds = []
        y_t = initial_noise
        for t in range(self.T):
            # estimate noise
            est = self.generator(y_t, log_mel_spec, t)
            y_t = y_t - est

            # adjust gain
            y_t = self.adjust_gain(y_t, pstft_spec)

            if (not return_only_last) or (t == self.T - 1):
                preds.append(y_t)

            # To avoid gradient loop
            y_t = y_t.detach()

        return preds

    def adjust_gain(self, z_t, pstft_spec):
        num_frame = pstft_spec.shape[-1]
        power_spec_z = get_amplitude_spec(
            z_t.squeeze(1), self.args_mel['n_fft'], self.args_mel['win_size'],
            self.args_mel['hop_size'], self.mel.fft_win, return_power=True
        )[..., :num_frame]

        assert power_spec_z.shape == pstft_spec.shape
        pow_z = power_spec_z.mean(dim=[1, 2])
        pow_c = pstft_spec.pow(2).mean(dim=[1, 2])

        return z_t * torch.sqrt(pow_c[:, None, None] / (pow_z[:, None, None] + self.EPS))
