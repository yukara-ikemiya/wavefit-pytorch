"""
Copyright (C) 2024 Yukara Ikemiya
"""

import typing as tp

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.audio_utils import get_amplitude_spec, MelSpectrogram


class MRSTFTLoss(nn.Module):
    """
    Multi-resolution STFT loss corresponding to the eq.(9)
    """

    def __init__(
        self,
        n_ffts: tp.List[int] = [512, 1024, 2048],
        win_sizes: tp.List[int] = [360, 900, 1800],
        hop_sizes: tp.List[int] = [80, 150, 300],
        EPS: float = 1e-5
    ):
        super().__init__()
        assert len(n_ffts) == len(win_sizes) == len(hop_sizes)
        self.n_ffts = n_ffts
        self.win_sizes = win_sizes
        self.hop_sizes = hop_sizes

        # NOTE: Since spectral convergence is quite sensitive to small values in the spectrum,
        #       I believe setting a higher lower bound will result in more stable training.
        self.EPS = EPS

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ):
        losses = {
            'G/mrstft_sc_loss': 0.,
            'G/mrstft_mag_loss': 0.
        }

        for n_fft, win_size, hop_size in zip(self.n_ffts, self.win_sizes, self.hop_sizes):
            window = torch.hann_window(win_size, device=pred.device)
            spec_t = get_amplitude_spec(target.squeeze(1), n_fft, win_size, hop_size, window)
            spec_p = get_amplitude_spec(pred.squeeze(1), n_fft, win_size, hop_size, window)

            # spectral convergence
            sc_loss = (spec_t - spec_p).norm(p=2) / (spec_t.norm(p=2) + self.EPS)

            # magnitude loss
            mag_loss = F.l1_loss(torch.log(spec_t.clamp(min=self.EPS)), torch.log(spec_p.clamp(min=self.EPS)))

            losses['G/mrstft_sc_loss'] += sc_loss
            losses['G/mrstft_mag_loss'] += mag_loss

        losses['G/mrstft_sc_loss'] /= len(self.n_ffts)
        losses['G/mrstft_mag_loss'] /= len(self.n_ffts)

        return losses


class MELMAELoss(nn.Module):
    """
    MAE(L1) loss of Mel spectrogram corresponding to the second term of the eq.(19)
    """

    def __init__(
        self,
        sr: int = 24000,
        n_fft: int = 1024,
        win_size: int = 900,
        hop_size: int = 150,
        n_mels: int = 128,
        fmin: float = 20.,
        fmax: float = 12000.
    ):
        super().__init__()

        self.mel = MelSpectrogram(sr, n_fft, win_size, hop_size, n_mels, fmin, fmax)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ):
        losses = {'G/mel_mae_loss': 0.}

        # Mel MAE (L1) loss
        mel_p = self.mel.compute_mel(pred.squeeze(1))
        mel_t = self.mel.compute_mel(target.squeeze(1))

        losses['G/mel_mae_loss'] = F.l1_loss(mel_t, mel_p)

        return losses
