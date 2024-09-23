"""
Copyright (C) 2024 Yukara Ikemiya
"""

import math

import torch
from torch import nn
import torch.nn.functional as F
from librosa.filters import mel as librosa_mel

EPS = 1e-10


def get_min_phase_filter(amplitude):
    """
    Adapted from the following repo's code.
    https://github.com/SpecDiff-GAN/
    """
    def concat_negative_freq(tensor):
        return torch.concat((tensor[..., :-1], tensor[..., 1:].flip(dims=(-1,))), -1)

    device = amplitude.device
    rank = amplitude.ndim
    num_bins = amplitude.shape[-1]
    amplitude = concat_negative_freq(amplitude)

    fftsize = (num_bins - 1) * 2
    m0 = torch.zeros((fftsize // 2 - 1,), dtype=torch.complex64, device=device)
    m1 = torch.ones((1,), dtype=torch.complex64, device=device)
    m2 = torch.ones((fftsize // 2 - 1,), dtype=torch.complex64, device=device) * 2.0
    minimum_phase_window = torch.concat([m1, m2, m1, m0], axis=0)

    if rank > 1:
        new_shape = [1] * (rank - 1) + [fftsize]
        minimum_phase_window = torch.reshape(minimum_phase_window, new_shape)

    cepstrum = torch.fft.ifft(torch.log(amplitude).to(torch.complex64))
    windowed_cepstrum = cepstrum * minimum_phase_window
    imag_phase = torch.imag(torch.fft.fft(windowed_cepstrum))
    phase = torch.exp(torch.complex(imag_phase * 0.0, imag_phase))
    minimum_phase = amplitude.to(torch.complex64) * phase
    return minimum_phase[..., :num_bins]


def get_amplitude_spec(x, n_fft, win_size, hop_size, window, return_power: bool = False):
    stft_spec = torch.stft(
        x, n_fft, hop_length=hop_size, win_length=win_size, window=window,
        center=True, normalized=False, onesided=True, return_complex=True)

    power_spec = torch.view_as_real(stft_spec).pow(2).sum(-1)

    return power_spec if return_power else torch.sqrt(power_spec + EPS)


class MelSpectrogram(nn.Module):
    def __init__(
        self,
        sr: int,
        # STFT setting
        n_fft: int, win_size: int, hop_size: int,
        # MelSpec setting
        n_mels: int, fmin: float, fmax: float,
    ):
        super().__init__()

        self.sr = sr
        self.n_fft = n_fft
        self.win_size = win_size
        self.hop_size = hop_size
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax

        mel_basis = librosa_mel(sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax)
        mel_basis = torch.from_numpy(mel_basis).float()
        mel_inv_basis = torch.linalg.pinv(mel_basis)

        self.register_buffer('fft_win', torch.hann_window(win_size))
        self.register_buffer('mel_basis', mel_basis)
        self.register_buffer('mel_inv_basis', mel_inv_basis)

    def compute_mel(self, x: torch.Tensor):
        """
        Compute Mel-spectrogram.

        Args:
            x: time_signal, (bs, length)
        Returns:
            mel_spec: Mel spectrogram, (bs, n_mels, num_frame)
        """
        assert x.dim() == 2
        L = x.shape[-1]
        # NOTE : To prevent different signal length in the final frame of the STFT between training and inference time,
        #        input signal length must be a multiple of hop_size.
        assert L % self.hop_size == 0, f"Input signal length must be a multiple of hop_size {self.hop_size}." + \
            f"Input shape -> {x.shape}"

        num_frame = L // self.hop_size

        # STFT
        stft_spec = get_amplitude_spec(x, self.n_fft, self.win_size, self.hop_size, self.fft_win)

        # Mel Spec
        mel_spec = torch.matmul(self.mel_basis, stft_spec)

        # NOTE : The last frame is removed here.
        #   When using center=True setting, output from torch.stft has frame length of (L//hopsize+1).
        #   For training WaveGrad-based architecture, the frame length must be (L//hopsize).
        #   There might be a better way, but I believe this has little to no impact on training
        #   since the whole signal information is contained in the previous frames even when removing the last one.
        mel_spec = mel_spec[..., :num_frame]

        return mel_spec

    def get_spec_env_from_mel(
        self,
        mel_spec: torch.Tensor,
        cep_order: int = 24,
        min_clamp: float = 1e-5,
        return_min_phase: bool = True,
        return_pseudo_stft: bool = False
    ):
        """
        Get spectral envelope from Mel spectrogram

        Args:
            mel_spec: Mel spectrogram, (bs, n_mels, num_frame)
            cep_order: Order of cepstrum lifter
            return_min_phase: If true, minimum phase filter (complex) is returned.
                If false, amplitude spectrum envelope (float) is returned.

        Returns:
            spec_env: Spectral envelope, (bs, binsize, num_frame)
        """
        # pseudo-inverse
        pstft_spec = torch.matmul(self.mel_inv_basis, mel_spec)  # (bs, n_fft/2 + 1, num_frame)

        # n_fft should be power of 2
        binsize = pstft_spec.shape[-2]
        n_fft = int(2 ** math.floor(math.log2(binsize)))

        # cepstrum
        cepstrum = torch.fft.ifft(torch.log(torch.clamp(pstft_spec[..., :n_fft, :], min=min_clamp)).to(torch.complex64), dim=-2)
        cepstrum[..., cep_order:, :] = 0

        # spectral envelope
        spec_env = torch.exp(torch.real(torch.fft.fft(cepstrum, dim=-2)))
        spec_env = F.pad(spec_env, (0, 0, 0, binsize - n_fft))  # zero-pad

        if return_min_phase:
            spec_env = get_min_phase_filter(torch.clamp(spec_env.transpose(-2, -1), min=min_clamp)).transpose(-2, -1)

        return (spec_env, pstft_spec) if return_pseudo_stft else spec_env

    def get_shaped_noise(self, audios: torch.Tensor):
        """
        Get inputs for WaveFit training including spectral-envelope shaped noise

        Args:
            audios: Target audio signal, (bs, 1, L)
        """
        # mel-spectrogram
        mel_spec = self.compute_mel(audios.squeeze(1))
        # spectral envelope
        spec_env, pstft_spec = self.get_spec_env_from_mel(
            mel_spec, cep_order=24, return_min_phase=True, return_pseudo_stft=True)
        # prepare noise
        num_frame = mel_spec.shape[-1]
        noise = torch.randn(*audios.shape, device=audios.device)
        noise_spec = torch.stft(
            noise.squeeze(1), self.n_fft, hop_length=self.hop_size,
            win_length=self.win_size, window=self.fft_win,
            center=True, normalized=False, onesided=True, return_complex=True)[..., :num_frame]
        # shaping
        noise_spec *= spec_env
        r_noise = torch.istft(
            noise_spec, self.n_fft, hop_length=self.hop_size,
            win_length=self.win_size, window=self.fft_win,
            center=True, normalized=False, length=audios.shape[-1])
        r_noise = r_noise.unsqueeze(1)

        return mel_spec, r_noise, pstft_spec
