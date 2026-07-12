"""
Copyright (C) 2026 Yukara Ikemiya
"""

import torch
import torch.nn as nn

from utils.audio_utils import MelSpectrogram, get_amplitude_spec


class LayerNorm1d(nn.Module):
    """LayerNorm applied over the channel dimension of a (bs, C, T) tensor."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=eps)

    def forward(self, x: torch.Tensor):
        return self.norm(x.transpose(1, 2)).transpose(1, 2)


class ConvNeXtBlock1d(nn.Module):
    """
    1D ConvNeXt block used to build the WaveNeXt-based generator (Fig.1(a) of the WaveNeXt2 paper).
    Adapted from the ConvNeXt architecture (Liu et al., CVPR 2022) for 1D signals.
    """

    def __init__(
        self,
        dim: int,
        intermediate_dim: int,
        kernel_size: int = 7,
        layer_scale_init_value: float = 1e-6
    ):
        super().__init__()
        self.dwconv = nn.Conv1d(dim, dim, kernel_size, padding=kernel_size // 2, groups=dim)
        self.norm = LayerNorm1d(dim)
        self.pwconv1 = nn.Linear(dim, intermediate_dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(intermediate_dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones(dim))

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (bs, dim, T)
        """
        residual = x

        x = self.dwconv(x)
        x = self.norm(x)
        x = x.transpose(1, 2)          # (bs, T, dim)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = self.gamma * x
        x = x.transpose(1, 2)          # (bs, dim, T)

        return residual + x


class WaveNeXtGenerator(nn.Module):
    """
    WaveNeXt-based generator used as one sub-model of WaveFitNeXt (Fig.1 of the WaveNeXt2 paper).

    Inputs:
        log_mel_spec: (bs, n_mels, num_frame)
        y_prev: Waveform predicted by the previous sub-model, (bs, 1, num_frame x hop_size)
        fft_win: Hann window used for the STFT module, (win_size,)
    Outputs:
        n_hat: Estimated residual noise, (bs, 1, num_frame x hop_size)
    """

    def __init__(
        self,
        n_mels: int,
        hop_size: int,
        n_fft: int,
        hidden_dim: int = 512,
        intermediate_dim: int = 1536,
        num_layers: int = 8,
        kernel_size: int = 7
    ):
        super().__init__()
        self.hop_size = hop_size
        self.n_fft = n_fft

        # STFT-spec channels = full real part (n_fft//2+1 bins) concatenated with the
        # imaginary part with its DC/Nyquist bins dropped (n_fft//2-1 bins) -> n_fft total.
        n_stft_channels = n_fft

        self.spec_proj = nn.Conv1d(n_mels + n_stft_channels, hidden_dim, 3, padding=1)
        self.input_norm = LayerNorm1d(hidden_dim)
        self.blocks = nn.ModuleList([
            ConvNeXtBlock1d(hidden_dim, intermediate_dim, kernel_size)
            for _ in range(num_layers)
        ])
        self.final_norm = LayerNorm1d(hidden_dim)
        self.linear1 = nn.Linear(hidden_dim, n_fft)
        self.linear2 = nn.Linear(n_fft, hop_size, bias=False)

        self.register_buffer('fft_win', torch.hann_window(n_fft))

    def forward(
        self,
        log_mel_spec: torch.Tensor,
        y_prev: torch.Tensor
    ):
        num_frame = log_mel_spec.shape[-1]

        stft_spec = torch.stft(
            y_prev.squeeze(1), self.n_fft, hop_length=self.hop_size, win_length=self.n_fft,
            window=self.fft_win, center=True, normalized=False, onesided=True, return_complex=True
        )[..., :num_frame]

        stft_feat = torch.cat([stft_spec.real, stft_spec.imag[:, 1:-1, :]], dim=1)

        # input projection
        x = self.spec_proj(torch.cat([log_mel_spec, stft_feat], dim=1))

        # WaveNeXt blocks
        x = self.input_norm(x)
        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x)

        x = x.transpose(1, 2)                  # (bs, num_frame, hidden_dim)
        # output projection
        x = self.linear2(self.linear1(x))  # (bs, num_frame, hop_size)
        # reshape to waveform
        n_hat = x.reshape(x.shape[0], 1, num_frame * self.hop_size)

        return n_hat


class WaveFitNeXt(nn.Module):
    """
    GAN-WaveNeXt2: a fast neural vocoder that combines the fixed-point-iteration based
    residual denoising of WaveFit with the iSTFT-free WaveNeXt generator, trading
    model size for a large speed-up of both training and inference.

    NOTE: Unlike WaveFit, by default, no initial noise input nor gain-adjustment module is used;
    iteration simply starts from silence (Sec.3.2 of the WaveNeXt2 paper).
    """

    def __init__(
        self,
        num_iteration: int,
        n_fft_stft: int = 1024,
        args_mel: dict = {
            'sr': 24000,
            'n_fft': 2048,
            'win_size': 1200,
            'hop_size': 300,
            'n_mels': 128,
            'fmin': 20.,
            'fmax': 12000.
        },
        generator_args: dict = {
            'hidden_dim': 512,
            'intermediate_dim': 1536,
            'num_layers': 8,
            'kernel_size': 7
        },
        # additional options
        use_noise_input: bool = False,
        use_gain_adjustment: bool = False
    ):
        super().__init__()

        self.T = num_iteration
        self.args_mel = args_mel
        self.mel = MelSpectrogram(**args_mel)
        self.generator = nn.ModuleList([
            WaveNeXtGenerator(
                n_mels=args_mel['n_mels'], hop_size=args_mel['hop_size'],
                n_fft=n_fft_stft, **generator_args
            )
            for _ in range(num_iteration)
        ])

        self.use_noise_input = use_noise_input
        self.use_gain_adjustment = use_gain_adjustment

    def forward(
        self,
        initial_noise: torch.Tensor,
        log_mel_spec: torch.Tensor,
        pstft_spec: torch.Tensor = None,
        # You can use this option at inference time
        return_only_last: bool = False
    ):
        """
        Args:
            initial_noise: (bs, 1, L). GAN-WaveNeXt2 starts from silence rather than
                noise by default, but you can still provide it if needed.
            log_mel_spec: Log Mel spectrogram, (bs, n_mels, L//hop_size).
            pstft_spec: Unused by default. Kept as the 3rd positional argument only so that this model
                shares the exact same call signature as WaveFit (`trainer.py` and
                `inference.py` call it positionally).
            return_only_last: If true, only the last output (y_0) is returned.
        Returns:
            preds: List of predictions (y_t)
        """
        assert initial_noise.dim() == log_mel_spec.dim() == 3
        assert initial_noise.shape[-1] == log_mel_spec.shape[-1] * self.args_mel['hop_size']

        if self.use_noise_input:
            y_t = initial_noise
        else:
            # silence input with small noise to avoid instability
            y_t = torch.zeros_like(initial_noise) + 1e-8 * torch.randn_like(initial_noise)

        preds = []
        for t, generator in enumerate(self.generator):
            # directly estimate speech signal
            y_t = generator(log_mel_spec, y_t)

            if self.use_gain_adjustment:
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
