"""
Copyright (C) 2024 Yukara Ikemiya
"""
import sys
sys.dont_write_bytecode = True

import torch
import torchaudio
import hydra
from omegaconf import DictConfig, OmegaConf

from utils.audio_utils import MelSpectrogram
from model import Generator
from utils.torch_common import count_parameters


@hydra.main(version_base=None, config_path='../configs/', config_name="default.yaml")
def main(cfg: DictConfig):

    sr = 24000
    n_fft = 2048
    win_size = 1200
    hop_size = 300
    n_mels = 128

    train_dataset = hydra.utils.instantiate(cfg.data.train)
    test_dataset = hydra.utils.instantiate(cfg.data.train)

    print(len(train_dataset), len(test_dataset))

    mel_module = MelSpectrogram(
        sr=sr, n_fft=n_fft, win_size=win_size, hop_size=hop_size,
        n_mels=n_mels, fmin=20., fmax=12000.)

    data, _ = test_dataset[1]
    print(data.shape)
    data = data.mean(0).unsqueeze(0)  # (1, n_samples)
    print(data.shape)

    mel_spec = mel_module.compute_mel(data)
    print(mel_spec.shape)

    spec_env = mel_module.get_spec_env_from_mel(mel_spec, cep_order=24, return_min_phase=True)
    print(f"Spec_env: {spec_env.shape}")

    # noise reshape test
    noise = torch.randn(*data.shape, device=data.device)

    print(f"Input audio std: {data.std().item()}")
    print(f"noise(before) std: {noise.std().item()}")
    print(f"noise(before): max: {noise.max().item()}, min: {noise.min().item()}")

    num_frame = mel_spec.shape[-1]
    fft_win = mel_module.fft_win
    noise_spec = torch.stft(
        noise, n_fft, hop_length=hop_size, win_length=win_size, window=fft_win,
        center=True, normalized=False, onesided=True, return_complex=True)

    noise_spec = noise_spec[..., :num_frame]
    print(noise_spec.shape, spec_env.shape)
    assert noise_spec.shape == spec_env.shape

    noise_spec *= spec_env

    r_noise = torch.istft(
        noise_spec, n_fft, hop_length=hop_size, win_length=win_size, window=fft_win,
        center=True, normalized=False, length=data.shape[-1])

    print(r_noise.shape)
    print(f"noise(after) std: {r_noise.std().item()}")
    print(f"noise(after): max: {r_noise.max().item()}, min: {r_noise.min().item()}")

    torchaudio.save("src.wav", data, sample_rate=sr, encoding="PCM_F")
    torchaudio.save("r_noise.wav", r_noise, sample_rate=sr, encoding="PCM_F")

    wavefit = Generator()
    wavefit.train()
    num_params_d = count_parameters(wavefit.downsample)
    num_params_u = count_parameters(wavefit.upsample)
    num_params_f = count_parameters(wavefit.film)
    print(f"Num params : {num_params_d + num_params_u + num_params_f}")
    print(f"Num params (down) : {num_params_d}")
    print(f"Num params (up) : {num_params_u}")
    print(f"Num params (film) : {num_params_f}")


if __name__ == '__main__':
    main()
