"""
Copyright (C) 2026 Yukara Ikemiya

Compare model size (parameter count) and computational cost (FLOPs) of
WaveFit and WaveFitNeXt (GAN-WaveNeXt2).

Usage:
    python scripts/compare_model.py
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../src'))

import torch
from torch.utils.flop_counter import FlopCounterMode

from model import WaveFit, WaveFitNeXt
from utils.torch_common import count_parameters

NUM_ITERATION = 3
SAMPLE_RATE = 24000
HOP_SIZE = 300
DURATION_SEC = 1.0


def profile_model(name: str, model: torch.nn.Module, audio: torch.Tensor):
    mel_spec, r_noise, pstft_spec = model.mel.get_shaped_noise(audio)
    log_mel_spec = torch.log(mel_spec.clamp(min=1e-8))

    model.eval()
    with torch.no_grad():
        flop_counter = FlopCounterMode(display=False)
        with flop_counter:
            model(r_noise, log_mel_spec, pstft_spec=pstft_spec)
        flops = flop_counter.get_total_flops()

    model.train()
    n_params = count_parameters(model, include_buffers=False)

    print(
        f"{name:<14} | params: {n_params / 1e6:8.2f} M "
        f"| FLOPs/sec of audio: {flops / DURATION_SEC / 1e9:9.2f} G"
    )


def main():
    n_frame = int(SAMPLE_RATE * DURATION_SEC) // HOP_SIZE
    audio = torch.randn(1, 1, n_frame * HOP_SIZE)

    print(f"Iterations: {NUM_ITERATION} / Input: {DURATION_SEC}s @ {SAMPLE_RATE}Hz\n")

    profile_model("WaveFit", WaveFit(num_iteration=NUM_ITERATION), audio)
    profile_model("WaveFitNeXt", WaveFitNeXt(num_iteration=NUM_ITERATION), audio)


if __name__ == '__main__':
    main()
