"""
Copyright (C) 2024 Yukara Ikemiya
"""

import os
import sys
sys.dont_write_bytecode = True
import argparse
import math

import hydra
import torch
import torchaudio
from accelerate import Accelerator
from omegaconf import DictConfig, OmegaConf

from utils.torch_common import get_rank, get_world_size, print_once
from data.dataset import get_audio_filenames


def make_audio_batch(audio, sample_size: int, overlap: int):
    """
    audio : (ch, L)
    """
    assert 0 <= overlap < sample_size
    L = audio.shape[-1]
    shift = sample_size - overlap

    n_split = math.ceil(max(L - sample_size, 0) / shift) + 1
    # to mono
    audio = audio.mean(0)  # (L)
    batch = []
    for n in range(n_split):
        b = audio[n * shift: n * shift + sample_size]
        if n == n_split - 1:
            b = torch.nn.functional.pad(b, (0, sample_size - len(b)))
        batch.append(b)

    batch = torch.stack(batch, dim=0).unsqueeze(1)  # (n_split, 1, sample_size)
    return batch, L


def cross_fade(preds, overlap: int, L: int):
    """
    preds: (bs, 1, sample_size)
    """
    bs, _, sample_size = preds.shape
    shift = sample_size - overlap
    full_L = sample_size + (bs - 1) * shift
    win = torch.bartlett_window(overlap * 2, device=preds.device)

    buf = torch.zeros(1, full_L, device=preds.device)
    for idx in range(bs):
        pred = preds[idx]  # (1, sample_size)
        ofs = idx * shift
        if idx != 0:
            pred[:, :overlap] *= win[None, :overlap]
        if idx != bs - 1:
            pred[:, -overlap:] *= win[None, overlap:]

        buf[:, ofs:ofs + sample_size] += pred

    buf = buf[..., :L]
    return buf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt-dir', type=str, help="Checkpoint directory.")
    parser.add_argument('--input-audio-dir', type=str, help="Root directory which contains input audio files.")
    parser.add_argument('--output-dir', type=str, help="Output directory.")
    parser.add_argument('--sample-size', type=int, default=36000, help="Input sample size.")
    parser.add_argument('--max-batch-size', type=int, default=10, help="Max batch size for inference.")
    parser.add_argument('--overlap-rate', type=float, default=0.02, help="Overlap rate for inference.")
    parser.add_argument('--use-original-name', default=True, type=bool, help="Whether to use an original file name as an output name.")
    args = parser.parse_args()

    ckpt_dir = args.ckpt_dir
    input_audio_dir = args.input_audio_dir
    output_dir = args.output_dir
    sample_size = args.sample_size
    max_batch_size = args.max_batch_size
    overlap_rate = args.overlap_rate
    use_original_name = args.use_original_name

    # Distributed inference
    accel = Accelerator()
    device = accel.device
    rank = get_rank()
    world_size = get_world_size()

    print_once(f"Checkpoint dir  : {ckpt_dir}")
    print_once(f"Input audio dir : {input_audio_dir}")
    print_once(f"Output dir      : {output_dir}")

    # Load WaveFit model
    cfg_ckpt = OmegaConf.load(f'{ckpt_dir}/config.yaml')
    wavefit = hydra.utils.instantiate(cfg_ckpt.model.generator.model)
    wavefit.load_state_dict(torch.load(f"{ckpt_dir}/generator.pth", weights_only=False))
    wavefit.to(device)
    wavefit.eval()
    print_once("->-> Successfully loaded WaveFit model from checkpoint.")

    hop_size = wavefit.args_mel['hop_size']
    overlap = math.ceil(sample_size * overlap_rate / hop_size) * hop_size

    # Get audio files
    files, _ = get_audio_filenames(input_audio_dir)
    print_once(f"->-> Found {len(files)} audio files.")
    # Split files
    files = files[rank::world_size]

    for idx, f_path in enumerate(files):
        # load and split audio
        audio, sr = torchaudio.load(f_path)
        audio_batch, L = make_audio_batch(audio, sample_size, overlap)
        n_iter = math.ceil(audio_batch.shape[0] / max_batch_size)

        audio_batch = audio_batch.to(device)

        # execute
        preds = []
        for n in range(n_iter):
            batch_ = audio_batch[n * max_batch_size:(n + 1) * max_batch_size]
            with torch.no_grad():
                mel_spec, r_noise, pstft_spec = wavefit.mel.get_shaped_noise(batch_)
                pred = wavefit(r_noise, torch.log(mel_spec.clamp(min=1e-8)), pstft_spec, return_only_last=True)[-1]

            preds.append(pred)

        preds = torch.cat(preds, dim=0)

        # cross-fade
        pred_audio = cross_fade(preds, overlap, L).cpu()

        # save audio
        out_name = os.path.splitext(os.path.basename(f_path))[0] if use_original_name else f"sample_{idx}"
        out_path = f"{output_dir}/{out_name}.wav"
        torchaudio.save(out_path, pred_audio, sample_rate=sr, encoding="PCM_F")

    print(f"--- Rank-{rank} : Finished. ---")


if __name__ == '__main__':
    main()
