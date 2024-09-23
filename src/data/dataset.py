"""
Copyright (C) 2024 Yukara Ikemiya
"""

import os
import random
import typing as tp

import torch
import numpy as np

from utils.torch_common import print_once
from .modification import Stereo, Mono, PhaseFlipper, VolumeChanger
from .audio_io import get_audio_metadata, load_audio_with_pad


def fast_scandir(dir: str, ext: tp.List[str]):
    """ Very fast `glob` alternative. from https://stackoverflow.com/a/59803793/4259243

    fast_scandir implementation by Scott Hawley originally in https://github.com/zqevans/audio-diffusion/blob/main/dataset/dataset.py

    Args:
        dir (str): top-level directory at which to begin scanning.
        ext (tp.List[str]): list of allowed file extensions.
    """
    subfolders, files = [], []
    # add starting period to extensions if needed
    ext = ['.' + x if x[0] != '.' else x for x in ext]

    try:  # hope to avoid 'permission denied' by this try
        for f in os.scandir(dir):
            try:  # 'hope to avoid too many levels of symbolic links' error
                if f.is_dir():
                    subfolders.append(f.path)
                elif f.is_file():
                    is_hidden = os.path.basename(f.path).startswith(".")
                    has_ext = os.path.splitext(f.name)[1].lower() in ext

                    if has_ext and (not is_hidden):
                        files.append(f.path)
            except Exception:
                pass
    except Exception:
        pass

    for dir in list(subfolders):
        sf, f = fast_scandir(dir, ext)
        subfolders.extend(sf)
        files.extend(f)

    return subfolders, files


def get_audio_filenames(
    paths: tp.List[str],  # directories in which to search
    exts: tp.List[str] = ['.wav', '.mp3', '.flac', '.ogg', '.aif', '.opus']
):
    """recursively get a list of audio filenames"""
    if isinstance(paths, str):
        paths = [paths]

    # get a list of relevant filenames
    filenames = []
    nums_file = []
    for p in paths:
        _, files = fast_scandir(p, exts)
        files.sort()
        filenames.extend(files)
        nums_file.append(len(files))

    return filenames, nums_file


class AudioFilesDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dir_list: tp.List[str],
        sample_size: int = 36000,
        sample_rate: int = 24000,
        out_channels="mono",
        exts: tp.List[str] = ['wav'],
        # augmentation
        augment_shift: bool = True,
        augment_flip: bool = True,
        augment_volume: bool = True,
        # Others
        max_samples: tp.Optional[int] = None
    ):
        assert out_channels in ['mono', 'stereo']

        super().__init__()
        self.sample_size = sample_size
        self.sr = sample_rate
        self.augment_shift = augment_shift
        self.out_channels = out_channels

        self.ch_encoding = torch.nn.Sequential(
            Stereo() if self.out_channels == "stereo" else torch.nn.Identity(),
            Mono() if self.out_channels == "mono" else torch.nn.Identity(),
        )

        self.augs = torch.nn.Sequential(
            PhaseFlipper() if augment_flip else torch.nn.Identity(),
            VolumeChanger() if augment_volume else torch.nn.Identity()
        )

        # find all audio files
        print_once('->-> Searching audio files...')
        self.filenames, _ = get_audio_filenames(dir_list, exts=exts)
        # sort
        self.filenames.sort()

        max_samples = max_samples if max_samples else len(self.filenames)
        self.filenames = self.filenames[:max_samples]
        print_once(f'->-> Found {len(self.filenames)} files.')

        # This may take long time if file number is large.
        print_once('->-> Loading audio metadata... (this may take a long time)')
        self.metas = []
        self.durs = []
        for idx, filepath in enumerate(self.filenames):
            info = get_audio_metadata(filepath, cache=True)
            self.metas.append(info)
            self.durs.append(info['num_frames'])

        self.cs_durs = np.cumsum(self.durs)

    def get_index_offset(self, item):
        """
        Return a track index and frame offset
        """
        # For a given dataset item and shift, return song index and offset within song
        half_size = self.sample_size // 2
        shift = np.random.randint(-half_size, half_size) if self.augment_shift else 0
        offset = item * self.sample_size + shift  # Note we centred shifts, so adding now
        midpoint = offset + half_size

        index = np.searchsorted(self.cs_durs, midpoint)
        start, end = self.cs_durs[index - 1] if index > 0 else 0, self.cs_durs[index]  # start and end of current song
        assert start <= midpoint <= end

        if offset > end - self.sample_size:  # Going over song
            offset = max(start, offset - half_size)
        elif offset < start:  # Going under song
            offset = start

        offset -= start
        return index, offset

    def __len__(self):
        return int(np.floor(self.cs_durs[-1] / self.sample_size))

    def __getitem__(self, idx):
        idx_file, offset = self.get_index_offset(idx)
        filename = self.filenames[idx_file]
        info = self.metas[idx_file]

        try:
            audio = load_audio_with_pad(filename, info, self.sr, self.sample_size, offset)

            # Fix channel number
            audio = self.ch_encoding(audio)

            # Audio augmentations
            audio = self.augs(audio)
            audio = audio.clamp(-1, 1)

            return (audio, info)

        except Exception as e:
            print(f'Couldn\'t load file {filename} (INFO: {info}, offset: {offset}): {e}')
            return self[random.randrange(len(self))]
