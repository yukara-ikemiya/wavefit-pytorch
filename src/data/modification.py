"""
Copyright (C) 2024 Yukara Ikemiya
"""

import math
import random
import typing as tp

import torch
from torch import nn
import numpy as np


# Channels

class Mono(nn.Module):
    def __call__(self, x: torch.Tensor):
        assert len(x.shape) <= 2
        return torch.mean(x, dim=0, keepdims=True) if len(x.shape) > 1 else x


class Stereo(nn.Module):
    def __call__(self, x: torch.Tensor):
        x_shape = x.shape
        assert len(x_shape) <= 2
        # Check if it's mono
        if len(x_shape) == 1:  # s -> 2, s
            x = x.unsqueeze(0).repeat(2, 1)
        elif len(x_shape) == 2:
            if x_shape[0] == 1:  # 1, s -> 2, s
                x = x.repeat(2, 1)
            elif x_shape[0] > 2:  # ?, s -> 2,s
                x = x[:2, :]

        return x


# Augumentation

class PhaseFlipper(nn.Module):
    """Randomly invert the phase of a signal"""

    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def __call__(self, x: torch.Tensor):
        assert len(x.shape) <= 2
        return -x if (random.random() < self.p) else x


class VolumeChanger(nn.Module):
    """Randomly change volume (amplitude) of a signal"""

    def __init__(self, min_db: float = -3., max_db: float = 6.):
        super().__init__()
        self.min_db = min_db
        self.max_db = max_db
        self.rng = np.random.default_rng()

    def __call__(self, x: torch.Tensor):
        assert len(x.shape) <= 2

        max_db = min(self.max_db, 20 * np.log10(1. / (np.abs(x).max() + 1.0e-8)))  # amp <= 1.0
        db = self.rng.uniform(self.min_db, max_db)
        gain = 10 ** (db / 20.)
        return x * gain
