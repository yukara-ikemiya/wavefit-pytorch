"""
Copyright (C) 2024 Yukara Ikemiya

Adapted from the following repo's code under MIT License.
https://github.com/descriptinc/melgan-neurips/
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def WNConv1d(*args, **kwargs):
    return weight_norm(nn.Conv1d(*args, **kwargs))


class NLayerDiscriminator(nn.Module):
    def __init__(self, ndf, n_layers, downsampling_factor):
        super().__init__()
        model = nn.ModuleDict()

        model["layer_0"] = nn.Sequential(
            nn.ReflectionPad1d(7),
            WNConv1d(1, ndf, kernel_size=15),
            nn.LeakyReLU(0.2, True),
        )

        nf = ndf
        stride = downsampling_factor
        for n in range(1, n_layers + 1):
            nf_prev = nf
            nf = min(nf * stride, 1024)

            model["layer_%d" % n] = nn.Sequential(
                WNConv1d(
                    nf_prev,
                    nf,
                    kernel_size=stride * 10 + 1,
                    stride=stride,
                    padding=stride * 5,
                    groups=nf_prev // 4,
                ),
                nn.LeakyReLU(0.2, True),
            )

        nf = min(nf * 2, 1024)
        model["layer_%d" % (n_layers + 1)] = nn.Sequential(
            WNConv1d(nf_prev, nf, kernel_size=5, stride=1, padding=2),
            nn.LeakyReLU(0.2, True),
        )

        model["layer_%d" % (n_layers + 2)] = WNConv1d(
            nf, 1, kernel_size=3, stride=1, padding=1
        )

        self.model = model

    def forward(self, x: torch.Tensor, return_feature: bool = True):
        """
        Args:
            x: input audio, (bs, 1, L)
        """
        n_layer = len(self.model)
        results = []
        for idx, (key, layer) in enumerate(self.model.items()):
            x = layer(x)
            if return_feature or (idx == n_layer - 1):
                results.append(x)

        return results


class Discriminator(nn.Module):
    def __init__(
        self,
        num_D: int = 3,
        ndf: int = 16,
        n_layers: int = 4,
        downsampling_factor: int = 4
    ):
        super().__init__()
        self.model = nn.ModuleDict()
        for i in range(num_D):
            self.model[f"disc_{i}"] = NLayerDiscriminator(
                ndf, n_layers, downsampling_factor
            )

        self.downsample = nn.AvgPool1d(4, stride=2, padding=1, count_include_pad=False)
        self.apply(weights_init)

    def forward(self, x: torch.Tensor, return_feature: bool = True):
        """
        Args:
            x: input audio, (bs, 1, L)
        """
        results = []
        for key, disc in self.model.items():
            results.append(disc(x, return_feature))
            x = self.downsample(x)

        return results

    def compute_G_loss(self, x_fake, x_real):
        """
        The eq.(18) loss
        """
        assert x_fake.shape == x_real.shape

        out_f = self(x_fake, return_feature=True)
        with torch.no_grad():
            out_r = self(x_real, return_feature=True)

        num_D = len(self.model)
        losses = {
            'G/disc_gan_loss': 0.,
            'G/disc_feat_loss': 0.
        }

        for i_d in range(num_D):
            n_layer = len(out_f[i_d])

            # GAN loss
            losses['G/disc_gan_loss'] += (1 - out_f[i_d][-1]).relu().mean()

            # Feature-matching loss
            # eq.(8)
            feat_loss = 0.
            for i_l in range(n_layer - 1):
                feat_loss += F.l1_loss(out_f[i_d][i_l], out_r[i_d][i_l])

            losses['G/disc_feat_loss'] += feat_loss / (n_layer - 1)

        losses['G/disc_gan_loss'] /= num_D
        losses['G/disc_feat_loss'] /= num_D

        return losses

    def compute_D_loss(self, x, mode: str):
        """
        The eq.(7) loss
        """
        assert mode in ['fake', 'real']
        sign = 1 if mode == 'fake' else -1

        out = self(x, return_feature=False)

        num_D = len(self.model)
        losses = {'D/loss': 0.}

        for i_d in range(num_D):
            # Hinge loss
            losses['D/loss'] += (1 + sign * out[i_d][-1]).relu().mean()

        losses['D/loss'] /= num_D

        return losses
