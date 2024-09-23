"""
Copyright (C) 2024 Yukara Ikemiya

Adapted from the following repo under Apache License 2.0.
https://github.com/drscotthawley/aeiou/
"""

from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg


def spectrogram_image(
    spec,
    db_range=[-45, 8],
    figsize=(6, 3),  # size of plot (if justimage==False)
):
    from librosa import power_to_db

    fig = plt.figure(figsize=figsize, dpi=100)
    canvas = FigureCanvasAgg(fig)
    axs = fig.add_subplot()
    spec = spec.squeeze()
    im = axs.imshow(power_to_db(spec), origin='lower', aspect='auto', vmin=db_range[0], vmax=db_range[1])

    axs.axis('off')
    plt.tight_layout()

    canvas.draw()
    rgba = np.asarray(canvas.buffer_rgba())
    im = Image.fromarray(rgba)

    b = 15  # border size
    im = im.crop((b, b, im.size[0] - b, im.size[1] - b))

    plt.clf()
    plt.close()

    return im
