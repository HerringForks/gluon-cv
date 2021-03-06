import random
from math import sqrt

import mxnet as mx
import mxnet.ndarray as nd

from modules import *
# pylint: disable-all

class Generator(nn.HybridBlock):
    def __init__(self, fused=True):
        super().__init__()

        self.progression = nn.HybridSequential()
        with self.progression.name_scope():
            self.progression.add(StyledConvBlock(512, 512, 3, 1, initial=True))  # 4
            self.progression.add(StyledConvBlock(512, 512, 3, 1, upsample=True))  # 8
            self.progression.add(StyledConvBlock(512, 512, 3, 1, upsample=True))  # 16
            self.progression.add(StyledConvBlock(512, 512, 3, 1, upsample=True))  # 32
            self.progression.add(StyledConvBlock(512, 256, 3, 1, upsample=True))  # 64
            self.progression.add(StyledConvBlock(256, 128, 3, 1, upsample=True, fused=fused))  # 128
            self.progression.add(StyledConvBlock(128, 64, 3, 1, upsample=True, fused=fused))  # 256
            self.progression.add(StyledConvBlock(64, 32, 3, 1, upsample=True, fused=fused))  # 512
            self.progression.add(StyledConvBlock(32, 16, 3, 1, upsample=True, fused=fused))  # 1024


        self.to_rgb = nn.HybridSequential()
        with self.to_rgb.name_scope():
            self.to_rgb.add(EqualConv2d(512, 3, 1))
            self.to_rgb.add(EqualConv2d(512, 3, 1))
            self.to_rgb.add(EqualConv2d(512, 3, 1))
            self.to_rgb.add(EqualConv2d(512, 3, 1))
            self.to_rgb.add(EqualConv2d(256, 3, 1))
            self.to_rgb.add(EqualConv2d(128, 3, 1))
            self.to_rgb.add(EqualConv2d(64, 3, 1))
            self.to_rgb.add(EqualConv2d(32, 3, 1))
            self.to_rgb.add(EqualConv2d(16, 3, 1))

    def hybrid_forward(self, F, style, noise, step=0, alpha=-1, mixing_range=(-1, -1)):

        out = nd.array(noise[0], ctx=style[0].context)

        if style.shape[0] < 2:
            inject_index = [len(self.progression) + 1]

        else:
            inject_index = random.sample(list(range(step)), style.shape[0] - 1)

        crossover = 0

        for i, (conv, to_rgb) in enumerate(zip(self.progression, self.to_rgb)):
            if mixing_range == (-1, -1):
                if crossover < len(inject_index) and i > inject_index[crossover]:
                    crossover = min(crossover + 1, len(style))

                style_step = style[crossover]

            else:
                if mixing_range[0] <= i <= mixing_range[1]:
                    style_step = style[1]

                else:
                    style_step = style[0]

            if i > 0 and step > 0:
                out_prev = out

            out = conv(out, style_step, nd.array(noise[i], ctx=style[0].context))

            if i == step:

                out = to_rgb(out)

                if i > 0 and 0 <= alpha < 1:
                    skip_rgb = self.to_rgb[i - 1](out_prev)
                    skip_rgb = F.UpSampling(skip_rgb, scale=2, sample_type='nearest')
                    out = (1 - alpha) * skip_rgb + alpha * out

                break

        return out


class StyledGenerator(nn.HybridBlock):
    r"""Style-based GAN
    Reference:
        Tero Karras, Samuli Laine, Timo Aila. "A Style-Based Generator
        Architecture for Generative Adversarial Networks." *CVPR*, 2019
    """
    def __init__(self, code_dim=512, n_mlp=8):
        super().__init__()

        self.generator = Generator()

        self.style = nn.HybridSequential()

        with self.style.name_scope():

            self.style.add(PixelNorm())

            for i in range(n_mlp):
                self.style.add(EqualLinear(code_dim, code_dim))
                self.style.add(nn.LeakyReLU(0.2))


    def hybrid_forward(self, F, x, step=0, alpha=-1, mean_style=None,
                       style_weight=0, noise=None, mixing_range=(-1, -1)):

        styles = []
        if type(x) not in (list, tuple):
            x = [x]

        for i in x:
            styles.append(self.style(i))

        batch = x[0].shape[0]

        if noise is None:
            noise = []

            for i in range(step + 1):
                size = 4 * 2 ** i
                noise.append(nd.random.randn(batch, 1, size, size, ctx=x[0].context))

        if mean_style is not None:
            styles_norm = []

            for style in styles:
                styles_norm.append(mean_style + style_weight * (style - mean_style))

            styles = styles_norm

        nd_styles = nd.empty((len(styles), styles[0].shape[0], styles[0].shape[1]))

        for i, style in enumerate(styles):
            nd_styles[i] = style

        return self.generator(nd_styles, noise, step, alpha, mixing_range)

    def mean_style(self, x):

        style = self.style(x).mean(axis=0, keepdims=True)

        return style
