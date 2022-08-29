""" Copyright 2021-2022 The Jackson Laboratory
"""
import torch.nn as nn


class UpsamplingUnit(nn.Module):
    def __init__(self, channels_in, channels_out, groups=False,
                 batch_norm=False,
                 bias=True):
        super(UpsamplingUnit, self).__init__()

        model = [nn.Conv2d(channels_in, channels_in, 3, 1, 1, 1,
                           channels_in if groups else 1,
                           bias=bias,
                           padding_mode='reflect')]

        if batch_norm:
            model.append(nn.BatchNorm2d(channels_in, affine=True))

        model.append(nn.LeakyReLU(inplace=False))
        model.append(nn.ConvTranspose2d(channels_in, channels_out, 3, 2, 1, 1,
                                        channels_in if groups else 1,
                                        bias=bias))

        if batch_norm:
            model.append(nn.BatchNorm2d(channels_out, affine=True))

        model.append(nn.LeakyReLU(inplace=False))

        self.model = nn.Sequential(*model)

    def forward(self, x):
        fx = self.model(x)
        return fx


class Synthesizer(nn.Module):
    def __init__(self, channels_org=3, channels_net=8, channels_bn=16,
                 compression_level=3,
                 channels_expansion=1,
                 groups=False,
                 batch_norm=False,
                 bias=False,
                 **kwargs):
        super(Synthesizer, self).__init__()

        # Initial convolution in the synthesis track
        up_track = [nn.Conv2d(channels_bn,
                              channels_net
                              * channels_expansion**compression_level,
                              kernel_size=3,
                              stride=1,
                              padding=1,
                              dilation=1,
                              groups=channels_bn if groups else 1,
                              bias=bias,
                              padding_mode='reflect')]
        up_track += [UpsamplingUnit(channels_in=channels_net
                                                * channels_expansion**(i+1),
                                    channels_out=channels_net
                                                 * channels_expansion**i,
                                    groups=groups,
                                    batch_norm=batch_norm,
                                    bias=bias)
                     for i in reversed(range(compression_level))]

        # Final color reconvertion
        up_track.append(nn.Conv2d(channels_net, channels_org, 3, 1, 1, 1, channels_org if groups else 1, bias=bias, padding_mode='reflect'))

        self.synthesis_track = nn.Sequential(*up_track)

    def forward(self, x):
        x_brg = []
        # DataParallel only sends 'x' to the GPU memory when the forward method is used and not for other methods
        fx = x.clone().to(self.synthesis_track[0].weight.device)
        for layer in self.synthesis_track[:-1]:
            fx = layer(fx)
            x_brg.append(fx)

        return x_brg
