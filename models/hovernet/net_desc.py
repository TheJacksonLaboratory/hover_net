import math
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .net_utils import (DenseBlock, Net, ResidualBlock, TFSamepaddingLayer,
                        UpSample2x)
from .utils import crop_op, crop_to_shape


####
class HoVerNet(Net):
    """Initialise HoVer-Net."""

    def __init__(self, input_ch=3, nr_types=None, freeze=False, mode='original', **kwargs):
        super().__init__()
        self.mode = mode
        self.freeze = freeze
        self.nr_types = nr_types
        self.output_ch = 3 if nr_types is None else 4

        assert mode == 'original' or mode == 'fast', \
                'Unknown mode `%s` for HoVerNet %s. Only support `original` or `fast`.' % mode

        module_list = [
            ("/", nn.Conv2d(input_ch, 64, 7, stride=1, padding=0, bias=False)),
            ("bn", nn.BatchNorm2d(64, eps=1e-5)),
            ("relu", nn.ReLU(inplace=True)),
        ]
        if mode == 'fast': # prepend the padding for `fast` mode
            module_list = [("pad", TFSamepaddingLayer(ksize=7, stride=1))] + module_list

        self.conv0 = nn.Sequential(OrderedDict(module_list))
        self.d0 = ResidualBlock(64, [1, 3, 1], [64, 64, 256], 3, stride=1)
        self.d1 = ResidualBlock(256, [1, 3, 1], [128, 128, 512], 4, stride=2)
        self.d2 = ResidualBlock(512, [1, 3, 1], [256, 256, 1024], 6, stride=2)
        self.d3 = ResidualBlock(1024, [1, 3, 1], [512, 512, 2048], 3, stride=2)

        self.conv_bot = nn.Conv2d(2048, 1024, 1, stride=1, padding=0, bias=False)

        def create_decoder_branch(out_ch=2, ksize=5):
            module_list = [ 
                ("conva", nn.Conv2d(1024, 256, ksize, stride=1, padding=0, bias=False)),
                ("dense", DenseBlock(256, [1, ksize], [128, 32], 8, split=4)),
                ("convf", nn.Conv2d(512, 512, 1, stride=1, padding=0, bias=False),),
            ]
            u3 = nn.Sequential(OrderedDict(module_list))

            module_list = [ 
                ("conva", nn.Conv2d(512, 128, ksize, stride=1, padding=0, bias=False)),
                ("dense", DenseBlock(128, [1, ksize], [128, 32], 4, split=4)),
                ("convf", nn.Conv2d(256, 256, 1, stride=1, padding=0, bias=False),),
            ]
            u2 = nn.Sequential(OrderedDict(module_list))

            module_list = [ 
                ("conva/pad", TFSamepaddingLayer(ksize=ksize, stride=1)),
                ("conva", nn.Conv2d(256, 64, ksize, stride=1, padding=0, bias=False),),
            ]
            u1 = nn.Sequential(OrderedDict(module_list))

            module_list = [ 
                ("bn", nn.BatchNorm2d(64, eps=1e-5)),
                ("relu", nn.ReLU(inplace=True)),
                ("conv", nn.Conv2d(64, out_ch, 1, stride=1, padding=0, bias=True),),
            ]
            u0 = nn.Sequential(OrderedDict(module_list))

            decoder = nn.Sequential(
                OrderedDict([("u3", u3), ("u2", u2), ("u1", u1), ("u0", u0),])
            )
            return decoder

        ksize = 5 if mode == 'original' else 3
        if nr_types is None:
            self.decoder = nn.ModuleDict(
                OrderedDict(
                    [
                        ("np", create_decoder_branch(ksize=ksize,out_ch=2)),
                        ("hv", create_decoder_branch(ksize=ksize,out_ch=2)),
                    ]
                )
            )
        else:
            self.decoder = nn.ModuleDict(
                OrderedDict(
                    [
                        ("tp", create_decoder_branch(ksize=ksize, out_ch=nr_types)),
                        ("np", create_decoder_branch(ksize=ksize, out_ch=2)),
                        ("hv", create_decoder_branch(ksize=ksize, out_ch=2)),
                    ]
                )
            )

        self.upsample2x = UpSample2x()
        # TODO: pytorch still require the channel eventhough its ignored
        self.weights_init()

    def forward(self, imgs):

        imgs = imgs / 255.0  # to 0-1 range to match XY

        if self.training:
            d0 = self.conv0(imgs)
            d0 = self.d0(d0, self.freeze)
            with torch.set_grad_enabled(not self.freeze):
                d1 = self.d1(d0)
                d2 = self.d2(d1)
                d3 = self.d3(d2)
            d3 = self.conv_bot(d3)
            d = [d0, d1, d2, d3]
        else:
            d0 = self.conv0(imgs)
            d0 = self.d0(d0)
            d1 = self.d1(d0)
            d2 = self.d2(d1)
            d3 = self.d3(d2)
            d3 = self.conv_bot(d3)
            d = [d0, d1, d2, d3]

        # TODO: switch to `crop_to_shape` ?
        if self.mode == 'original':
            d[0] = crop_op(d[0], [184, 184])
            d[1] = crop_op(d[1], [72, 72])
        else:
            d[0] = crop_op(d[0], [92, 92])
            d[1] = crop_op(d[1], [36, 36])

        out_dict = OrderedDict()
        for branch_name, branch_desc in self.decoder.items():
            u3 = self.upsample2x(d[-1]) + d[-2]
            u3 = branch_desc[0](u3)

            u2 = self.upsample2x(u3) + d[-3]
            u2 = branch_desc[1](u2)

            u1 = self.upsample2x(u2) + d[-4]
            u1 = branch_desc[2](u1)

            u0 = branch_desc[3](u1)
            out_dict[branch_name] = u0

        return out_dict


####
class HoVerNetComp(Net):
    """Initialise HoVer-Net for compressed images."""

    def __init__(self, input_ch=48, nr_types=None, freeze=False, mode='compressed_rec', rec_model=None):
        super().__init__()
        self.mode = mode
        self.freeze = freeze
        self.nr_types = nr_types
        self.output_ch = 3 if nr_types is None else 4

        assert mode == 'compressed_rec', \
                'Unknown mode `%s` for HoVerNetComp %s. Only support `compressed_rec`.' % mode

        self.d0 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=1, stride=1, padding=0, bias=False)
        self.d1 = nn.Conv2d(in_channels=128, out_channels=512, kernel_size=1, stride=1, padding=0, bias=False)
        self.d2 = nn.Conv2d(in_channels=128, out_channels=1024, kernel_size=1, stride=1, padding=0, bias=False)
        self.d3 = nn.Conv2d(in_channels=128, out_channels=1024, kernel_size=1, stride=1, padding=0, bias=False)

        def create_decoder_branch(out_ch=2, ksize=5):
            module_list = [ 
                ("conva", nn.Conv2d(1024, 256, ksize, stride=1, padding=0, bias=False)),
                ("dense", DenseBlock(256, [1, ksize], [128, 32], 8, split=4)),
                ("convf", nn.Conv2d(512, 512, 1, stride=1, padding=0, bias=False),),
            ]
            u3 = nn.Sequential(OrderedDict(module_list))

            module_list = [ 
                ("conva", nn.Conv2d(512, 128, ksize, stride=1, padding=0, bias=False)),
                ("dense", DenseBlock(128, [1, ksize], [128, 32], 4, split=4)),
                ("convf", nn.Conv2d(256, 256, 1, stride=1, padding=0, bias=False),),
            ]
            u2 = nn.Sequential(OrderedDict(module_list))

            module_list = [ 
                ("conva/pad", TFSamepaddingLayer(ksize=ksize, stride=1)),
                ("conva", nn.Conv2d(256, 64, ksize, stride=1, padding=0, bias=False),),
            ]
            u1 = nn.Sequential(OrderedDict(module_list))

            module_list = [ 
                ("bn", nn.BatchNorm2d(64, eps=1e-5)),
                ("relu", nn.ReLU(inplace=True)),
                ("conv", nn.Conv2d(64, out_ch, 1, stride=1, padding=0, bias=True),),
            ]
            u0 = nn.Sequential(OrderedDict(module_list))

            decoder = nn.Sequential(
                OrderedDict([("u3", u3), ("u2", u2), ("u1", u1), ("u0", u0),])
            )
            return decoder

        ksize = 5 if mode == 'original' else 3
        if nr_types is None:
            self.decoder = nn.ModuleDict(
                OrderedDict(
                    [
                        ("np", create_decoder_branch(ksize=ksize,out_ch=2)),
                        ("hv", create_decoder_branch(ksize=ksize,out_ch=2)),
                    ]
                )
            )
        else:
            self.decoder = nn.ModuleDict(
                OrderedDict(
                    [
                        ("tp", create_decoder_branch(ksize=ksize, out_ch=nr_types)),
                        ("np", create_decoder_branch(ksize=ksize, out_ch=2)),
                        ("hv", create_decoder_branch(ksize=ksize, out_ch=2)),
                    ]
                )
            )

        self.upsample2x = UpSample2x()
        # TODO: pytorch still require the channel eventhough its ignored
        self.weights_init()

        # The reconstruction model should be initialized already with the
        # training checkpoint.
        self._rec_model = rec_model

    def forward(self, imgs):
        # Center the compressed tensor in 0.
        imgs = imgs - 127.5

        with torch.no_grad():
            r = self._rec_model(imgs)

        if self.training:
            d0 = self.d0(r[3], self.freeze)
            with torch.set_grad_enabled(not self.freeze):
                d1 = self.d1(r[2])
                d2 = self.d2(r[1])
                d3 = self.d3(r[0])
        else:
            d0 = self.d0(r[3])
            d1 = self.d1(r[2])
            d2 = self.d2(r[1])
            d3 = self.d3(r[0])
        d = [d0, d1, d2, d3]

        # TODO: switch to `crop_to_shape` ?
        if self.mode == 'original':
            d[0] = crop_op(d[0], [184, 184])
            d[1] = crop_op(d[1], [72, 72])
        else:
            d[0] = crop_op(d[0], [92, 92])
            d[1] = crop_op(d[1], [36, 36])

        out_dict = OrderedDict()
        for branch_name, branch_desc in self.decoder.items():
            u3 = self.upsample2x(d[-1]) + d[-2]
            u3 = branch_desc[0](u3)

            u2 = self.upsample2x(u3) + d[-3]
            u2 = branch_desc[1](u2)

            u1 = self.upsample2x(u2) + d[-4]
            u1 = branch_desc[2](u1)

            u0 = branch_desc[3](u1)
            out_dict[branch_name] = u0

        return out_dict


####
def create_model(mode=None, **kwargs):
    if mode not in ['original', 'fast', 'compressed_rec']:
        assert "Unknown Model Mode %s" % mode
    if mode == 'compressed_rec':
        return HoVerNetComp(mode=mode, **kwargs)
    return HoVerNet(mode=mode, **kwargs)
