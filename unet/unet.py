""" Full assembly of the parts to form the complete network """

import torch.nn.functional as F

from .unet_blocks import *


class UNet(nn.Module):
    # TODO: check structure
    def __init__(self, in_channels, n_classes, num_filters=[32,64,128,192], bilinear=True):
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.n_filters = num_filters
        self.bilinear = bilinear

        self.inc = SequentialConv(self.in_channels, 32)
        self.down1 = DownConvBlock(32, 64)
        self.down2 = DownConvBlock(64, 128)
        self.down3 = DownConvBlock(128, 192)
        self.down4 = DownConvBlock(192, 192)
        self.up1 = UpConvBlock(384, 128, bilinear)
        self.up2 = UpConvBlock(256, 64, bilinear)
        self.up3 = UpConvBlock(128, 32, bilinear)
        self.up4 = UpConvBlock(64, 32, bilinear)
        self.outc = OutConv(32, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits
