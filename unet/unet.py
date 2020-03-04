""" Full assembly of the parts to form the complete network """

import torch.nn.functional as F

from .unet_blocks import *
from torchsummary import summary
# from .Unet_ResBlock import *


class UNet(nn.Module):
    def __init__(self, in_channels, n_classes, num_filters=[32,64,128,192], if_last_layer=True, bilinear=True, initializers = None):
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.n_filters = num_filters
        self.bilinear = bilinear
        self.if_last_layer = if_last_layer
        self.initializers = initializers

        self.inc = SequentialConv(self.in_channels, self.n_filters[0])
        layers_down = []
        layers_up = []
        for i in range(len(self.n_filters) - 1):
            layers_down.append(DownConvBlock(num_filters[i], num_filters[i+1]))
        layers_down.append(DownConvBlock(num_filters[-1], num_filters[-1]))
        for j in range(1, len(self.n_filters)):
            layers_up.append(UpConvBlock(num_filters[-j]*2, num_filters[-(j+1)]))
        layers_up.append(UpConvBlock(num_filters[0]*2, num_filters[0]))
        self.down = nn.ModuleList(layers_down)
        self.up = nn.ModuleList(layers_up)

        self.outc = OutConv(self.n_filters[0], n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        feature_map = [x1]
        for c, layer in enumerate(self.down):
            feature_map.append(layer(feature_map[c]))

        for c, up_layer in enumerate(self.up):
            if c == 0:
                x = up_layer(feature_map[-1], feature_map[-2])
            else:
                x = up_layer(x, feature_map[-c-2])
        if self.if_last_layer:
            logits = self.outc(x)
        else:
            logits = x

        return logits


if __name__ == '__main__':
    net = UNet(in_channels=1, n_classes=1, bilinear=True).cuda()
    summary(net, (1, 128, 128))