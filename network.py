
from networks import *
import torch.nn as nn

class HeatMapRegressionNetwork(nn.Module):
    def __init__(self, n_locations, n_limbs ,  backbone):
        super(HeatMapRegressionNetwork, self).__init__()

        if backbone == "shufflenetv2":
            self.resnet = ShuffleNetV2.shufflenetv2_ed(width_mult=1.0)
            self.outsize = 64
        elif backbone == "mobilenetv2":
            self.resnet = MobileNetV2.mobilenetv2_ed(width_mult=1.0)
            self.outsize = 32

        self.hm_conv = nn.Sequential(nn.Conv2d(self.outsize,128 , kernel_size=1, stride=1, padding=0),
                                 nn.BatchNorm2d(128),
                                 nn.ReLU(inplace=True),
                                 nn.Conv2d(128, n_locations+n_limbs, kernel_size=1, stride=1, padding=0))

    def forward(self, images):
        resnet_out = self.resnet(images)
        heatmaps = self.hm_conv(resnet_out)



        return heatmaps