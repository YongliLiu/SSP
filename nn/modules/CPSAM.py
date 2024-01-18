import torch
import numpy as np
import torch.nn as nn
from ultralytics.nn.modules import Conv


class PYSAM(nn.Module):
    def __init__(self, c1, c2):

# This section will be made public upon acceptance of the paper.


class CAM(nn.Module):
    def __init__(self, in_planes, reduction=16):
        super(CAM, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // reduction, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // reduction, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        short = x
        x_out1 = self.avg_pool(x) + self.max_pool(x)
        x_out2 = self.fc2(self.relu1(self.fc1(x_out1)))
        # avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        # max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        # out = avg_out + max_out
        return self.sigmoid(x_out2) * short


class MSCAM(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(MSCAM, self).__init__()
        self.c_ = int(in_channel * 0.5)
        self.cv1 = nn.Conv2d(self.c_, self.c_, kernel_size=(1, 3), padding=(0, 1))
        self.cv2 = nn.Conv2d(self.c_, self.c_, kernel_size=(1, 5), padding=(0, 2))
        self.cv3 = nn.Conv2d(self.c_, self.c_, kernel_size=(1, 7), padding=(0, 3))
        self.cv4 = nn.Conv2d(self.c_, self.c_, kernel_size=(3, 1), padding=(1, 0))
        self.cv5 = nn.Conv2d(self.c_, self.c_, kernel_size=(5, 1), padding=(2, 0))
        self.cv6 = nn.Conv2d(self.c_, self.c_, kernel_size=(7, 1), padding=(3, 0))
        self.act = nn.SiLU()
        self.cv = Conv(in_channel, out_channel, 1, 1)

    def forward(self, x):
        short = x
        y = list(torch.split(x, (self.c_, self.c_), dim=1))
        x1 = self.act(self.cv1(y[0]) + self.cv2(y[0]) + self.cv3(y[0]))
        x2 = self.act(self.cv4(y[1]) + self.cv5(y[1]) + self.cv6(y[1]))
        x_out = torch.cat((x1, x2), 1)
        x_out = self.cv(x_out)
        return short * x_out


class CPSAM(nn.Module):
    def __init__(self, c1, c2):
        super(CPSAM, self).__init__()
        self.CAM = CAM(c1, reduction=32)
        self.PYSAM = PYSAM(c1)

    def forward(self, x):
        x_out = self.CAM(x)
        x_out = self.PYSAM(x_out)
        return x_out

