import torch as t
import torch.nn as nn
import torch.nn.functional as F
from Am_Res import ResidualCBAMBlock, CBAMBlock

class DoubleConv(nn.Module):
    def __init__(self, in_channel, out_channels):
        super().__init__()
        self.conv_op = nn.Sequential(
            nn.Conv2d(in_channel, out_channels, kernel_size=(3,3), padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=(3,3), padding=1),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.conv_op(x)

class Downsample(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.conv = DoubleConv(in_channel, out_channel)
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

    def forward(self, x):
        conv = self.conv(x)
        pool = self.pool(conv)
        return conv, pool

class UpSample(nn.Module):
    def __init__(self, in_channel, out_channel,attention_residual=True):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channel, in_channel//2, kernel_size=(2,2), stride=2)
        self.conv = DoubleConv(in_channel, out_channel)
        self.attention_residual = attention_residual

        if attention_residual:
            self.res = ResidualCBAMBlock(out_channel,out_channel)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = t.cat([x1, x2], 1)
        x = self.conv(x)
        if self.attention_residual:
            x = self.res(x)
        return x
