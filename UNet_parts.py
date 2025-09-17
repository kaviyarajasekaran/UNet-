import torch as t
import torch.nn as nn
import torch.nn.functional as F

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16, use_linear=False, bias=True):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.use_linear = use_linear

        if use_linear:
            self.fc = nn.Sequential(
                nn.Linear(in_channels, in_channels // reduction, bias=bias),
                nn.ReLU(),
                nn.Linear(in_channels // reduction, in_channels, bias=bias)
            )
        else:
            self.fc = nn.Sequential(
                nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1, bias=bias),
                nn.ReLU(),
                nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1, bias=bias)
            )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        if self.use_linear:
            avg_out = self.fc(self.avg_pool(x).view(x.size(0), -1))
            max_out = self.fc(self.max_pool(x).view(x.size(0), -1))
            out = (avg_out + max_out).unsqueeze(2).unsqueeze(3)
        else:
            avg_out = self.fc(self.avg_pool(x))
            max_out = self.fc(self.max_pool(x))
            out = avg_out + max_out
        return self.sigmoid(out) * x

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=(kernel_size-1)//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        return self.sigmoid(out) * x

class CBAMBlock(nn.Module):
    def __init__(self, in_channels, reduction=16, kernel_size=7):
        super().__init__()
        self.channel_att = ChannelAttention(in_channels, reduction)
        self.spatial_att = SpatialAttention(kernel_size)

    def forward(self, x):
        out = self.channel_att(x)
        out = self.spatial_att(out)
        return out

class ResidualCBAMBlock(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=16, kernel_size=7):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.cbam = CBAMBlock(out_channels, reduction, kernel_size)
    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.cbam(out)  # Apply CBAM after conv layers
        out += residual       # Residual connection
        out = F.relu(out)
        return out


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
