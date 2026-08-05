import torch
from torch import nn


class DoubleConv(nn.Module):
    """(합성곱 -> 배치정규화 -> ReLU)를 2번 반복하는 U-Net의 기본 블록"""

    def __init__(self, in_channels, out_channels, dropout_p: float = 0.0):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if dropout_p > 0:
            layers.append(nn.Dropout(p=dropout_p))

        self.double_conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.double_conv(x)


class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base_channel=32):
        super().__init__()
        c = base_channel

        self.down1 = DoubleConv(in_channels, c)
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = DoubleConv(c, c * 2)
        self.pool2 = nn.MaxPool2d(2)
        self.down3 = DoubleConv(c * 2, c * 4)
        self.pool3 = nn.MaxPool2d(2)
        self.down4 = DoubleConv(c * 4, c * 8, dropout_p=0.2)
        self.pool4 = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(c * 8, c * 16, dropout_p=0.3)

        self.up1 = nn.ConvTranspose2d(c * 16, c * 8, kernel_size=2, stride=2)
        self.conv_up1 = DoubleConv(c * 16, c * 8, dropout_p=0.2)

        self.up2 = nn.ConvTranspose2d(c * 8, c * 4, kernel_size=2, stride=2)
        self.conv_up2 = DoubleConv(c * 8, c * 4)

        self.up3 = nn.ConvTranspose2d(c * 4, c * 2, kernel_size=2, stride=2)
        self.conv_up3 = DoubleConv(c * 4, c * 2)

        self.up4 = nn.ConvTranspose2d(c * 2, c, kernel_size=2, stride=2)
        self.conv_up4 = DoubleConv(c * 2, c)

        self.out = nn.Conv2d(c, out_channels, kernel_size=1)

    def forward(self, x):
        # Downsampling path
        d1 = self.down1(x)
        d2 = self.down2(self.pool1(d1))
        d3 = self.down3(self.pool2(d2))
        d4 = self.down4(self.pool3(d3))

        b = self.bottleneck(self.pool4(d4))

        u1 = self.up1(b)
        u1 = torch.cat((u1, d4), dim=1)
        c1 = self.conv_up1(u1)

        u2 = self.up2(c1)
        u2 = torch.cat((u2, d3), dim=1)
        c2 = self.conv_up2(u2)

        u3 = self.up3(c2)
        u3 = torch.cat((u3, d2), dim=1)
        c3 = self.conv_up3(u3)

        u4 = self.up4(c3)
        u4 = torch.cat((u4, d1), dim=1)
        c4 = self.conv_up4(u4)

        return self.out(c4)
