# src/unet.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """
    (Conv -> BN -> ReLU) * 2
    """
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Down(nn.Module):
    """
    Downscaling with maxpool then double conv
    """
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            DoubleConv(in_ch, out_ch)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Up(nn.Module):
    """
    Upscaling then double conv.
    Uses bilinear upsampling by default (more stable on CPU).
    """
    def __init__(self, in_ch: int, out_ch: int, bilinear: bool = True):
        super().__init__()
        self.bilinear = bilinear

        if bilinear:
            # in_ch is channels of concatenated feature map (skip + upsampled)
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
            self.conv = DoubleConv(in_ch, out_ch)
        else:
            # When using ConvTranspose2d, we upsample and reduce channels before concat.
            # Here, we expect in_ch to be channels after concat, but transposed conv needs half.
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)

        # Handle odd sizes safely (shouldn't happen with 256, but good practice)
        diff_y = skip.size(2) - x.size(2)
        diff_x = skip.size(3) - x.size(3)
        if diff_y != 0 or diff_x != 0:
            x = F.pad(x, [diff_x // 2, diff_x - diff_x // 2,
                          diff_y // 2, diff_y - diff_y // 2])

        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """
    Final 1x1 conv to produce logits for each class
    """
    def __init__(self, in_ch: int, num_classes: int):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UNet(nn.Module):
    """
    U-Net for semantic segmentation.

    Input : (B, in_channels, H, W)
    Output: (B, num_classes, H, W)  (logits, no softmax)
    """
    def __init__(self, in_channels: int = 4, num_classes: int = 10, base_channels: int = 64, bilinear: bool = True):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.base_channels = base_channels
        self.bilinear = bilinear

        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4
        c4 = base_channels * 8
        c5 = base_channels * 16

        # Encoder
        self.inc = DoubleConv(in_channels, c1)
        self.down1 = Down(c1, c2)
        self.down2 = Down(c2, c3)
        self.down3 = Down(c3, c4)

        # If bilinear, we usually reduce the deepest channels (standard U-Net trick)
        factor = 2 if bilinear else 1
        self.down4 = Down(c4, c5 // factor)

        # Decoder (note in_ch = channels after concat)
        self.up1 = Up(c5, c4 // factor, bilinear=bilinear)
        self.up2 = Up(c4, c3 // factor, bilinear=bilinear)
        self.up3 = Up(c3, c2 // factor, bilinear=bilinear)
        self.up4 = Up(c2, c1, bilinear=bilinear)

        self.outc = OutConv(c1, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        x1 = self.inc(x)      # (B, c1, H, W)
        x2 = self.down1(x1)   # (B, c2, H/2, W/2)
        x3 = self.down2(x2)   # (B, c3, H/4, W/4)
        x4 = self.down3(x3)   # (B, c4, H/8, W/8)
        x5 = self.down4(x4)   # (B, c5/factor, H/16, W/16)

        # Decoder
        x = self.up1(x5, x4)  # (B, c4/factor, H/8, W/8)
        x = self.up2(x, x3)   # (B, c3/factor, H/4, W/4)
        x = self.up3(x, x2)   # (B, c2/factor, H/2, W/2)
        x = self.up4(x, x1)   # (B, c1, H, W)

        logits = self.outc(x)  # (B, num_classes, H, W)
        return logits
