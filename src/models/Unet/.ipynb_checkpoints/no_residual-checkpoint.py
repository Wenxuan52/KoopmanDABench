import torchvision.transforms.functional as tf
from torchsummary import summary
import torch
import torch.nn as nn

# 版本2: 完全没有残差连接
class DoubleConv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class Encoder(nn.Module):

    def __init__(self, in_channels=1, channels=(64, 128, 256, 512)):
        super(Encoder, self).__init__()
        self.down = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        for channel in channels:
            self.down.append(DoubleConv(in_channels, channel))
            in_channels = channel

    def forward(self, x):
        for down in self.down:
            x = down(x)
            x = self.pool(x)

        return x


class Decoder(nn.Module):

    def __init__(self, out_channel=1, channels=(512, 256, 128, 64)):
        super(Decoder, self).__init__()

        self.up_tp = nn.ModuleList()
        self.dc = DoubleConv(512, 1024)

        # 第一层: 1024 -> 512
        self.up_tp.append(self.up_transpose(1024, channels[0]))
        self.up_tp.append(DoubleConv(channels[0], channels[0]))

        # 后续层: channel -> next_channel
        for i in range(1, len(channels)):
            self.up_tp.append(self.up_transpose(channels[i-1], channels[i]))
            self.up_tp.append(DoubleConv(channels[i], channels[i]))

        self.final_conv = nn.Conv2d(channels[-1], out_channel, kernel_size=1)

    @staticmethod
    def up_transpose(in_channel, out_channel):
        return nn.ConvTranspose2d(in_channels=in_channel, out_channels=out_channel, kernel_size=2, stride=2)

    def forward(self, x):
        out = self.dc(x)

        for i in range(0, len(self.up_tp), 2):
            out = self.up_tp[i](out)
            out = self.up_tp[i + 1](out)

        return self.final_conv(out)


class UNET_NoResidual(nn.Module):

    def __init__(self):
        super().__init__()
        self.encoder = Encoder(in_channels=1, channels=(64, 128, 256, 512))
        self.decoder = Decoder(out_channel=1, channels=(512, 256, 128, 64))

        self.spatial_size = 4
        self.feature_dim = 512 * self.spatial_size * self.spatial_size
        
        self.rank = 256
        self.U = nn.Linear(self.feature_dim, self.rank, bias=False)
        self.V = nn.Linear(self.rank, self.feature_dim, bias=True)

    def forward(self, x):
        encoder_output = self.encoder(x)

        B, C, H, W = encoder_output.shape
        flattened = encoder_output.view(B, -1)  # (B, C*H*W)
        
        intermediate = self.U(flattened)
        transformed = self.V(intermediate)
        
        K_output = transformed.view(B, C, H, W)

        output = self.decoder(K_output)
        
        return output


if __name__ == "__main__":
    model = UNET_NoResidual()
    result = model(torch.randn(1, 1, 64, 64))
    print(result.shape)