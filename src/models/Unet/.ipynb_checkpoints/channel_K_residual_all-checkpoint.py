import torchvision.transforms.functional as tf
from torchsummary import summary
import torch
import torch.nn as nn

# Code reference from Online Github repo
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
        residual_connections = []

        for down in self.down:
            x = down(x)
            residual_connections.append(x)
            x = self.pool(x)

        return x, residual_connections


class Decoder(nn.Module):

    def __init__(self, out_channel=1, channels=(512, 256, 128, 64)):
        super(Decoder, self).__init__()

        self.up_tp = nn.ModuleList()
        self.dc = DoubleConv(512, 1024)

        for channel in channels:
            self.up_tp.append(
                self.up_transpose(channel * 2, channel),
            )
            self.up_tp.append(DoubleConv(channel * 2, channel))

        self.final_conv = nn.Conv2d(channels[-1], out_channel, kernel_size=1)

    @staticmethod
    def up_transpose(in_channel, out_channel):
        return nn.ConvTranspose2d(in_channels=in_channel, out_channels=out_channel, kernel_size=2, stride=2)

    def forward(self, x):
        out, residual_connections = x

        residual_connections = residual_connections[::-1]

        out = self.dc(out)

        for i in range(0, len(self.up_tp), 2):
            out = self.up_tp[i](out)
            residual_connection = residual_connections[i // 2]

            if out.shape != residual_connection:
                out = tf.resize(out, size=residual_connection.shape[2:])

            concat_residue = torch.cat((residual_connection, out), dim=1)

            out = self.up_tp[i + 1](concat_residue)

        return self.final_conv(out)


class channel_UNET(nn.Module):

    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()
        self.encoder = Encoder(in_channels=in_channels, channels=(64, 128, 256, 512))
        self.decoder = Decoder(out_channel=out_channels, channels=(512, 256, 128, 64))

        self.linear = nn.Linear(512, 512)

    def forward(self, x):
        encoder_output, residual_connections = self.encoder(x)

        encoder_output = encoder_output.permute(0, 2, 3, 1)

        encoder_output = self.linear(encoder_output)

        encoder_output = encoder_output.permute(0, 3, 1, 2)

        decoder_input = (encoder_output, residual_connections)
        x = self.decoder(decoder_input)
        return x


if __name__ == "__main__":
    model = channel_UNET(in_channels=2, out_channels=2)
    result = model(torch.randn(1, 2, 64, 64))
    print(result.shape)