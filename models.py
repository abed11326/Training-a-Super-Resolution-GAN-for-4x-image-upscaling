import torch
from torch.nn import Module, Conv2d, LeakyReLU, PReLU, BatchNorm2d, Sequential, PixelShuffle, AdaptiveAvgPool2d, Flatten, Linear, Dropout2d, Dropout

class ResidualUnit(Module):
    def __init__(self):
        super(ResidualUnit, self).__init__()
        self.conv1 = Sequential(Conv2d(64, 64, 3, 1, "same"), BatchNorm2d(64), PReLU(64))
        self.conv2 = Sequential(Conv2d(64, 64, 3, 1, "same"), BatchNorm2d(64))

    def forward(self, input):
        out = self.conv1(input)
        out = self.conv2(out)
        return input + out

class UpsampleUnit(Module):
    def __init__(self):
        super(UpsampleUnit, self).__init__()
        self.conv = Conv2d(64, 256, 3, 1, "same")
        self.shuffle = PixelShuffle(2)
        self.activation = PReLU(64)

    def forward(self, X):
        return self.activation(self.shuffle(self.conv(X)))
    

class Generator(Module):
    def __init__(self, no_resBlocks):
        super(Generator, self).__init__()
        self.residuals = Sequential(*[ResidualUnit()] * no_resBlocks)
        self.upsample = Sequential(UpsampleUnit(), UpsampleUnit())
        self.initialConv = Sequential(Conv2d(3, 64, 9, 1, "same"), PReLU(64))
        self.midConv = Sequential(Conv2d(64, 64, 3, 1, "same"), BatchNorm2d(64))
        self.finalConv = Conv2d(64, 3, 9, 1, "same")

    def forward(self, input):
        input = self.initialConv(input)
        out = self.residuals(input)
        out = self.midConv(out)
        out = out + input
        out = self.upsample(out)
        out = self.finalConv(out)
        return torch.tanh(out)

class DiscConvBlock(Module):
    def __init__(self, in_channels, out_channels, stride):
        super(DiscConvBlock, self).__init__()
        self.conv = Conv2d(in_channels, out_channels, 3, stride, 1)
        self.bn = BatchNorm2d(out_channels)
        self.activation = LeakyReLU(0.2)
        self.dropout = Dropout2d(p=0.50)

    def forward(self, X):
        return self.dropout(self.activation(self.bn(self.conv(X))))

    
class Discriminator(Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.initial_conv = Sequential(
            Conv2d(3, 64, 3, 1, "same"),
            LeakyReLU(0.2),
            Dropout2d(p=0.5)
        )
        self.conv_seq = Sequential(
            DiscConvBlock(64, 64, 2),
            DiscConvBlock(64, 128, 1),
            DiscConvBlock(128, 128, 2),
            DiscConvBlock(128, 256, 1),
            DiscConvBlock(256, 256, 2),
            DiscConvBlock(256, 512, 1),
            DiscConvBlock(512, 512, 2),
            AdaptiveAvgPool2d(1),
            Flatten()
        )
        self.fc = Sequential(
            Linear(512, 1024),
            LeakyReLU(0.2),
            Dropout(0.50),
            Linear(1024, 1)
        )

    def forward(self, X):
        return self.fc(self.conv_seq(self.initial_conv(X)))
