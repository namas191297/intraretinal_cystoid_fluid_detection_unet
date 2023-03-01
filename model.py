import torch
import torch.nn as nn
import torchvision
from config import hyperparams

class UNet(torch.nn.Module):
    def __init__(self, img_channels=1, out_channels=1):
        super().__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.contraction_dc1 = self.DoubleConv(img_channels, 64)
        self.contraction_dc2 = self.DoubleConv(64, 128)
        self.contraction_dc3 = self.DoubleConv(128, 256)
        self.contraction_dc4 = self.DoubleConv(256, 512)
        self.bottleneck = self.DoubleConv(512, 1024)
        self.upsample4 = self.UpsampleConv(1024, 512)
        self.expansion_dc4 = self.DoubleConv(1024, 512)
        self.upsample3 = self.UpsampleConv(512, 256)
        self.expansion_dc3 = self.DoubleConv(512, 256)
        self.upsample2 = self.UpsampleConv(256, 128)
        self.expansion_dc2 = self.DoubleConv(256, 128)
        self.upsample1 = self.UpsampleConv(128, 64)
        self.expansion_dc1 = self.DoubleConv(128, 64)
        self.output = nn.Conv2d(64, out_channels, kernel_size=1)

    def DoubleConv(self, in_channels, features):
        return nn.Sequential(
            nn.Conv2d(in_channels, features, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(features),
            nn.ReLU(),
            nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(features),
            nn.ReLU()
        )

    def UpsampleConv(self, in_channels, features):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, features, kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(features),
            nn.ReLU()
        )

    def forward(self, x):

        # Contraction Path
        dc1 = self.contraction_dc1(x)
        out = self.maxpool(dc1)
        dc2 = self.contraction_dc2(out)
        out = self.maxpool(dc2)
        dc3 = self.contraction_dc3(out)
        out = self.maxpool(dc3)
        dc4 = self.contraction_dc4(out)
        out = self.maxpool(dc4)

        # Bottleneck
        bottleneck = self.bottleneck(out)

        # Expansion Path
        ups4 = self.upsample4(bottleneck)
        if ups4.shape != dc4.shape:
            ups4 = torchvision.transforms.functional.resize(ups4, size=dc4.shape[2:])
        out = torch.cat((dc4, ups4), dim=1)
        out = self.expansion_dc4(out)

        ups3 = self.upsample3(out)
        if ups3.shape != dc3.shape:
            ups3 = torchvision.transforms.functional.resize(ups3, size=dc3.shape[2:])
        out = torch.cat((dc3, ups3), dim=1)
        out = self.expansion_dc3(out)

        ups2 = self.upsample2(out)
        if ups2.shape != dc2.shape:
            ups2 = torchvision.transforms.functional.resize(ups2, size=dc2.shape[2:])
        out = torch.cat((dc2, ups2), dim=1)
        out = self.expansion_dc2(out)

        ups1 = self.upsample1(out)
        if ups1.shape != dc1.shape:
            ups1 = torchvision.transforms.functional.resize(ups1, size=dc1.shape[2:])
        out = torch.cat((dc1, ups1), dim=1)
        out = self.expansion_dc1(out)

        # Final Output
        out = self.output(out)
        return out

def test_unet_architecture(image_height=300, image_width=300):
    model = UNet()
    imgs = torch.randn(hyperparams['img_channels'], hyperparams['out_channels'], 300, 300)  # N x C x H x W
    output = model(imgs)
    assert imgs.shape == output.shape
    print(f'UNet accepts the {image_height}x{image_width} image dimensions.')

if __name__ == '__main__':
    test_unet_architecture()