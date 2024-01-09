import torch
import torch.nn as nn

class UNet3DCore(nn.Module):
    """ UNET core block Conv + BNorm + PReLU """
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3, 3)):
        super(UNet3DCore, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, padding=1)
        self.batchnorm1 = nn.BatchNorm3d(out_channels)
        self.prelu1 = nn.PReLU()
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=kernel_size, padding=1)
        self.batchnorm2 = nn.BatchNorm3d(out_channels)
        self.prelu2 = nn.PReLU()

    def forward(self, x):
        x = self.prelu1(self.batchnorm1(self.conv1(x)))
        x = self.prelu2(self.batchnorm2(self.conv2(x)))
        return x

class UNet3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet3D, self).__init__()

        self.encode1 = UNet3DCore(in_channels, 64)
        self.pool1 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.encode2 = UNet3DCore(64, 64*2)
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.encode3 = UNet3DCore(64*2, 64*4)
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.encode4 = UNet3DCore(64*4, 64*8)
        self.pool4 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        # implemented with double convolution block
        self.bridge = UNet3DCore(64*8, 64*16)

        self.upconv1 = nn.ConvTranspose3d(64*16, 64*16, kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=0)
        self.decode1 = UNet3DCore(64*16+64*8, 64*8)
        
        self.upconv2 = nn.ConvTranspose3d(64*8, 64*8, kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=0)
        self.decode2 = UNet3DCore(64*8+64*4, 64*4)
        
        self.upconv3 = nn.ConvTranspose3d(64*4, 64*4, kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=0)
        self.decode3 = UNet3DCore(64*4+64*2, 64*2)
        
        self.upconv4 = nn.ConvTranspose3d(64*2, 64*2, kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=0)
        self.decode4 = UNet3DCore(64*2+64, 64)

        self.final_conv = nn.Conv3d(64, out_channels, kernel_size=1)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        d1 = self.encode1(x)
        l = self.pool1(d1)

        d2 = self.encode2(l)
        l = self.pool2(d2)

        d3 = self.encode3(l)
        l = self.pool3(d3)

        d4 = self.encode4(l)
        l = self.pool4(d4)

        b = self.bridge(l)

        l = self.upconv1(b)
        l = torch.cat([l, d4], dim=1)
        u4 = self.decode1(l)

        l = self.upconv2(u4)
        l = torch.cat([l, d3], dim=1)
        u3 = self.decode2(l)

        l = self.upconv3(u3)
        l = torch.cat([l, d2], dim=1)
        u2 = self.decode3(l)

        l = self.upconv4(u2)
        l = torch.cat([l, d1], dim=1)
        u1 = self.decode4(l)

        output = self.final_conv(u1)
        logit = self.activation(output)
        return logit
