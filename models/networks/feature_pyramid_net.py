import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1),
            
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1)
        )

    def forward(self, x):
        return self.double_conv(x)
    
class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1)
        )
        
    def forward(self, x, y):
        """
        Upsample x and add with y
        
        Args:
            x: top feature map to be upsampled
            y: skip-connected feature map
            
        Returns:
            added feature map
        """
        return self.upsample(x) + y
    
class FeaturePyramidNet(nn.Module):
    """
    Feature Pyramid Network with bottom up and top down pathway
    Bottom up pathway: CNN reduce resolution and increase feature
    Top down pathway: transposed CNN increase resolution and reduce feature
    In between: skip-connection
    """
    def __init__(self):
        super().__init__()
        
        self.layer1 = DoubleConv(in_channels=3, out_channels=16)
        self.layer2 = DoubleConv(in_channels=16, out_channels=32)
        self.layer3 = DoubleConv(in_channels=32, out_channels=64)
        self.layer4 = DoubleConv(in_channels=64, out_channels=96)
        self.layer5 = DoubleConv(in_channels=96, out_channels=128)
        self.layer6 = DoubleConv(in_channels=128, out_channels=196)

        self.pyr_top = nn.Sequential(
            nn.Conv2d(196, 196, kernel_size=1, bias=False),
            nn.BatchNorm2d(196),
            nn.LeakyReLU(0.1)
        )

        self.upsample5 = Upsample(196, 128)
        self.upsample4 = Upsample(128, 96)
        self.upsample3 = Upsample(96, 64)
        self.upsample2 = Upsample(64, 32)
        
        
    def forward(self, x):
        # bottom-up
        c1 = self.layer1(x)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)
        c5 = self.layer5(c4)
        c6 = self.layer6(c5)
        
        # top-down
        p6 = self.pyr_top(c6)
        p5 = self.upsample5(p6, c5)
        p4 = self.upsample4(p5, c4)
        p3 = self.upsample3(p4, c3)
        p2 = self.upsample2(p3, c2)
        
        return p2, p3, p4, p5, p6
      
    
    