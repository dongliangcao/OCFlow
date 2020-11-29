import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import init  
class Downsample(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, proj_ratio=4):
        super().__init__()
        inter_channels = in_channels // proj_ratio
        self.conv1 = nn.Conv2d(in_channels, inter_channels, 2, stride=2, bias=False)
        self.bn1 = nn.BatchNorm2d(inter_channels)
        self.lrelu1 = nn.LeakyReLU(0.1)
        
        self.conv2 = nn.Conv2d(inter_channels, inter_channels, kernel_size, padding=kernel_size//2, bias=False)
        self.bn2 = nn.BatchNorm2d(inter_channels)
        self.lrelu2 = nn.LeakyReLU(0.1)
        
        self.conv3 = nn.Conv2d(inter_channels, out_channels, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.lrelu3 = nn.LeakyReLU(0.1)
        
    def forward(self, x):
        x = self.lrelu1(self.bn1(self.conv1(x)))
        x = self.lrelu2(self.bn2(self.conv2(x)))
        x = self.lrelu3(self.bn3(self.conv3(x)))
        return x

class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels, proj_ratio=4, activation=True):
        super().__init__()
        inter_channels = in_channels // proj_ratio
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv1 = nn.Conv2d(in_channels, inter_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(inter_channels)
        self.lrelu1 = nn.LeakyReLU(0.1)
        
        self.conv2 = nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(inter_channels)
        self.lrelu2 = nn.LeakyReLU(0.1)
        
        self.conv3 = nn.Conv2d(inter_channels, out_channels, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.lrelu3 = nn.LeakyReLU(0.1) if activation else nn.Identity()
        
    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        
        x1 = F.pad(x1, [diffX//2, diffX - diffX//2, diffY//2, diffY - diffY//2])
        x = torch.cat([x2, x1], dim=1)
        
        x = self.lrelu1(self.bn1(self.conv1(x)))
        x = self.lrelu2(self.bn2(self.conv2(x)))
        x = self.lrelu3(self.bn3(self.conv3(x)))
        return x
        
class InpaintingNet(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        ## Encoder part
        self.down1 = Downsample(in_channels, 32, kernel_size=7, proj_ratio=1)
        self.down2 = Downsample(32, 64, kernel_size=5)
        self.down3 = Downsample(64, 128, kernel_size=5)
        self.down4 = Downsample(128, 128)
        self.down5 = Downsample(128, 128)
        self.down6 = Downsample(128, 128)
        
        ## Decoder part
        self.up1 = Upsample(128+128, 128, proj_ratio=8) # skip-connected with output from down5
        self.up2 = Upsample(128+128, 128, proj_ratio=8) # skip-connected with output from down4
        self.up3 = Upsample(128+128, 128, proj_ratio=8) # skip-connected with output from down3
        self.up4 = Upsample(128+64, 64) # skip-connected with output from down2
        self.up5 = Upsample(64+32, 32) # skip-connected with output from down1
        self.up6 = Upsample(32+in_channels, in_channels, activation=False) # skip-connected with output from input
        self.tanh = nn.Tanh()
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    init.uniform_(m.bias)
                init.xavier_uniform_(m.weight)
        
    def forward(self, img):
        # encoder
        x1 = self.down1(img) 
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        x5 = self.down5(x4)
        x6 = self.down6(x5)
        #decoder
        x = self.up1(x6, x5)
        x = self.up2(x, x4)
        x = self.up3(x, x3)
        x = self.up4(x, x2)
        x = self.up5(x, x1)
        x = self.up6(x, img)
        x = self.tanh(x)
        
        return x
 
