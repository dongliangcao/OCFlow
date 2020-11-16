import torch
import torch.nn.functional as F
from torch import nn, cuda
from torch.autograd import Variable     

class Downsample(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, normalize=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, kernel_size//2, bias=not normalize)
        self.normalize = nn.BatchNorm2d(out_channels) if normalize else None
        self.activation = nn.ReLU()
        
    def forward(self, x):
        out = self.conv(x)
        if self.normalize:
            out = self.normalize(out)
        out = self.activation(out)
        return out
    
class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, normalize=True, activation=True):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, 1, kernel_size//2, bias=not normalize)
        self.normalize = nn.BatchNorm2d(out_channels) if normalize else None
        self.activation = nn.LeakyReLU(0.2) if activation else None
        
    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        
        x1 = F.pad(x1, [diffX//2, diffX - diffX//2, diffY//2, diffY - diffY//2])
        x = torch.cat([x2, x1], dim=1)
        
        out = self.conv(x)
        if self.normalize:
            out = self.normalize(out)
        if self.activation:
            out = self.activation(out)
        return out
        
        
class SceneCompletionNet(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        ## Encoder part
        self.down1 = Downsample(in_channels, 64, 7, 2, normalize=False)
        self.down2 = Downsample(64, 128, 5, 2)
        self.down3 = Downsample(128, 256, 5, 2)
        self.down4 = Downsample(256, 512, 3, 2)
        self.down5 = Downsample(512, 512, 3, 2)
        self.down6 = Downsample(512, 512, 3, 2)
        self.down7 = Downsample(512, 512, 3, 2)
        
        ## Decoder part
        self.up1 = Upsample(512+512, 512, 3) # skip-connected with output from down6
        self.up2 = Upsample(512+512, 512, 3) # skip-connected with output from down5
        self.up3 = Upsample(512+512, 512, 3) # skip-connected with output from down4
        self.up4 = Upsample(512+256, 256, 3) # skip-connected with output from down3
        self.up5 = Upsample(256+128, 128, 3) # skip-connected with output from down2
        self.up6 = Upsample(128+64, 64, 3) # skip-connected with output from down1
        self.up7 = Upsample(64+in_channels, in_channels, 3, activation=False) # skip-connected with output from input
        self.tanh = nn.Tanh()
        
    def forward(self, img):
        # encoder
        x1 = self.down1(img) 
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        x5 = self.down5(x4)
        x6 = self.down6(x5)
        x7 = self.down7(x6)
        #decoder
        x = self.up1(x7, x6)
        x = self.up2(x, x5)
        x = self.up3(x, x4)
        x = self.up4(x, x3)
        x = self.up5(x, x2)
        x = self.up6(x, x1)
        x = self.up7(x, img)
        x = self.tanh(x)
        
        return x
 
