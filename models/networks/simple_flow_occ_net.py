import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

class Downsample(nn.Module):
    def __init__(self, in_channels, out_channels, proj_ratio=4):
        super().__init__()
        inter_channels = in_channels // proj_ratio
        self.conv1 = nn.Conv2d(in_channels, inter_channels, 2, stride=2, bias=False)
        self.bn1 = nn.BatchNorm2d(inter_channels)
        self.lrelu1 = nn.LeakyReLU(0.1)
        
        self.conv2 = nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False)
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
    def __init__(self, in_channels, out_channels, proj_ratio=4):
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
        self.lrelu3 = nn.LeakyReLU(0.1)
        
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
        
        
class SimpleFlowOccNet(nn.Module):
    """"
    A simple optical flow prediction network take the input as concatenated image pairs and predict the optical flow and occlusion map
    """
    def __init__(self, in_channels=6):
        super(SimpleFlowOccNet, self).__init__()
        ## Encoder part
        self.down1 = Downsample(in_channels, 16, proj_ratio=1)
        
        self.down2 = Downsample(16, 32, proj_ratio=2)
        
        self.down3 = Downsample(32, 64, proj_ratio=4)
        
        self.down4 = Downsample(64, 96, proj_ratio=4)
        
        self.down5 = Downsample(96, 128, proj_ratio=4)
        
        ## Decoder part
        self.up1 = Upsample(128+96+3, 96) # skip-connected with output from down4
        self.up2 = Upsample(96+64+3, 64)  # skip-connected with output from down3
        self.up3 = Upsample(64+32+3, 32)  # skip-connected with output from down2
        self.up4 = Upsample(32+16+3, 16)  # skip-connected with output from down1


        ## flow prediction
        self.predict_flow5 = predict_flow(128)
        self.predict_flow4 = predict_flow(96)
        self.predict_flow3 = predict_flow(64)
        self.predict_flow2 = predict_flow(32)
        self.predict_flow1 = predict_flow(16)
        self.upsample_flow = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        
        ## occlusion prediction
        self.predict_occ5 = predict_occ(128)
        self.predict_occ4 = predict_occ(96)
        self.predict_occ3 = predict_occ(64)
        self.predict_occ2 = predict_occ(32)
        self.predict_occ1 = predict_occ(16, is_last=True)
        self.upsample_occ = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        
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

        #decoder
        flow5 = self.predict_flow5(x5)
        occ5 = self.predict_occ5(x5)
        x = torch.cat((x5, flow5, occ5), dim=1)
        x = self.up1(x, x4)
        
        flow4 = self.predict_flow4(x)
        occ4 = self.predict_occ4(x)
        x = torch.cat((x, flow4, occ4), dim=1)
        x = self.up2(x, x3)
        
        flow3 = self.predict_flow3(x)
        occ3 = self.predict_occ3(x)
        x = torch.cat((x, flow3, occ3), dim=1)
        x = self.up3(x, x2)
        
        flow2 = self.predict_flow2(x)
        occ2 = self.predict_occ2(x)
        x = torch.cat((x, flow2, occ2), dim=1)
        x = self.up4(x, x1)
        
        flow = self.upsample_flow(self.predict_flow1(x))
        occ = self.predict_occ1(x)
        occ_soft = torch.sigmoid(10 * self.upsample_occ(occ))
        occ_hard = torch.where(occ_soft > 0.5, 1.0, 0.0)
        return flow, (occ_hard - occ_soft).detach() + occ_soft

def predict_flow(in_channels):
    return nn.Sequential(
        conv(in_channels, 32),
        conv(32, 16),
        conv(16, 2, activation=False)
    )    

def predict_occ(in_channels, is_last=False):
    return nn.Sequential(
        conv(in_channels, 32),
        conv(32, 16),
        conv(16, 1, activation=False),
        nn.Sigmoid() if not is_last else nn.Identity()
    )

def conv(in_channels, out_channels, kernel_size=3, stride=1, activation=True):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=(kernel_size-1)//2),
        nn.LeakyReLU(0.1,inplace=True) if activation else nn.Identity()
    )
    

