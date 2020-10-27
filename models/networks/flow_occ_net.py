"""network to estimate optical flow and occlusion mask"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class OpticalFlowEstimator(nn.Module): 
    def __init__(self, input_channels, level, highest_resolution = False):
        self.highest_res = highest_resolution
        self.conv1 = nn.Conv2d(input_channels, 128, 3, 1, 1)
        self.conv2 = nn.Conv2d(128, 128, 3, 1, 1)
        self.conv3 = nn.Conv2d(128, 96, 3, 1, 1)
        self.conv4 = nn.Conv2d(96, 64, 3, 1, 1)
        self.conv5 = nn.Conv2d(64, 32, 3, 1, 1)
        self.conv6 = nn.Conv2d(32,4, 3, 1, 1)

        self.upconv1 = nn.ConvTranspose2d(4, 4, 4, 2, 1)
        self.upconv2 = nn.ConvTranspose2d(4, 4, 4, 2, 1)

    def forward(self, x): 
        c1 = F.leaky_relu(self.conv1(x), negative_slope= 0.1)
        c2 = F.leaky_relu(self.conv2(c1), negative_slope= 0.1)
        c3 = F.leaky_relu(self.conv3(c2), negative_slope= 0.1)
        c4 = F.leaky_relu(self.conv4(c3), negative_slope= 0.1)
        f_lev = F.leaky_relu(self.conv5(c4), negative_slope=0.1)
        w_lev = self.conv6(f_lev)
        if self.highest_res: 
            return (f_lev, w_lev)
        else: 
            flow_up = self.upconv1(w_lev)
            feature_up = self.upconv2(f_lev)
            return(w_lev, flow_up, feature_up)



