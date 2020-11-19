"""code adapt from https://github.com/NVIDIA/flownet2-pytorch"""
import torch
import torch.nn as nn
from torch.nn import init

import math
import numpy as np


class FlowOccNetS(nn.Module):
    """
    optical flow/occlusion predicion network, structure similar to flownet-s
    """
    def __init__(self, input_channels = 6, batchNorm=True):
        super(FlowOccNetS,self).__init__()

        self.batchNorm = batchNorm
        # convolution part
        self.conv1   = conv(self.batchNorm,  input_channels,   64, kernel_size=7, stride=2)
        self.conv2   = conv(self.batchNorm,  64,  128, kernel_size=5, stride=2)
        self.conv3   = conv(self.batchNorm, 128,  256, kernel_size=5, stride=2)
        self.conv3_1 = conv(self.batchNorm, 256,  256)
        self.conv4   = conv(self.batchNorm, 256,  512, stride=2)
        self.conv4_1 = conv(self.batchNorm, 512,  512)
        self.conv5   = conv(self.batchNorm, 512,  512, stride=2)
        self.conv5_1 = conv(self.batchNorm, 512,  512)
        self.conv6   = conv(self.batchNorm, 512, 1024, stride=2)
        self.conv6_1 = conv(self.batchNorm,1024, 1024)
        
        # deconv part
        self.deconv5 = deconv(1024,512)
        self.deconv4 = deconv(1027,256)
        self.deconv3 = deconv(771,128)
        self.deconv2 = deconv(387,64)

        # flow prediction part
        self.predict_flow6 = predict_flow(1024)
        self.predict_flow5 = predict_flow(1027)
        self.predict_flow4 = predict_flow(771)
        self.predict_flow3 = predict_flow(387)
        self.predict_flow2 = predict_flow(195)
        
        # occlusion prediction part
        self.predict_occ6 = predict_occlusion(1024)
        self.predict_occ5 = predict_occlusion(1027)
        self.predict_occ4 = predict_occlusion(771)
        self.predict_occ3 = predict_occlusion(387)
        self.predict_occ2 = predict_occlusion(195)

        # flow upsample part
        self.upsampled_flow6_to_5 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)
        self.upsampled_flow5_to_4 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)
        self.upsampled_flow4_to_3 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)
        self.upsampled_flow3_to_2 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)
        
        # occlusion upsample part
        self.upsampled_occ6_to_5 = nn.ConvTranspose2d(1, 1, 4, 2, 1, bias=False)
        self.upsampled_occ5_to_4 = nn.ConvTranspose2d(1, 1, 4, 2, 1, bias=False)
        self.upsampled_occ4_to_3 = nn.ConvTranspose2d(1, 1, 4, 2, 1, bias=False)
        self.upsampled_occ3_to_2 = nn.ConvTranspose2d(1, 1, 4, 2, 1, bias=False)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.bias is not None:
                    init.uniform_(m.bias)
                init.xavier_uniform_(m.weight)

            if isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    init.uniform_(m.bias)
                init.xavier_uniform_(m.weight)
        
        # upsampler
        self.upsample1 = nn.Upsample(scale_factor=4, mode='bilinear')
        self.upsample2 = nn.Upsample(scale_factor=4, mode='bilinear')
        
    def forward(self, x):
        out_conv1 = self.conv1(x)

        out_conv2 = self.conv2(out_conv1)
        out_conv3 = self.conv3_1(self.conv3(out_conv2))
        out_conv4 = self.conv4_1(self.conv4(out_conv3))
        out_conv5 = self.conv5_1(self.conv5(out_conv4))
        out_conv6 = self.conv6_1(self.conv6(out_conv5))

        flow6       = self.predict_flow6(out_conv6)
        flow6_up    = self.upsampled_flow6_to_5(flow6)
        occ6        = self.predict_occ6(out_conv6)
        occ6_up     = self.upsampled_occ6_to_5(occ6)
        out_deconv5 = self.deconv5(out_conv6)
        
        concat5 = torch.cat((out_conv5,out_deconv5,flow6_up,occ6_up),1)
        flow5       = self.predict_flow5(concat5)
        flow5_up    = self.upsampled_flow5_to_4(flow5)
        occ5        = self.predict_occ5(concat5)
        occ5_up     = self.upsampled_occ5_to_4(occ5)
        out_deconv4 = self.deconv4(concat5)
        
        concat4 = torch.cat((out_conv4,out_deconv4,flow5_up,occ5_up),1)
        flow4       = self.predict_flow4(concat4)
        flow4_up    = self.upsampled_flow4_to_3(flow4)
        occ4        = self.predict_occ4(concat4)
        occ4_up     = self.upsampled_occ4_to_3(occ4)
        out_deconv3 = self.deconv3(concat4)
        
        concat3 = torch.cat((out_conv3,out_deconv3,flow4_up,occ4_up),1)
        flow3       = self.predict_flow3(concat3)
        flow3_up    = self.upsampled_flow3_to_2(flow3)
        occ3        = self.predict_occ3(concat3)
        occ3_up     = self.upsampled_occ3_to_2(occ3)
        out_deconv2 = self.deconv2(concat3)

        concat2 = torch.cat((out_conv2,out_deconv2,flow3_up,occ3_up),1)
        flow2 = self.predict_flow2(concat2)
        occ2 = self.predict_occ2(concat2)

        return self.upsample1(flow2), self.upsample2(occ2)

def conv(batchNorm, in_planes, out_planes, kernel_size=3, stride=1):
    if batchNorm:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.LeakyReLU(0.1,inplace=True)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=True),
            nn.LeakyReLU(0.1,inplace=True)
        )

def deconv(in_planes, out_planes):
    return nn.Sequential(
        nn.ConvTranspose2d(in_planes, out_planes, kernel_size=4, stride=2, padding=1, bias=True),
        nn.LeakyReLU(0.1,inplace=True)
    )

def predict_flow(in_planes):
    return nn.Conv2d(in_planes,2,kernel_size=3,stride=1,padding=1,bias=True)

def predict_occlusion(in_planes):
    return nn.Sequential(
        nn.Conv2d(in_planes,1,kernel_size=3,stride=1,padding=1,bias=True),
        nn.Sigmoid()
    )