import torch
import torch.nn as nn
from torch.nn import init

import math
import numpy as np

from models.networks.cost_volume_net import CostVolumeLayer

class OcclusionNetC(nn.Module):
    """
    occlusion network prediction, structure similar to flownet-c
    """
    def __init__(self, batchNorm=True):
        super(OcclusionNetC,self).__init__()

        self.batchNorm = batchNorm

        self.conv1   = conv(self.batchNorm,   3,   64, kernel_size=7, stride=2)
        self.conv2   = conv(self.batchNorm,  64,  128, kernel_size=5, stride=2)
        self.conv3   = conv(self.batchNorm, 128,  256, kernel_size=5, stride=2)
        self.conv_redir  = conv(self.batchNorm, 256,   32, kernel_size=1, stride=1)

        self.corr = CostVolumeLayer(10)

        self.corr_activation = nn.LeakyReLU(0.1,inplace=True)
        self.conv3_1 = conv(self.batchNorm, 473,  256)
        self.conv4   = conv(self.batchNorm, 256,  512, stride=2)
        self.conv4_1 = conv(self.batchNorm, 512,  512)
        self.conv5   = conv(self.batchNorm, 512,  512, stride=2)
        self.conv5_1 = conv(self.batchNorm, 512,  512)
        self.conv6   = conv(self.batchNorm, 512, 1024, stride=2)
        self.conv6_1 = conv(self.batchNorm,1024, 1024)

        self.deconv5 = deconv(1024,512)
        self.deconv4 = deconv(1025,256)
        self.deconv3 = deconv(769,128)
        self.deconv2 = deconv(385,64)

        self.predict_occ6 = predict_occlusion(1024)
        self.predict_occ5 = predict_occlusion(1025)
        self.predict_occ4 = predict_occlusion(769)
        self.predict_occ3 = predict_occlusion(385)
        self.predict_occ2 = predict_occlusion(193)

        self.upsampled_occ6_to_5 = nn.ConvTranspose2d(1, 1, 4, 2, 1, bias=True)
        self.upsampled_occ5_to_4 = nn.ConvTranspose2d(1, 1, 4, 2, 1, bias=True)
        self.upsampled_occ4_to_3 = nn.ConvTranspose2d(1, 1, 4, 2, 1, bias=True)
        self.upsampled_occ3_to_2 = nn.ConvTranspose2d(1, 1, 4, 2, 1, bias=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.bias is not None:
                    init.uniform_(m.bias)
                init.xavier_uniform_(m.weight)

            if isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    init.uniform_(m.bias)
                init.xavier_uniform_(m.weight)
                # init_deconv_bilinear(m.weight)
        self.upsample1 = nn.Upsample(scale_factor=4, mode='bilinear')

    def forward(self, x):
        x1 = x[:,0:3,:,:]
        x2 = x[:,3::,:,:]

        out_conv1a = self.conv1(x1)
        out_conv2a = self.conv2(out_conv1a)
        out_conv3a = self.conv3(out_conv2a)

        # OccnetC bottom input stream
        out_conv1b = self.conv1(x2)
        
        out_conv2b = self.conv2(out_conv1b)
        out_conv3b = self.conv3(out_conv2b)

        # Merge streams
        out_corr = self.corr(out_conv3a, out_conv3b)
        out_corr = self.corr_activation(out_corr)

        # Redirect top input stream and concatenate
        out_conv_redir = self.conv_redir(out_conv3a)

        in_conv3_1 = torch.cat((out_conv_redir, out_corr), 1)

        # Merged conv layers
        out_conv3_1 = self.conv3_1(in_conv3_1)

        out_conv4 = self.conv4_1(self.conv4(out_conv3_1))

        out_conv5 = self.conv5_1(self.conv5(out_conv4))
        out_conv6 = self.conv6_1(self.conv6(out_conv5))

        occ6       = self.predict_occ6(out_conv6)
        occ6_up    = self.upsampled_occ6_to_5(occ6)
        out_deconv5 = self.deconv5(out_conv6)

        concat5 = torch.cat((out_conv5,out_deconv5,occ6_up),1)

        occ5       = self.predict_occ5(concat5)
        occ5_up    = self.upsampled_occ5_to_4(occ5)
        out_deconv4 = self.deconv4(concat5)
        concat4 = torch.cat((out_conv4,out_deconv4,occ5_up),1)

        occ4       = self.predict_occ4(concat4)
        occ4_up    = self.upsampled_occ4_to_3(occ4)
        out_deconv3 = self.deconv3(concat4)
        concat3 = torch.cat((out_conv3_1,out_deconv3,occ4_up),1)

        occ3       = self.predict_occ3(concat3)
        occ3_up    = self.upsampled_occ3_to_2(occ3)
        out_deconv2 = self.deconv2(concat3)
        concat2 = torch.cat((out_conv2a,out_deconv2,occ3_up),1)

        occ2 = self.predict_occ2(concat2)

        return self.upsample1(occ2)

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
    
def predict_occlusion(in_planes):
    return nn.Sequential(
        nn.Conv2d(in_planes,1,kernel_size=3,stride=1,padding=1,bias=True),
        nn.Sigmoid()
    )

def deconv(in_planes, out_planes):
    return nn.Sequential(
        nn.ConvTranspose2d(in_planes, out_planes, kernel_size=4, stride=2, padding=1, bias=True),
        nn.LeakyReLU(0.1,inplace=True)
    )