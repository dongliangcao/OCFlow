from models.networks.feature_pyramid_net import FeaturePyramidNet
from models.networks.cost_volume_net import CostVolumeLayer
from models.networks.context_net import ContextNetwork

import torch
import torch.nn as nn
import torch.nn.functional as F

class OpticalFlowEstimator(nn.Module): 
    """
    Network for predicting optical flow from cost volumes of masked, warped feature of second frame and feature of first frame. 
    """
    def __init__(self, input_channels, level, highest_resolution = False):
        super(OpticalFlowEstimator, self).__init__()
        self.highest_res = highest_resolution
        self.conv1 = nn.Conv2d(input_channels, 128, 3, 1, 1)
        self.conv2 = nn.Conv2d(128, 128, 3, 1, 1)
        self.conv3 = nn.Conv2d(128, 96, 3, 1, 1)
        self.conv4 = nn.Conv2d(96, 64, 3, 1, 1)
        self.conv5 = nn.Conv2d(64, 32, 3, 1, 1)
        self.conv6 = nn.Conv2d(32,2, 3, 1, 1)

        self.upconv1 = nn.ConvTranspose2d(2, 2, 3, 2, 1)
        self.upconv2 = nn.ConvTranspose2d(32, 2, 3, 2, 1)

    def forward(self, x, h_up =None, w_up = None): 
        c1 = F.leaky_relu(self.conv1(x), negative_slope= 0.1)
        c2 = F.leaky_relu(self.conv2(c1), negative_slope= 0.1)
        c3 = F.leaky_relu(self.conv3(c2), negative_slope= 0.1)
        c4 = F.leaky_relu(self.conv4(c3), negative_slope= 0.1)
        feat = F.leaky_relu(self.conv5(c4), negative_slope=0.1)
        flow = self.conv6(feat)
        if self.highest_res: 
            return (flow, feat)
        else: 
            flow_up = self.upconv1(flow, output_size = (h_up, w_up)) #(2,6,20)
            feature_up = self.upconv2(feat, output_size = (h_up, w_up))
            return(flow, flow_up, feature_up)

class FlowNet(nn.Module):
    """optical flow prediction network"""
    def __init__(self):
        super().__init__()
        self.correlation_layer = CostVolumeLayer()
        self.feature_pyramid_network = FeaturePyramidNet()
        self.opticalflow_estimators = nn.ModuleList()
        for (d, l) in zip([277, 213, 181, 149, 117], [6, 5, 4, 3, 2]):
            self.opticalflow_estimators.append(OpticalFlowEstimator(d, level=l, highest_resolution=(l==2)))
        self.context_network = ContextNetwork(34)
        self.upsample = nn.Upsample(scale_factor=4, mode='bilinear')
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.bias is not None:
                    init.uniform_(m.bias)
                init.xavier_uniform_(m.weight)

            if isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    init.uniform_(m.bias)
                init.xavier_uniform_(m.weight)
    
    def warp(self, img, flow):
        """
        warp an image/tensor (im2) back to im1, according to the optical flow

        x: [B, C, H, W] (im2)
        flo: [B, 2, H, W] flow

        """
        B, C, H, W = img.size()
        # create mesh grid
        xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
        yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
        xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
        yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
        grid = torch.cat((xx, yy), 1).float()
        # cast into cuda
        if img.is_cuda:
            grid = grid.cuda()
        # require gradient
        grid.requires_grad = True
        vgrid = grid + flow
        # scale grid to [-1, 1] to support grid_sample function in pytorch
        # https://pytorch.org/docs/stable/nn.functional.html#grid-sample
        vgrid[:,0,:,:] = 2.0 * vgrid[:,0,:,:].clone() / max(W-1, 1) - 1.0
        vgrid[:,1,:,:] = 2.0 * vgrid[:,1,:,:].clone() / max(H-1, 1) - 1.0
        # permute vgrid to size [B, H, W, 2] to support grid_sample function
        vgrid = vgrid.permute(0, 2, 3, 1)
        
        output = F.grid_sample(img, vgrid, align_corners=False)
        
        return output
    
    def forward(self, inputs):
        im1 = inputs[:, 0:3, :, :]
        im2 = inputs[:, 3:6, :, :]
        input_h, input_w = im1.size()[2], im1.size()[3]
        pyramid_im1 = self.feature_pyramid_network(im1)
        pyramid_im2 = self.feature_pyramid_network(im2)
        
        flow_up, flow_feature_up = None, None
        feature, flow = None, None
        
        for i, (feat1, feat2) in enumerate(zip(pyramid_im1, pyramid_im2)):
            level = 6 - i
            first_iteration = (i==0)
            last_iteration  =(level==2)
            if not last_iteration: 
                h_up = pyramid_im1[i+1].size()[2]
                w_up = pyramid_im1[i+1].size()[3]

            if first_iteration:
                warped2 = feat2
            else:
                flow_displacement = flow_up * 20.0 / (2.0 ** level) #flow has size Bx2xHxW
                warped2 = self.warp(feat2, flow_displacement)
            corr = self.correlation_layer(feat1, warped2)
            
            flow_input = [corr, feat1]
            if not first_iteration:
                flow_input.append(flow_up)
                flow_input.append(flow_feature_up)
            flow_input = torch.cat(flow_input, dim=1)
            
            if last_iteration:
                flow, feature = self.opticalflow_estimators[i](flow_input)
            else:
                flow, flow_up, flow_feature_up = self.opticalflow_estimators[i](flow_input, h_up, w_up)
            
        residual_flow = self.context_network(torch.cat([feature, flow], dim=1))
        refined_flow = flow + residual_flow
        # I omit the multiplication with 20
        predicted_flow = self.upsample(refined_flow)# *20.0
        
        return predicted_flow
        
