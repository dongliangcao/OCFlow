from models.networks.feature_pyramid_net import FeaturePyramidNet
from models.networks.cost_volume_net import CostVolumeLayer
from models.networks.context_net import ContextNetwork

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import init

class OpticalFlowEstimator(nn.Module): 
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
        
class OcclusionEstimator(nn.Module): 
    def __init__(self, input_channels, level, highest_resolution = False):
        super(OcclusionEstimator, self).__init__()
        self.highest_res = highest_resolution
        self.conv1 = nn.Conv2d(input_channels, 128, 3, 1, 1)
        self.conv2 = nn.Conv2d(128, 96, 3, 1, 1)
        self.conv3 = nn.Conv2d(96, 64, 3, 1, 1)
        self.conv4 = nn.Conv2d(64, 32, 3, 1, 1)
        self.feat_layer = nn.Conv2d(32, 16, 3, 1, 1)
        self.mask_layer = nn.Conv2d(16, 1, 3, 1, 1)

        self.upconv1 = nn.ConvTranspose2d(16, 1, 3, 2, 1)
        self.upconv2 = nn.ConvTranspose2d(1, 1, 3, 2, 1)
    
    def forward(self, features, h_up =None, w_up = None):
        c1 = F.leaky_relu(self.conv1(features), negative_slope= 0.1)
        c2 = F.leaky_relu(self.conv2(c1), negative_slope= 0.1)
        c3 = F.leaky_relu(self.conv3(c2), negative_slope= 0.1)
        c4 = F.leaky_relu(self.conv4(c3), negative_slope= 0.1)
        feat = F.leaky_relu(self.feat_layer(c4), negative_slope= 0.1)
        occ = self.mask_layer(feat)  
        if self.highest_res:
            occ = torch.sigmoid(10 * occ)
            return occ
        else: 
            occ = torch.sigmoid(occ)
            feature_up = torch.sigmoid(self.upconv1(feat, output_size = (h_up, w_up)))
            occ_up = torch.sigmoid(self.upconv2(occ, output_size =(h_up, w_up)))
            return (occ, occ_up, feature_up)

class FlowOccNet(nn.Module):
    def __init__(self):
        super(FlowOccNet,self).__init__()
        # correlation calculation
        self.correlation_layer = CostVolumeLayer()
        # feature pyramid network
        self.feature_pyramid_network = FeaturePyramidNet()
        # optical flow estimator
        self.opticalflow_estimators = nn.ModuleList()
        for (d, l) in zip([277, 213, 181, 149, 117], [6, 5, 4, 3, 2]):
            self.opticalflow_estimators.append(OpticalFlowEstimator(d, level=l, highest_resolution=(l==2)))
        # occlusion estimatr
        self.occlusion_estimators = nn.ModuleList()
        for (d, l) in zip([392, 258, 194, 130, 66], [6, 5, 4, 3, 2]):
            self.occlusion_estimators.append(OcclusionEstimator(d, level=l, highest_resolution=(l==2)))
        # residual flow estimator
        self.context_network = ContextNetwork(34)
        # upsampler
        self.upsample1 = nn.Upsample(scale_factor=4, mode='bilinear')
        self.upsample2 = nn.Upsample(scale_factor=4, mode='bilinear')
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight.data, mode='fan_in')
                if m.bias is not None:
                    m.bias.data.zero_()
    
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
    
    def forward(self, inputs, training=False, mask=None):
        im1 = inputs[:, 0:3, :, :]
        im2 = inputs[:, 3:6, :, :]
        input_h = im1.size()[2]
        input_w = im1.size()[3]
        pyramid_im1 = self.feature_pyramid_network(im1)
        pyramid_im2 = self.feature_pyramid_network(im2)

        # upsampled flow and flow feature
        flow_up, flow_feature_up = None, None
        # 
        flow, feature = None, None
        # upsampled occlusion map and occlusion feature
        occ_up, occ_feature_up = None, None

        for i, (feat1, feat2) in enumerate(zip(pyramid_im1,pyramid_im2)):
            level = 6 - i
            first_iteration = (i==0)
            last_iteration  = (level==2)
            if not last_iteration: 
                h_up = pyramid_im1[i+1].size()[2]
                w_up = pyramid_im1[i+1].size()[3]

            if first_iteration:
                warped2 = feat2
            else:
                flow_displacement = flow_up * 20.0 / (2.0 ** level) #flow has size Bx2xHxW
                warped2 = self.warp(feat2, flow_displacement)
            
            # occlusion map prediction
            occ_inputs = torch.cat([feat1, warped2], dim=1)
            if not first_iteration:  # all but the first iteration
                occ_inputs = torch.cat([occ_inputs, occ_feature_up, occ_up], dim=1)
            if last_iteration:
                occ = self.occlusion_estimators[i](occ_inputs)
            else:
                occ, occ_up, occ_feature_up = self.occlusion_estimators[i](occ_inputs, h_up, w_up)
            
            # exclude occluded pixels before correlation calculation
            warped2 *= occ
            corr = self.correlation_layer(feat1, warped2)

            # optical flow prediction
            flow_inputs = [corr, feat1]
            if not first_iteration:  # all but the first iteration
                flow_inputs.append(flow_up)
                flow_inputs.append(flow_feature_up)
            flow_inputs = torch.cat(flow_inputs, dim=1)

            if last_iteration:
                flow, flow_feature = self.opticalflow_estimators[i](flow_inputs)
            else:
                flow, flow_up, flow_feature_up = self.opticalflow_estimators[i](flow_inputs, h_up, w_up)

        # refine residual flow
        residual_flow = self.context_network(torch.cat([flow_feature, flow], dim=1))
        refined_flow = flow + residual_flow


        predicted_flow = self.upsample1(refined_flow)
        predicted_occ = self.upsample2(occ)
        return predicted_flow, predicted_occ 






