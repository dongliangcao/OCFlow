from models.networks.feature_pyramid_net import FeaturePyramidNet
from models.networks.flow_occ_net import OpticalFlowEstimator, CostVolumeLayer
from models.networks.context_net import ContextNetwork

import torch
import torch.nn as nn
import torch.nn.functional as F

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
        
        up_flow, up_feature = None, None
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
                flow_displacement = up_flow * 20.0 / (2.0 ** level) #flow has size Bx2xHxW
                warped2 = self.warp(feat2, flow_displacement)
            corr = self.correlation_layer(feat1, warped2)
            
            input_list = [corr, feat1]
            if not first_iteration:
                input_list.append(up_flow)
                input_list.append(up_feature)
            estimator_input = torch.cat(input_list, dim=1)
            
            if last_iteration:
                feature, flow = self.opticalflow_estimators[i](estimator_input)
            else:
                flow, up_flow, up_feature = self.opticalflow_estimators[i](estimator_input, h_up, w_up)
            
        residual_flow = self.context_network(torch.cat([feature, flow], dim=1))
        refined_flow = flow + residual_flow
        
        predicted_flow = self.upsample(refined_flow)*20.0
        
        return predicted_flow
        