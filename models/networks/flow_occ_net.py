"""network to estimate optical flow and occlusion mask"""
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
        f_lev = F.leaky_relu(self.conv5(c4), negative_slope=0.1)
        w_lev = self.conv6(f_lev)
        if self.highest_res: 
            return (f_lev, w_lev)
        else: 
            flow_up = self.upconv1(w_lev, output_size = (h_up, w_up)) #(2,6,20)
            feature_up = self.upconv2(f_lev, output_size = (h_up, w_up))
            return(w_lev, flow_up, feature_up)
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
        occ_mask = torch.sigmoid(self.mask_layer(feat))
        if self.highest_res: 
            return occ_mask
        else: 
            features_up = torch.sigmoid(self.upconv1(feat, output_size = (h_up, w_up)))
            occ_mask_up = torch.sigmoid(self.upconv2(occ_mask, output_size =(h_up, w_up)))
            return (occ_mask, features_up, occ_mask_up)
class CostVolumeLayer(nn.Module):
    """
    Calculate the cost volume between the warped feature and the reference feature 
    """
    def __init__(self, search_range=4):
        super(CostVolumeLayer, self).__init__()
        self.window = search_range
    def forward(self,x, warped): 
        """
        Args: 
        x: input feature, torch.Tensor [B, C, H, W]
        warped: warped feature, torch.Tensor[B,C,H,W]
        Returns: 
        stacked: cost volume tensor, torch.Tensor [B, (search_range*2+1)**2, H, W] 
        """
        total = []
        keys = []

        row_shifted = [warped]

        for i in range(self.window+1):
            if i != 0:
                row_shifted = [F.pad(row_shifted[0], (0,0,0,1)), F.pad(row_shifted[1], (0,0,1,0))]

                row_shifted = [row_shifted[0][:, :, 1:, :], row_shifted[1][:, :, :-1, :]]

            for side in range(len(row_shifted)):
                total.append(torch.mean(row_shifted[side] * x, dim = 1))
                keys.append([i * (-1) ** side, 0])
                col_previous = [row_shifted[side], row_shifted[side]]
                for j in range(1, self.window+1):
                    col_shifted = [F.pad(col_previous[0], (0,1)), F.pad(col_previous[1], (1,0))]

                    col_shifted = [col_shifted[0][:, :, :, 1:], col_shifted[1][:, :, :, :-1]]

                    for col_side in range(len(col_shifted)):
                        total.append(torch.mean(col_shifted[col_side] * x, dim=1))
                        keys.append([i * (-1) ** side, j * (-1) ** col_side])
                    col_previous = col_shifted

            if i == 0:
                row_shifted *= 2

        total = [t for t, _ in sorted(zip(total, keys), key=lambda pair: pair[1])]
        stacked = torch.stack(total, dim =1)

        return stacked / ((2.0*self.window+1)**2.0)



