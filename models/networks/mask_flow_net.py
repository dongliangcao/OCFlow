from models.networks.feature_pyramid_net import FeaturePyramidNet
from models.networks.flow_occ_net import OpticalFlowEstimator, OcclusionEstimator, CostVolumeLayer
from models.networks.context_net import ContextNetwork
from models.networks.warping_layer import Warping
import torch
import torch.nn as nn
import torch.nn.functional as F
class MaskFlowNet(nn.Module):
    def __init__(self, occlusion = True, mean_pixel = None):
        super().__init__()
        self.occlusion = occlusion
        self.mean_pixel = torch.zeros(3, dtype= torch.float32, requires_grad= False)
        if mean_pixel:
            self.mean_pixel =  mean_pixel
        self.correlation_layer = CostVolumeLayer()
        self.feature_pyramid_network = FeaturePyramidNet()
        self.opticalflow_estimators = nn.ModuleList()
        for (d, l) in zip([277, 213, 181, 149, 117], [6, 5, 4, 3, 2]):
            self.opticalflow_estimators.append(OpticalFlowEstimator(d, level=l, highest_resolution=(l==2)))
        self.context_network = ContextNetwork(34)
        if occlusion:
            self.occlusion_estimators = nn.ModuleList()
            for (d, l) in zip([392, 258, 194, 130, 66], [6, 5, 4, 3, 2]):
                self.occlusion_estimators.append(OcclusionEstimator(d, level=l, highest_resolution=(l==2)))
        
    def forward(self, inputs, training=False, mask=None):
        im1 = inputs[:, 0, :, :, :]
        im2 = inputs[:, 1, :, :, :]
        input_h = im1.size()[2]
        input_w = im1.size()[3]
        pyramid_im1 = self.feature_pyramid_network(im1)
        pyramid_im2 = self.feature_pyramid_network(im2)

        up_flow, up_feature = None, None
        features, flow = None, None
        occ_features_up, occ_mask_up = [], []
        flows = []  # multi-scale output

        for i, (feat1, feat2) in enumerate(zip(pyramid_im1,pyramid_im2)):
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
                warping_layer = Warping() #feat2 has size BxCxHxW
                warped2 = warping_layer(feat2, flow_displacement)
            if self.occlusion:
                occ_inputs = torch.cat([feat1, warped2], dim=1)
                if not first_iteration:  # all but the first iteration
                    occ_inputs = torch.cat([occ_inputs, occ_features_up.pop(0), occ_mask_up.pop(0)], dim=1)
                if last_iteration:
                    occ_mask = self.occlusion_estimators[i](occ_inputs)
                else:
                    occ_mask, feat_up, mask_up = self.occlusion_estimators[i](occ_inputs, h_up, w_up)
                    occ_features_up.append(feat_up)
                    occ_mask_up.append(mask_up)
                warped2 *= occ_mask
            cv2 = self.correlation_layer(feat1, warped2)

            input_list = [cv2,feat1]
            if not first_iteration:  # all but the first iteration
                input_list.append(up_flow)
                input_list.append(up_feature)
            estimator_input = torch.cat(input_list, dim=1)

            if last_iteration:
                features, flow = self.opticalflow_estimators[i](estimator_input)
            else:
                flow, up_flow, up_feature = self.opticalflow_estimators[i](estimator_input, h_up, w_up)
                flows.append(flow)

        residual_flow = self.context_network(torch.cat([features, flow], dim=1))
        refined_flow = flow + residual_flow
        flows.append(refined_flow)

        #prediction = tf.multiply(tf.image.resize(refined_flow, size=(input_h, input_w)), 20.0, name='final_prediction')
        self.upsample = nn.Upsample((input_h, input_w), mode='bilinear')
        predicted_flow = self.upsample(refined_flow)*20.0
        predicted_occ_mask = self.upsample(occ_mask)
        return predicted_flow, predicted_occ_mask 






