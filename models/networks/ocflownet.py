import torch
import torch.nn as nn
import torch.nn.functional as F
from models.networks.mask_flow_net import MaskFlowNet
from models.networks.warping_layer import Warping
#from models.networks.scene_completion_net import SceneCompletionNet
from models.networks.image_inpainting_net import SceneCompletionNet
class OCFlowNet(nn.Module): 
    def __init__(self):
        super(OCFlowNet,self).__init__()
        self.mask_flow_net = MaskFlowNet()
        self.completion_net = SceneCompletionNet()
        self.warping = Warping()
    def forward(self, batch):
        I1 = batch[:, 0, :, :, :]
        I2 = batch[:, 1, :, :, :]
        F12, O_s = self.mask_flow_net(batch)
        Iw1= self.warping(I2, F12)
        O_h = torch.where(O_s > 0.5, 1, 0)
        Io1 = Iw1*O_h
        Ic1 = self.completion_net(Io1)

        return O_s, O_h, Ic1, Iw1
