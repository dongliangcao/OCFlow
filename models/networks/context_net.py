import torch
import torch.nn as nn
import torch.nn.functional as F

class ContextNetwork(nn.Module): 
    """
    Network for refining the flow estimation
    """
    def __init__(self, input_channels):
        super(ContextNetwork, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 128, 3, 1, 1, 1)
        self.conv2 = nn.Conv2d(128, 128, 3, 1, 1, 2)
        self.conv3 = nn.Conv2d(128, 128, 3, 1, 1, 4)
        self.conv4 = nn.Conv2d(128, 96, 3, 1, 1, 8)
        self.conv5 = nn.Conv2d(96, 64, 3, 1, 1, 16)
        self.conv6 = nn.Conv2d(64, 32, 3, 1, 1, 1)
        self.conv7 = nn.Conv2d(32, 4, 3, 1, 1, 1)
    def forward(self, x): 
        c1 = F.leaky_relu(self.conv1(x), negative_slope= 0.1)
        c2 = F.leaky_relu(self.conv2(c1), negative_slope= 0.1)
        c3 = F.leaky_relu(self.conv3(c2), negative_slope= 0.1)
        c4 = F.leaky_relu(self.conv4(c3), negative_slope= 0.1)
        c5 = F.leaky_relu(self.conv5(c4), negative_slope= 0.1)
        c6 = F.leaky_relu(self.conv6(c5), negative_slope= 0.1)
        c7 = self.conv7(c6)
        return c7


