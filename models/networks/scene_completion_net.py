import torch.nn as nn
import torch.nn.functional as F
import torch

# code adapted from 
# https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/context_encoder/models.py
def weights_init_normal(model):
    classname = model.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(model.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal_(model.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(model.bias.data, 0.0)

class Generator(nn.Module):
    def __init__(self, channels=3):
        super().__init__()
        
        def downsample(in_channels, out_channels, normalize=True):
            layers = [nn.Conv2d(in_channels, out_channels, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_channels, 0.8))
            layers.append(nn.LeakyReLU(0.2))
            return layers
        
        def upsample(in_channels, out_channels, normalize=True):
            layers = [nn.ConvTranspose2d(in_channels, out_channels, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_channels, 0.8))
            layers.append(nn.ReLU())
            return layers
        
        self.model = nn.Sequential(
            *downsample(channels, 64, normalize=False),
            *downsample(64, 64),
            *downsample(64, 128),
            *downsample(128, 256),
            *downsample(256, 512),
            # channel-wise fully-connected layer
            nn.Conv2d(512, 4000, 1),
            *upsample(4000, 512),
            *upsample(512, 256),
            *upsample(256, 128),
            *upsample(128, 64),
            *upsample(64, 64),
            nn.Conv2d(64, channels, 3, 1, 1),
            nn.Tanh()
        )
        self.model.apply(weights_init_normal)
        
        
    def forward(self, x):
        return self.model(x)
    
    
