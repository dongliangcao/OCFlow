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

class Downsample(nn.Module):
    def __init__(self, in_channels, out_channels, normalize=True):
        super(Downsample,self).__init__()
        layers = [nn.Conv2d(in_channels, out_channels, 4, stride=2, padding=1)]
        if normalize:
            layers.append(nn.BatchNorm2d(out_channels, 0.8))
        layers.append(nn.LeakyReLU(0.2))
        self.model = nn.Sequential(*layers)
    def forward(self, x):
        return self.model(x)
    
class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels, normalize=True):
        super(Upsample,self).__init__()
        layers = [nn.ConvTranspose2d(in_channels, out_channels, 4, stride=2, padding=1)]
        if normalize:
            layers.append(nn.BatchNorm2d(out_channels, 0.8))
        layers.append(nn.ReLU())
        self.model = nn.Sequential(*layers)
        
    def forward(self, x, output_size=None):
        for layer in self.model:
            if isinstance(layer, nn.ConvTranspose2d):
                x = layer(x, output_size=output_size)
            else:
                x = layer(x)
        return x
        
class SceneCompletionNet(nn.Module):
    def __init__(self, channels=3):
        super(SceneCompletionNet, self).__init__()
        self.model = nn.Sequential(
            Downsample(channels, 64, normalize=False),
            Downsample(64, 64),
            Downsample(64, 128),
            Downsample(128, 256),
            Downsample(256, 512),
            # channel-wise fully-connected layer
            nn.Conv2d(512, 4000, 1),
            Upsample(4000, 512),
            Upsample(512, 256),
            Upsample(256, 128),
            Upsample(128, 64),
            Upsample(64, 64),
            nn.Conv2d(64, channels, 3, 1, 1),
            # restrict the output between [0, 1]
            nn.Sigmoid()
        )
        self.model.apply(weights_init_normal)
        
        
    def forward(self, x):
        sizes = []
        for layer in self.model:
            if isinstance(layer, Downsample):
                sizes.append(x.size())
                x = layer(x)
            elif isinstance(layer, Upsample):
                size = sizes.pop()
                x = layer(x, output_size=size)
            else:
                x = layer(x)
        return x
    
    
