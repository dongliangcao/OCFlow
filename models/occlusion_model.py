"""Test occlusion map prediction network with ground truth occlusion map"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from torch.optim import Adam

from models.networks.simple_occlusion_net import SimpleOcclusionNet
from models.networks.occlusion_net_s import OcclusionNetS
from models.networks.occlusion_net_c import OcclusionNetC

from torchvision import transforms

import os
from math import ceil

class OcclusionModel(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.lr = hparams['learning_rate']
        model = self.hparams.get('model', 'simple')
        if model == 'simple':
            self.model = SimpleOcclusionNet()
        elif model == 'occnets':
            self.model = OcclusionNetS()
        elif model == 'occnetc':
            self.model = OcclusionNetC()
        else:
            raise ValueError(f'Unsupported model: {model}')
        
    def forward(self, x):
        out = self.model(x)
        
        return out
    
    @property
    def is_cuda(self):
        return next(self.parameters()).is_cuda
    
    def save_state_dict(self, path):
        torch.save(self.state_dict(), path)
        
    def general_step(self, batch, batch_idx, mode):
        if not isinstance(batch, (list, tuple)):
            raise ValueError('Not supported dataset')
        elif len(batch) == 2:
            imgs, occ = batch
        elif len(batch) == 3:
            imgs, _, occ = batch
        else:
            raise ValueError('Not supported dataset')
        occ_pred = self(imgs)
        ## focal loss 
        BCE_loss = F.binary_cross_entropy(occ_pred, occ, reduction='none')
        pt = torch.exp(-BCE_loss) # prevents nans when probability 0
#         alpha = 0.25
        gamma = 2
#         alpha_tensor = (1 - alpha) + occ * (2 * alpha - 1)
        focal_loss = (1 - pt)**gamma * BCE_loss
        return focal_loss.mean()
    
    def training_step(self, batch, batch_idx):
        loss = self.general_step(batch, batch_idx, 'train')
        self.log('train_loss', loss, prog_bar = True, on_step = True, on_epoch = True, logger = True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss = self.general_step(batch, batch_idx, 'val')
        self.log('val_loss', loss, prog_bar= True, logger = True)
        return loss
    
    def test_step(self, batch, batch_idx):
        loss = self.general_step(batch, batch_idx, 'test')
        self.log('test_loss', loss, prog_bar= True, logger= True)
        return loss
    
    def configure_optimizers(self):
        return Adam(self.model.parameters(), self.hparams['learning_rate'])
        