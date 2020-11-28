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
        imgs, occ = batch
        occ_pred = self.forward(imgs)
        
        loss = F.binary_cross_entropy(occ_pred, occ)
        
        return loss
    
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
        