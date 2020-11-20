"""Test flow/occlusion prediction network with ground truth flow/occlusion"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from torch.optim import Adam

from models.data.datasets import ImgFlowOccFromFolder
from models.networks.simple_flow_occ_net import SimpleFlowOccNet
from models.networks.flow_occ_net_s import FlowOccNetS
from models.networks.flow_occ_net_c import FlowOccNetC
from models.networks.cost_volume_flow_occ_net import FlowOccNetCV
from models.networks.flow_occ_net import FlowOccNet
from torchvision import transforms

import os
from math import ceil

class FlowOccModel(pl.LightningModule):
    def __init__(self, root, hparams):
        super().__init__()
        self.root = root
        self.hparams = hparams
        model = self.hparams.get('model', 'simple')
        if model == 'simple':
            self.model = SimpleFlowOccNet()
        elif model == 'pwoc':
            self.model = FlowOccNetCV()
        elif model == 'flowoccnets':
            self.model = FlowOccNetS()
        elif model == 'flowoccnetc':
            self.model = FlowOccNetC()
        elif model == 'flowoccnet':
            self.model = FlowOccNet()
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
        imgs, flow, occ = batch
        
        flow_pred, occ_pred = self.forward(imgs)
        
        flow_loss = F.l1_loss(flow_pred, flow)
        occ_loss = F.binary_cross_entropy(occ_pred, occ)
        return flow_loss + occ_loss
    
    def general_epoch_end(self, outputs, mode):
        avg_loss = torch.stack([output[mode + '_loss'] for output in outputs]).cpu().mean()
        
        return avg_loss
    
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
        