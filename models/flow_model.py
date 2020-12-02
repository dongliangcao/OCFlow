"""Test optical flow prediction network with ground truth optical flow"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from torch.optim import Adam

from models.networks.simple_flow_net import SimpleFlowNet
from models.networks.flow_net_s import FlowNetS
from models.networks.flow_net_c import FlowNetC
from models.networks.cost_volume_flow_net import FlowNetCV
from models.networks.flow_net import FlowNet
from models.networks.efficient_flow_net import EFlowNet, EFlowNet2

class FlowModel(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.lr = hparams['learning_rate']
        model = self.hparams.get('model', 'simple')
        if model == 'simple':
            self.model = SimpleFlowNet()
        elif model == 'pwc':
            self.model = FlowNetCV()
        elif model == 'flownets':
            self.model = FlowNetS()
        elif model == 'flownetc':
            self.model = FlowNetC()
        elif model == 'flownet':
            self.model = FlowNet()
        elif model == 'eflownet':
            self.model = EFlowNet()
        elif model == 'eflownet2':
            self.model = EFlowNet2()
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
            imgs, flow = batch
        elif len(batch) == 3:
            imgs, flow, _ = batch
        else:
            raise ValueError('Not supported dataset')
        flow_pred = self.model(imgs)
        
        loss = F.mse_loss(flow_pred, flow)
        
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
        return Adam(self.parameters(), self.lr)
        