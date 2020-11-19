"""Test optical flow prediction network with ground truth optical flow"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from torch.optim import Adam

from models.data.datasets import ImgFlowOccFromFolder
from models.networks.simple_flow_net import SimpleFlowNet
from models.networks.flow_net_s import FlowNetS
from models.networks.flow_net_c import FlowNetC
from models.networks.cost_volume_flow_net import FlowNetCV
from models.networks.flow_net import FlowNet
from torchvision import transforms

import os
from math import ceil

class FlowModel(pl.LightningModule):
    def __init__(self, root, hparams):
        super().__init__()
        self.root = root
        self.hparams = hparams
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
        imgs, flow, _ = batch
        #imgs, flow = imgs.to(self.device), flow.to(self.device)
        
        flow_pred = self.model(imgs)
        
        loss = F.l1_loss(flow_pred, flow)
        
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
    
    
    def prepare_data(self):
        self.datasets = dict()
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        dataset = ImgFlowOccFromFolder(root=self.root, transform=transform, resize=transforms.Resize(self.hparams['image_size']), stack_imgs=False)
        train_dset, val_dset, test_dset = random_split(dataset, [ceil(len(dataset)*0.8), ceil(len(dataset)*0.1), len(dataset) - ceil(len(dataset)*0.8) - ceil(len(dataset)*0.1)])
        self.datasets['train'] = train_dset
        self.datasets['val'] = val_dset
        self.datasets['test'] = test_dset
        
    def train_dataloader(self):
        batch_size = self.hparams['batch_size']
        return DataLoader(self.datasets['train'], shuffle=True, batch_size=batch_size)
    
    def val_dataloader(self):
        batch_size = self.hparams['batch_size']
        return DataLoader(self.datasets['val'], shuffle=False, batch_size=batch_size)
    
    def test_dataloader(self):
        batch_size = self.hparams['batch_size']
        return DataLoader(self.datasets['test'], shuffle=False, batch_size=batch_size)
    
    def configure_optimizers(self):
        return Adam(self.model.parameters(), self.hparams['learning_rate'])
        