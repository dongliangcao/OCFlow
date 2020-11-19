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
        imgs, flow, occ = imgs.to(self.device), flow.to(self.device), occ.to(self.device)
        
        flow_pred, occ_pred = self.forward(imgs)
        
        flow_loss = F.l1_loss(flow_pred, flow)
        occ_loss = F.binary_cross_entropy(occ_pred, occ)
        return flow_loss + occ_loss
    
    def general_epoch_end(self, outputs, mode):
        avg_loss = torch.stack([output[mode + '_loss'] for output in outputs]).cpu().mean()
        
        return avg_loss
    
    def training_step(self, batch, batch_idx):
        loss = self.general_step(batch, batch_idx, 'train')
        tensorboard_logs = {'train_loss': loss}
        
        return {'loss': loss, 'train_loss': loss, 'log': tensorboard_logs}
    
    def training_epoch_end(self, outputs):
        avg_loss = self.general_epoch_end(outputs, 'train')
        
        return {'train_avg_loss': avg_loss}
    
    def validation_step(self, batch, batch_idx):
        loss = self.general_step(batch, batch_idx, 'val')
        
        return {'val_loss': loss}
    
    def validation_epoch_end(self, outputs):
        avg_loss = self.general_epoch_end(outputs, 'val')
        
        print(f'Val-Loss: {avg_loss:.4f}')
        
        tensorboard_logs = {'val_avg_loss': avg_loss}
        
        return {'val_avg_loss': avg_loss, 'log': tensorboard_logs}
    
    def test_step(self, batch, batch_idx):
        loss = self.general_step(batch, batch_idx, 'test')
        
        return {'test_loss': loss}
    
    def test_epoch_end(self, outputs):
        avg_loss = self.general_epoch_end(outputs, 'test')
     
        return {'test_avg_loss': avg_loss}
    
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
        
    @pl.data_loader
    def train_dataloader(self):
        batch_size = self.hparams['batch_size']
        return DataLoader(self.datasets['train'], shuffle=True, batch_size=batch_size)
    
    @pl.data_loader
    def val_dataloader(self):
        batch_size = self.hparams['batch_size']
        return DataLoader(self.datasets['val'], shuffle=False, batch_size=batch_size)
    
    @pl.data_loader
    def test_dataloader(self):
        batch_size = self.hparams['batch_size']
        return DataLoader(self.datasets['test'], shuffle=False, batch_size=batch_size)
    
    def configure_optimizers(self):
        return Adam(self.model.parameters(), self.hparams['learning_rate'])
        