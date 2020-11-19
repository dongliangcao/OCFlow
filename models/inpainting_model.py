"""Test inpainting neural network with ground truth occlusion map and optical flow"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from torch.optim import Adam

from models.data.datasets import ImgFlowOccFromFolder
from models.networks.image_inpainting_net import SceneCompletionNet
from torchvision import transforms

import os
from math import ceil

class InpaintingModel(pl.LightningModule):
    def __init__(self, root, hparams):
        super().__init__()
        self.root = root
        self.hparams = hparams
        self.model = SceneCompletionNet()
        
    def forward(self, x):
        out = self.model(x)
        
        return out
    
    def warp(self, img, flow):
        """
        warp an image/tensor (im2) back to im1, according to the optical flow

        x: [B, C, H, W] (im2)
        flo: [B, 2, H, W] flow

        """
        B, C, H, W = img.size()
        # create mesh grid
        xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
        yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
        xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
        yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
        grid = torch.cat((xx, yy), 1).float()
        # cast into cuda
        if img.is_cuda:
            grid = grid.cuda()
        # require gradient
        grid.requires_grad = True
        vgrid = grid + flow
        # scale grid to [-1, 1] to support grid_sample function in pytorch
        # https://pytorch.org/docs/stable/nn.functional.html#grid-sample
        vgrid[:,0,:,:] = 2.0 * vgrid[:,0,:,:].clone() / max(W-1, 1) - 1.0
        vgrid[:,1,:,:] = 2.0 * vgrid[:,1,:,:].clone() / max(H-1, 1) - 1.0
        # permute vgrid to size [B, H, W, 2] to support grid_sample function
        vgrid = vgrid.permute(0, 2, 3, 1)
        
        output = F.grid_sample(img, vgrid, align_corners=False)
        
        return output
    
    @property
    def is_cuda(self):
        return next(self.parameters()).is_cuda
    
    
    def save_state_dict(self, path):
        torch.save(self.state_dict(), path)
        
    def general_step(self, batch, batch_idx, mode):
        imgs, flow, occ = batch
        imgs, flow, occ = imgs.to(self.device), flow.to(self.device), occ.to(self.device)
        
        img1, img2 = imgs[:, 0:3, :, :], imgs[:, 3:6, :, :]
        
        img_warped = self.warp(img2, flow)
        img_occluded = img_warped * (1 - occ) # 1: occluded, 0: non-occluded
        
        img_completed = self.forward(img_occluded)
        
        loss = (torch.abs(img1 - img_completed) * occ).sum() / (occ.sum() + 1e-12)
        
        return loss
    
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
        