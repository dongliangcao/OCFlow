"""Test optical flow prediction network with ground truth optical flow"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from torch.optim import Adam

from models.networks.simple_flow_net import SimpleFlowNet
from models.networks.simple_occlusion_net import SimpleOcclusionNet
from models.networks.image_inpainting_net import InpaintingNet

class OneStageModel(pl.LightningModule):
    """
    Training with one stages: optical flow prediction
    """
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.lr = hparams['learning_rate']
        self.flow_pred = SimpleFlowNet()
        
    def forward(self, x):
        return self.flow_pred(x)
    
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
        if not isinstance(batch, (list, tuple)):
            imgs = batch
        elif len(batch) == 2:
            imgs, _ = batch
        elif len(batch) == 3:
            imgs, _, _ = batch
        else:
            raise ValueError('Not supported dataset')
        img1, img2 = imgs[:, 0:3, :, :], imgs[:, 3:6, :, :]
        # flow prediction
        flow_pred = self(imgs)
        img_warped = self.warp(img2, flow_pred)
        
        # calculate photometric error
        loss = F.l1_loss(img_warped, img1)
        
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
        
        self.log('test_loss', loss, prog_bar= True, logger = True)
        return loss
    
    def configure_optimizers(self):
        return Adam(self.parameters(), self.lr)

class TwoStageModel(pl.LightningModule):
    """
    Training with two stages:
    First stage: optical flow and occlusion map prediction
    Second stage: inpainting network predicts pixel value for the occluded regions 
    """
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.lr = hparams['learning_rate']
        self.flow_pred = SimpleFlowNet()
        self.occ_pred = SimpleOcclusionNet()
        self.inpainting = InpaintingNet()
    
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
        if not isinstance(batch, (list, tuple)):
            imgs = batch
        elif len(batch) == 2:
            imgs, _ = batch
        elif len(batch) == 3:
            imgs, _, _ = batch
        else:
            raise ValueError('Not supported dataset')
        img1, img2 = imgs[:, 0:3, :, :], imgs[:, 3:6, :, :]
        # flow prediction
        flow_pred = self.flow_pred(imgs)
        # occlusion prediction
        occ_pred = self.occ_pred(imgs)
        # warp image
        img_warped = self.warp(img2, flow_pred)
        # get occluded image
        img_occluded = img_warped * (1 - occ_pred) # 1: occluded 0: non-occluded
        # calculate photometric error
        photometric_error = (torch.abs(img1 - img_warped) * (1 - occ_pred)).sum() / (3*(1 - occ_pred).sum() + 1e-16)
        # get completed image
        img_completed = self.inpainting(img_occluded)
        # calculate the reconstruction error
        reconst_error = (torch.abs(img1 - img_completed) * occ_pred).sum() / (3*occ_pred.sum() + 1e-16)
        
        return photometric_error, reconst_error
    
    
    def training_step(self, batch, batch_idx):
        photometric_error, reconst_error = self.general_step(batch, batch_idx, 'train')
        loss = 3.0 * photometric_error + reconst_error
        self.log('train_photometric_error', photometric_error, logger = True)
        self.log('train_reconst_error', reconst_error, logger = True)
        self.log('train_loss', loss, prog_bar = True, on_step = True, on_epoch = True, logger = True)
        return loss
    
    
    def validation_step(self, batch, batch_idx):
        photometric_error, reconst_error = self.general_step(batch, batch_idx, 'val')
        loss = 3.0 * photometric_error + reconst_error
        self.log('val_photometric_error', photometric_error, logger = True)
        self.log('val_reconst_error', reconst_error, logger = True)
        self.log('val_loss', loss, prog_bar= True, logger = True)
        return loss
    
    def test_step(self, batch, batch_idx):
        photometric_error, reconst_error = self.general_step(batch, batch_idx, 'test')
        loss = 3.0 * photometric_error + reconst_error
        self.log('test_photometric_error', photometric_error, logger = True)
        self.log('test_reconst_error', reconst_error, logger = True)
        self.log('test_loss', loss, prog_bar= True, logger = True)
        return loss
    
    def configure_optimizers(self):
        return Adam(self.parameters(), self.lr)
        