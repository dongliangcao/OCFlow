"""Test inpainting neural network with ground truth occlusion map and optical flow"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from torch.optim import Adam

from models.networks.image_inpainting_net import InpaintingNet

class InpaintingModel(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.model = InpaintingNet()
        
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
        
        img1, img2 = imgs[:, 0:3, :, :], imgs[:, 3:6, :, :]
        
        img_warped = self.warp(img2, flow)
        img_occluded = img_warped * (1 - occ) # 1: occluded, 0: non-occluded
        
        img_completed = self.forward(img_occluded)
        
#         square_error = F.mse_loss(img_completed, img1, reduction='none')
        loss = (torch.abs(img_completed-img1) * occ).sum() / (3*occ.sum() + 1e-16)
        
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
        