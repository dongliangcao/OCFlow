"""Test optical flow prediction network with ground truth optical flow"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from models.networks.simple_flow_net import SimpleFlowNet
from models.networks.simple_occlusion_net import SimpleOcclusionNet
from models.networks.image_inpainting_net import InpaintingNet

def charbonnier_loss(loss, alpha=0.001, reduction=True):
    """
    Args:
        loss: torch.Tensor
    Return:
        cb_loss: torch.Tensor, charbonnier_loss
    """
    cb_loss = torch.sqrt(loss**2 + alpha**2)
    if reduction:
        cb_loss = cb_loss.mean()
    return cb_loss

def second_order_photometric_error(img_pred, img, reduction=True):
    img_pred_dx, img_pred_dy = gradient(img_pred)
    img_dx, img_dy = gradient(img)
    img_pred_dx_norm = torch.norm(img_pred_dx, p=2, dim=1)
    img_pred_dy_norm = torch.norm(img_pred_dy, p=2, dim=1)
    img_dx_norm = torch.norm(img_dx, p=2, dim=1)
    img_dy_norm = torch.norm(img_dy, p=2, dim=1)
    
    loss = charbonnier_loss(img_pred_dx_norm - img_dx_norm, reduction=reduction) + charbonnier_loss(img_pred_dy_norm - img_dy_norm, reduction=reduction)
    return loss

def gradient(img):
    """
    Args:
        img: torch.Tensor, dim: [B, C, H, W]
    Return:
        dx: forward gradient in direction x, dim: [B, C, H, W]
        dy: forward gradient in direction y, dim: [B, C, H, W]
    """
    assert img.dim() == 4
    _, _, ny, nx = img.shape
    dx = img[:, :, :, [*range(1, nx), nx-1]] - img
    dy = img[:, :, [*range(1, ny), ny - 1], :] - img
    
    return dx, dy

def edge_aware_smoothness_loss(img, flow, alpha=10.0, reduction=True):
    """
    Args:
        img: torch.Tensor, dim: [B, C, H, W]
        flow: torch.Tensor, dim: [B, 2, H, W]
        alpha: float, control the edge awareness
    Return:
        loss: torch.Tensor
    """
    assert img.dim() == 4 
    assert flow.dim() == 4 and flow.shape[1] == 2

    img_dx, img_dy = gradient(img)
    flow_dx, flow_dy = gradient(flow)
    
    img_dx_norm = torch.norm(img_dx, p=2, dim=1)
    img_dy_norm = torch.norm(img_dy, p=2, dim=1)
    flow_dx_norm = torch.norm(flow_dx, p=2, dim=1)
    flow_dy_norm = torch.norm(flow_dy, p=2, dim=1)
    loss = (flow_dx_norm) * torch.exp(-alpha * img_dx_norm) + (flow_dy_norm) * torch.exp(-alpha * img_dy_norm)
    
    return charbonnier_loss(loss, reduction=reduction)

class FlowStageModel(pl.LightningModule):
    """
    Training with one stages: optical flow prediction
    """
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.lr = hparams['learning_rate']
        self.smoothness_weight = hparams.get('smoothness_weight', 0.0)
        self.second_order_weight = hparams.get('second_order_weight', 0.0)
        self.with_occ = hparams.get('with_occ', False)
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
        photometric_error = charbonnier_loss(img_warped - img1)
        smoothness_term = edge_aware_smoothness_loss(img1, flow_pred)
        second_order_error = second_order_photometric_error(img_warped, img1)
        return photometric_error, smoothness_term, second_order_error
    
    def general_step_occ(self, batch, batch_idx, mode):
        imgs, flow, occ = batch
        img1, img2 = imgs[:, 0:3, :, :], imgs[:, 3:6, :, :]
        # flow prediction
        flow_pred = self(imgs)
        img_warped = self.warp(img2, flow_pred)
        
        # calculate photometric error
        photometric_error = (charbonnier_loss(img_warped - img1, reduction=False) * (1 - occ)).sum() / (3*(1 - occ).sum() + 1e-16)
        second_order_error = (second_order_photometric_error(img_warped, img1, reduction=False) * (1 - occ)).sum() / (3*(1 - occ).sum() + 1e-16)
        smoothness_term = edge_aware_smoothness_loss(img1, flow_pred)
        
        return photometric_error, smoothness_term, second_order_error
    
    def training_step(self, batch, batch_idx):
        if not self.with_occ:
            photometric_error, smoothness_term, second_order_error = self.general_step(batch, batch_idx, 'train')
        else:
            photometric_error, smoothness_term, second_order_error = self.general_step_occ(batch, batch_idx, 'train')
        loss = photometric_error + self.smoothness_weight * smoothness_term + self.second_order_weight * second_order_error
        
        self.log('train_photometric', photometric_error, logger = True)
        self.log('train_smoothness', smoothness_term, logger = True)
        self.log('train_second_order', second_order_error, logger = True)
        
        self.log('train_loss', loss, prog_bar = True, on_step = True, on_epoch = True, logger = True)
        return loss
    
    
    def validation_step(self, batch, batch_idx):
        if not self.with_occ:
            photometric_error, smoothness_term, second_order_error = self.general_step(batch, batch_idx, 'val')
        else:
            photometric_error, smoothness_term, second_order_error = self.general_step_occ(batch, batch_idx, 'val')
        loss = photometric_error + self.smoothness_weight * smoothness_term + self.second_order_weight * second_order_error
        
        self.log('val_photometric', photometric_error, logger = True)
        self.log('val_smoothness', smoothness_term, logger = True)
        self.log('val_second_order', second_order_error, logger = True)
        
        self.log('val_loss', loss, prog_bar= True, logger = True)
        return loss
    
    def test_step(self, batch, batch_idx):
        if not self.with_occ:
            photometric_error, smoothness_term, second_order_error = self.general_step(batch, batch_idx, 'test')
        else:
            photometric_error, smoothness_term, second_order_error = self.general_step_occ(batch, batch_idx, 'test')
        loss = photometric_error + self.smoothness_weight * smoothness_term + self.second_order_weight * second_order_error
        
        self.log('test_photometric', photometric_error, logger = True)
        self.log('test_smoothness', smoothness_term, logger = True)
        self.log('test_second_order', second_order_error, logger = True)
        
        self.log('test_loss', loss, prog_bar= True, logger = True)
        return loss
    
    def configure_optimizers(self):
        return Adam(self.parameters(), self.lr)
    
class InpaintingStageModel(pl.LightningModule):
    """
    Training with two stages:
    First stage: optical flow and occlusion map prediction
    Second stage: inpainting network predicts pixel value for the occluded regions 
    """
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.lr = hparams['learning_rate']
        self.second_order_weight = hparams.get('second_order_weight', 0.0)
        self.model = InpaintingNet()
    
    def forward(self, img):
        return self.model(img)
    
    @property
    def is_cuda(self):
        return next(self.parameters()).is_cuda
    
    
    def save_state_dict(self, path):
        torch.save(self.state_dict(), path)
        
    def general_step(self, batch, batch_idx, mode):
        imgs, complete_imgs, occlusion_map = batch
        # inpainting
        pred_imgs = self(complete_imgs * (1 - occlusion_map))
        reconst_error = (charbonnier_loss(pred_imgs - complete_imgs, reduction=False) * occlusion_map).sum() / (3*occlusion_map.sum() + 1e-16)
        second_order_error = (second_order_photometric_error(pred_imgs, complete_imgs, reduction=False) * occlusion_map).sum() / (3*occlusion_map.sum() + 1e-16)
        return reconst_error, second_order_error
    
    
    def training_step(self, batch, batch_idx):
        reconst_error, second_order_error = self.general_step(batch, batch_idx, 'train')
        loss = reconst_error + self.second_order_weight * second_order_error
        
        self.log('train_reconst', reconst_error, logger = True)
        self.log('train_second_order', second_order_error, logger = True)
        self.log('train_loss', loss, prog_bar = True, on_step = True, on_epoch = True, logger = True)
        return loss
    
    
    def validation_step(self, batch, batch_idx):
        reconst_error, second_order_error = self.general_step(batch, batch_idx, 'val')
        loss = reconst_error + self.second_order_weight * second_order_error
        
        self.log('val_reconst', reconst_error, logger = True)
        self.log('val_second_order', second_order_error, logger = True)
        self.log('val_loss', loss, prog_bar= True, logger = True)
        return loss
    
    def test_step(self, batch, batch_idx):
        reconst_error, second_order_error = self.general_step(batch, batch_idx, 'test')
        loss = reconst_error + self.second_order_weight * second_order_error
        
        self.log('test_reconst', reconst_error, logger = True)
        self.log('test_second_order', second_order_error, logger = True)
        self.log('test_loss', loss, prog_bar= True, logger = True)
        return loss
    
    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), self.lr)
        return optimizer
#         scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=20)
#         return {
#             'optimizer': optimizer,
#             'lr_scheduler': scheduler,
#             'monitor': 'val_reconst'
#         }

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
        self.smoothness_weight = hparams.get('smoothness_weight', 0.0)
        self.reconst_weight = hparams.get('reconst_weight', 2.0)
        
        flow_root = hparams.get('flow_root', None)
        inpainting_root = hparams.get('inpainting_root', None)
        self.flow_pred = SimpleFlowNet()
        self.occ_pred = SimpleOcclusionNet()
        self.inpainting = InpaintingNet()
        if flow_root:
            self.flow_pred.load_state_dict(torch.load(flow_root))
        if inpainting_root:
            self.inpainting.load_state_dict(torch.load(inpainting_root))
        
        # we will freeze the optical flow network and inpainting network
        for param in self.flow_pred.parameters():
            param.requires_grad = False
        for param in self.inpainting.parameters():
            param.requires_grad = False
    
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

        # get completed image
        img_completed = self.inpainting(img_occluded)
        
        # calculate the reconstruction error
        photometric_error = charbonnier_loss((img_warped - img1) * (1 - occ_pred), reduction=False).sum() / (3*(1 - occ_pred).sum() + 1e-16)
        reconst_error = charbonnier_loss(torch.abs(img1 - img_completed) * occ_pred, reduction=False).sum() / (3*occ_pred.sum() + 1e-16)
        smoothness_term = (edge_aware_smoothness_loss(img1, flow_pred, reductiom=False) * (1 - occ_pred)).sum() / (3*(1 - occ_pred).sum() + 1e-16)
        return photometric_error, reconst_error, smoothness_term
    
    
    def training_step(self, batch, batch_idx):
        photometric_error, reconst_error, smoothness_term = self.general_step(batch, batch_idx, 'train')
        loss = photometric_error + self.reconst_weight * reconst_error + self.smoothness_weight * smoothness_term
        self.log('train_photometric', photometric_error, logger = True)
        self.log('train_reconst', reconst_error, logger = True)
        self.log('train_flow_smooth', smoothness_term, logger=True)
        self.log('train_loss', loss, prog_bar = True, on_step = True, on_epoch = True, logger = True)
        return loss
    
    
    def validation_step(self, batch, batch_idx):
        photometric_error, reconst_error, smoothness_term = self.general_step(batch, batch_idx, 'val')
        loss = photometric_error + self.reconst_weight * reconst_error + self.smoothness_weight * smoothness_term
        self.log('val_photometric', photometric_error, logger = True)
        self.log('val_reconst', reconst_error, logger = True)
        self.log('val_flow_smooth', smoothness_term, logger = True)
        self.log('val_loss', loss, prog_bar= True, logger = True)
        return loss
    
    def test_step(self, batch, batch_idx):
        photometric_error, reconst_error, smoothness_term = self.general_step(batch, batch_idx, 'test')
        loss = photometric_error + self.reconst_weight * reconst_error + self.smoothness_weight * smoothness_term
        self.log('test_photometric', photometric_error, logger = True)
        self.log('test_reconst', reconst_error, logger = True)
        self.log('test_flow_smooth', smoothness_term, logger = True)
        self.log('test_loss', loss, prog_bar= True, logger = True)
        return loss
    
    def configure_optimizers(self):
        return Adam(self.parameters(), self.lr)
        