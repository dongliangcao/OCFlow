"""Test optical flow prediction network with ground truth optical flow"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

import torchvision

from models.networks.simple_flow_net import SimpleFlowNet
from models.networks.flow_net_s import FlowNetS
from models.networks.flow_net_c import FlowNetC
from models.networks.cost_volume_flow_net import FlowNetCV
from models.networks.flow_net import FlowNet
from models.networks.pwc_net import PWCNet
from models.networks.efficient_flow_net import EFlowNet, EFlowNet2
from models.flow_model import FlowModel
from models.networks.simple_occlusion_net import SimpleOcclusionNet
from models.networks.image_inpainting_net import InpaintingNet
from models.networks.gated_conv_inpainting_net import InpaintSANet, InpaintSANetOrg, InpaintSADiscriminator, InpaintSADiscriminatorOrg, SNDisLoss, SNGenLoss, ReconLoss
from PIL import Image
import numpy as np
import os

def robust_l1(x, alpha=0.001):
    """
    Args:
        x: torch.Tensor
    Return:
        loss: torch.Tensor, charbonnier_loss
    """
    loss = (x**2 + alpha**2)**0.5 
    return loss

def photometric_error(img_pred, img, occ=None):
    """
    occ: 1 occluded, 0 non-occluded
    """
    error = robust_l1(img_pred - img)
    if occ is not None:
        loss = torch.sum(error * (1-occ))/(torch.sum(1-occ)*3 + 1e-16)
    else:
        loss = torch.mean(error)
    return loss

def gradient(img, stride=1):
    """
    Args:
        img: torch.Tensor, dim: [B, C, H, W]
    Return:
        dx: forward gradient in direction x, dim: [B, C, H, W]
        dy: forward gradient in direction y, dim: [B, C, H, W]
    """
    assert img.dim() == 4
    _, _, ny, nx = img.shape
    dx = img[:, :, :, stride:] - img[:, :, :, :-stride]
    dy = img[:, :, stride:, :] - img[:, :, :-stride, :]
    
    return dx, dy

def edge_aware_smoothness_loss(img, flow, alpha=100.0):
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
    loss_dx = (flow_dx_norm) * torch.exp(-alpha * img_dx_norm)
    loss_dy = (flow_dy_norm) * torch.exp(-alpha * img_dy_norm)
    loss = 0.5 * (robust_l1(loss_dx) + robust_l1(loss_dy))
    loss = torch.mean(loss)
    return loss

def first_order_smoothness_loss(img, flow, alpha=100.0):
    img_gx, img_gy = gradient(img)
    weights_x = torch.exp(-torch.mean((alpha * img_gx)**2, dim=1, keepdim=True))
    weights_y = torch.exp(-torch.mean((alpha * img_gy)**2, dim=1, keepdim=True))

    flow_gx, flow_gy = gradient(flow)
    loss = 0.5 * (torch.mean(weights_x * robust_l1(flow_gx)) + torch.mean(weights_y * robust_l1(flow_gy)))
    
    return loss

def second_order_smoothness_loss(img, flow, alpha=100.0):
    img_gx, img_gy = gradient(img, stride=2)
    weights_xx = torch.exp(-torch.mean((alpha * img_gx)**2, dim=1, keepdim=True))
    weights_yy = torch.exp(-torch.mean((alpha * img_gy)**2, dim=1, keepdim=True))

    flow_gx, flow_gy = gradient(flow)
    flow_gxx, _ = gradient(flow_gx)
    _, flow_gyy = gradient(flow_gy)

    loss = 0.5 * (torch.mean(weights_xx * robust_l1(flow_gxx)) + torch.mean(weights_yy * robust_l1(flow_gyy)))
    loss = torch.mean(loss)
    return loss

def img2photo(imgs, rgb = True):
    if rgb: 
        return ((imgs+1)*127.5).transpose(1,2).transpose(2,3).detach().cpu().numpy()
    else: 
        return imgs.transpose(1,2).transpose(2,3).detach().cpu().numpy()


class VGGPerceptualLoss(nn.Module):
    def __init__(self, resize=False, w = [1.0,1.0,1.0,1.0]):
        super(VGGPerceptualLoss, self).__init__()
        self.w = torch.nn.Parameter(torch.tensor(w))
        blocks = []
        blocks.append(torchvision.models.vgg16(pretrained=True).features[:4].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[4:9].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[9:16].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[16:23].eval())
        # freeze vgg net
        for bl in blocks:
            for p in bl:
                p.requires_grad = False
        self.blocks = nn.ModuleList(blocks)
        self.transform = F.interpolate
        self.resize = resize

    def forward(self, input, target):
        if self.resize:
            input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
            target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)
        losses = []
        x = input
        y = target
        for block in self.blocks:
            x = block(x)
            y = block(y)
            losses.append(F.l1_loss(x, y))
        loss = torch.stack(losses)
        return (loss*self.w).sum()


class FlowStageModel(pl.LightningModule):
    """
    Training with one stages: optical flow prediction
    """
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.lr = hparams['learning_rate']
        self.photo_weight = hparams.get('photo_weight', 1.0)
        self.smooth1_weight = hparams.get('smooth1_weight', 0.0)
        self.smooth2_weight = hparams.get('smooth2_weight', 1.0)
        self.with_occ = hparams.get('with_occ', False)
        self.log_every_n_steps = hparams.get('log_every_n_steps', 20)
        self.occ_aware = hparams.get('occ_aware', False)
        self.displacement = hparams.get('displacement', 4)
        model = self.hparams.get('model', 'simple')
        self.model = model
        if model == 'simple':
            self.flow_pred = SimpleFlowNet()
        elif model == 'pwc':
            self.flow_pred = FlowNetCV(displacement=self.displacement)
        elif model == 'flownets':
            self.flow_pred = FlowNetS()
        elif model == 'flownetc':
            self.flow_pred = FlowNetC()
        elif model == 'flownet':
            self.flow_pred = FlowNet()
        elif model == 'eflownet':
            self.flow_pred = EFlowNet()
        elif model == 'eflownet2':
            self.flow_pred = EFlowNet2()
        else:
            raise ValueError(f'Unsupported model: {model}')
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
        
        output = F.grid_sample(img, vgrid, align_corners=True)
        
        return output
    
    def flow_to_warp(self, flow):
        """
        Compute the warp from the flow field
        Args:
            flow: optical flow shape [B, H, W, 2]
        Returns:
            warp: the endpoints of the estimated flow. shape [B, H, W, 2]
        """
        B, H, W, _ = flow.size()
        xx = torch.arange(0, W).view(1,-1).repeat(H,1)
        yy = torch.arange(0, H).view(-1,1).repeat(1,W)
        xx = xx.view(1,1,H,W).repeat(B,1,1,1)
        yy = yy.view(1,1,H,W).repeat(B,1,1,1)
        grid = torch.cat((xx,yy),1).float()
        if flow.is_cuda:
            grid = grid.cuda()
        grid = grid.permute(0, 2, 3, 1)
        warp = grid + flow
        return warp
    
    def compute_range_map(self, flow):
        # permute flow from [B, 2, H, W] to shape [B, H, W, 2]
        flow = flow.permute(0, 2, 3, 1)
        batch_size, input_height, input_width, _ = flow.size()

        coords = self.flow_to_warp(flow)

        # split coordinates into an integer part and a float offset for interpolation.
        coords_floor = torch.floor(coords)
        coords_offset = coords - coords_floor
        coords_floor = coords_floor.to(torch.int32)

        # Define a batch offset for flattened indexes into all pixels
        batch_range = torch.reshape(torch.arange(batch_size), [batch_size, 1, 1])
        if flow.is_cuda:
            batch_range = batch_range.cuda()
        idx_batch_offset = batch_range.repeat(1, input_height, input_width) * input_height * input_width

        # Flatten everything
        coords_floor_flattened = coords_floor.reshape(-1, 2)
        coords_offset_flattened = coords_offset.reshape(-1, 2)
        idx_batch_offset_flattened = idx_batch_offset.reshape(-1)

        # Initialize results
        idxs_list = []
        weights_list = []

        # Loop over different di and dj to the four neighboring pixels
        for di in range(2):
            for dj in range(2):
                # Compute the neighboring pixel coordinates
                idxs_i = coords_floor_flattened[:, 0] + di
                idxs_j = coords_floor_flattened[:, 1] + dj
                # Compute the flat index into all pixels
                idxs = idx_batch_offset_flattened + idxs_j * input_width + idxs_i

                # Only count valid pixels
                mask = torch.nonzero(torch.logical_and(
                    torch.logical_and(idxs_i >= 0, idxs_i < input_width),
                    torch.logical_and(idxs_j >= 0, idxs_j < input_height)
                ), as_tuple=True)
                valid_idxs = idxs[mask]
                valid_offsets = coords_offset_flattened[mask]

                # Compute weights according to bilinear interpolation
                weights_i = (1. - di) - (-1)**di * valid_offsets[:, 0]
                weights_j = (1. - dj) - (-1)**dj * valid_offsets[:, 1]
                weights = weights_i * weights_j

                # Append indices and weights
                idxs_list.append(valid_idxs)
                weights_list.append(weights)

        # Concatenate everything
        idxs = torch.cat(idxs_list, dim=0)
        weights = torch.cat(weights_list, dim=0)
        counts = torch.zeros(batch_size * input_width * input_height, dtype=weights.dtype)
        if flow.is_cuda:
            counts = counts.cuda()
        counts.scatter_add_(0, idxs, weights)
        range_map = counts.reshape(batch_size, 1, input_height, input_width)

        return range_map
    
    @property
    def is_cuda(self):
        return next(self.parameters()).is_cuda
    
    
    def save_state_dict(self, path):
        torch.save(self.state_dict(), path)
        
    def general_step(self, batch, batch_idx, mode):
        if len(batch) == 2:
            imgs, flow = batch
        elif len(batch) == 3:
            imgs, flow, occ = batch
        else:
            raise ValueError('Not supported dataset')
        img1, img2 = imgs[:, 0:3, :, :], imgs[:, 3:6, :, :]
        # flow prediction
        if self.model == 'pwc':
            flow_pred, flow_l2 = self(imgs)
        else:
            flow_pred = self(imgs)
        img_warped = self.warp(img2, flow_pred)
        
        # calculate photometric error
        photo = photometric_error(img_warped, img1)
        if self.model == 'pwc':
            img1_l2 = F.interpolate(img1, scale_factor=0.25, mode='bilinear', align_corners=True)
            smooth1 = first_order_smoothness_loss(img1_l2, flow_l2)
            smooth2 = second_order_smoothness_loss(img1_l2, flow_l2)
        else:
            smooth1 = first_order_smoothness_loss(img1, flow_pred)
            smooth2 = second_order_smoothness_loss(img1, flow_pred)
        #calculate difference between predicted flow and ground truth flow
        flow_error = F.mse_loss(flow_pred, flow)
        return photo, smooth1, smooth2, flow_error
    
    def general_step_occ(self, batch, batch_idx, mode):
        imgs, flow, occ = batch
        img1, img2 = imgs[:, 0:3, :, :], imgs[:, 3:6, :, :]
        # flow prediction
        if self.model == 'pwc':
            flow_pred, flow_l2 = self(imgs)
        else:
            flow_pred = self(imgs)
        img_warped = self.warp(img2, flow_pred)
        
        # calculate photometric error
        photo = photometric_error(img_warped, img1, occ)
        if self.model == 'pwc':
            img1_l2 = F.interpolate(img1, scale_factor=0.25, mode='bilinear', align_corners=True)
            smooth1 = first_order_smoothness_loss(img1_l2, flow_l2)
            smooth2 = second_order_smoothness_loss(img1_l2, flow_l2)
        else:
            smooth1 = first_order_smoothness_loss(img1, flow_pred)
            smooth2 = second_order_smoothness_loss(img1, flow_pred)
        #calculate difference between predicted flow and ground truth flow
        flow_error = F.mse_loss(flow_pred, flow)
        return photo, smooth1, smooth2, flow_error
    
    def general_step_occ_aware(self, batch, batch_idx, mode):
        if len(batch) == 2:
            imgs, flow = batch
        elif len(batch) == 3:
            imgs, flow, occ = batch
        else:
            raise ValueError('Not supported dataset')
        img1, img2 = imgs[:, 0:3, :, :], imgs[:, 3:6, :, :]
        # flow prediction
        if self.model == 'pwc':
            flow_pred, flow_l2 = self(imgs)
        else:
            flow_pred = self(imgs)
        img_warped = self.warp(img2, flow_pred)
        # range map calculation
        with torch.no_grad():
            # backward flow prediction
            if self.model == 'pwc':
                back_flow_pred, _ = self(torch.cat((img2, img1), dim=1))
            else:
                back_flow_pred = self(torch.cat((img2, img1), dim=1))
            # calculate range map
            range_map = self.compute_range_map(back_flow_pred)
            # compute occlusion mask
            # 0: non-occluded, 1: occluded
            occ_pred = 1. - torch.clamp(range_map, min=0.0, max=1.0)
            
        # calculate photometric error
        photo = photometric_error(img_warped, img1, occ_pred)
        if self.model == 'pwc':
            img1_l2 = F.interpolate(img1, scale_factor=0.25, mode='bilinear', align_corners=True)
            smooth1 = first_order_smoothness_loss(img1_l2, flow_l2)
            smooth2 = second_order_smoothness_loss(img1_l2, flow_l2)
        else:
            smooth1 = first_order_smoothness_loss(img1, flow_pred)
            smooth2 = second_order_smoothness_loss(img1, flow_pred)
        #calculate difference between predicted flow and ground truth flow
        flow_error = F.mse_loss(flow_pred, flow)
        # calculate the photometric error in occluded region
        photo_occ = photometric_error(img_warped, img1, 1.0-occ_pred)
        if occ is not None:
            occ_error = F.binary_cross_entropy(occ, occ_pred)
            return photo, smooth1, smooth2, flow_error, photo_occ, occ_error
        return photo, smooth1, smooth2, flow_error, photo_occ
    
    def training_step(self, batch, batch_idx):
        if not self.occ_aware:
            if not self.with_occ:
                photo, smooth1, smooth2, flow_error = self.general_step(batch, batch_idx, 'train')
            else:
                photo, smooth1, smooth2, flow_error = self.general_step_occ(batch, batch_idx, 'train')
        else:
            losses = self.general_step_occ_aware(batch, batch_idx, 'train')
            if len(losses) == 5:
                photo, smooth1, smooth2, flow_error, photo_occ = losses[0], losses[1], losses[2], losses[3], losses[4]
            else:
                photo, smooth1, smooth2, flow_error, photo_occ, occ_error = losses[0], losses[1], losses[2], losses[3], losses[4], losses[5]
                
        loss = self.photo_weight * photo + self.smooth1_weight * smooth1 + self.smooth2_weight * smooth2
        
        if self.global_step % self.log_every_n_steps == 0: 
            tensorboard = self.logger.experiment
            tensorboard.add_scalar("train_photometric", photo, global_step = self.global_step)
            tensorboard.add_scalar("train_smooth1", smooth1, global_step = self.global_step)
            tensorboard.add_scalar("train_smooth2", smooth2, global_step = self.global_step)
            tensorboard.add_scalar("train_flow_error", flow_error, global_step = self.global_step)
            if self.occ_aware and len(batch) == 3:
                tensorboard.add_scalar("train_occ_error", occ_error, global_step = self.global_step)
            if self.occ_aware:
                tensorboard.add_scalar("train_photometric_occ", photo_occ, global_step = self.global_step)
        return loss
    
    def training_epoch_end(self, outputs): 
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        tensorboard = self.logger.experiment
        tensorboard.add_scalars("losses", {"train_loss": avg_loss}, global_step = self.current_epoch)
    
    def validation_step(self, batch, batch_idx):
        if not self.occ_aware:
            if not self.with_occ:
                photo, smooth1, smooth2, flow_error = self.general_step(batch, batch_idx, 'val')
            else:
                photo, smooth1, smooth2, flow_error = self.general_step_occ(batch, batch_idx, 'val')
        else:
            losses = self.general_step_occ_aware(batch, batch_idx, 'val')
            if len(losses) == 5:
                photo, smooth1, smooth2, flow_error, photo_occ = losses[0], losses[1], losses[2], losses[3], losses[4]
            else:
                photo, smooth1, smooth2, flow_error, photo_occ, occ_error = losses[0], losses[1], losses[2], losses[3], losses[4], losses[5]
                
        loss = self.photo_weight * photo + self.smooth1_weight * smooth1 + self.smooth2_weight * smooth2
        
        if self.global_step % self.log_every_n_steps == 0: 
            tensorboard = self.logger.experiment
            tensorboard.add_scalar("val_photometric", photo, global_step = self.global_step)
            tensorboard.add_scalar("val_smooth1", smooth1, global_step = self.global_step)
            tensorboard.add_scalar("val_smooth2", smooth2, global_step = self.global_step)
            tensorboard.add_scalar("val_flow_error", flow_error, global_step = self.global_step)
            if self.occ_aware and len(batch) == 3:
                tensorboard.add_scalar("val_occ_error", occ_error, global_step = self.global_step)
            if self.occ_aware:
                tensorboard.add_scalar("val_photometric_occ", photo_occ, global_step = self.global_step)
        return loss

    def validation_epoch_end(self, outputs): 
        avg_loss = torch.stack([x for x in outputs]).mean()
        tensorboard = self.logger.experiment
        tensorboard.add_scalars("losses", {"val_loss": avg_loss}, global_step = self.current_epoch)
        self.log('monitored_loss', avg_loss, prog_bar= True, logger = True)    
    
    def test_step(self, batch, batch_idx):
        if not self.occ_aware:
            if not self.with_occ:
                photo, smooth1, smooth2, flow_error = self.general_step(batch, batch_idx, 'test')
            else:
                photo, smooth1, smooth2, flow_error = self.general_step_occ(batch, batch_idx, 'test')
        else:
            losses = self.general_step_occ_aware(batch, batch_idx, 'test')
            if len(losses) == 5:
                photo, smooth1, smooth2, flow_error, photo_occ = losses[0], losses[1], losses[2], losses[3], losses[4]
            else:
                photo, smooth1, smooth2, flow_error, photo_occ, occ_error = losses[0], losses[1], losses[2], losses[3], losses[4], losses[5]
                
        loss = self.photo_weight * photo + self.smooth1_weight * smooth1 + self.smooth2_weight * smooth2
        
        if self.global_step % self.log_every_n_steps == 0: 
            tensorboard = self.logger.experiment
            tensorboard.add_scalar("test_photometric", photo, global_step = self.global_step)
            tensorboard.add_scalar("test_smooth1", smooth1, global_step = self.global_step)
            tensorboard.add_scalar("test_smooth2", smooth2, global_step = self.global_step)
            tensorboard.add_scalar("test_flow_error", flow_error, global_step = self.global_step)
            if self.occ_aware and len(batch) == 3:
                tensorboard.add_scalar("test_occ_error", occ_error, global_step = self.global_step)
            if self.occ_aware:
                tensorboard.add_scalar("test_photometric_occ", photo_occ, global_step = self.global_step)
        return loss

    def test_epoch_end(self, outputs): 
        avg_loss = torch.stack([x for x in outputs]).mean()
        tensorboard = self.logger.experiment
        tensorboard.add_scalars("losses", {"test_loss": avg_loss}, global_step = self.current_epoch)
    
    def configure_optimizers(self):
        return Adam(self.parameters(), self.lr)

class InpaintingStageModel(pl.LightningModule):
    """
    Networks for training inpainting task. Two options for loss: pixel-wise(l1, l2) or perceptual(VGG) loss 
    """
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.lr = hparams.get('learning_rate',1e-4)
        self.log_every_n_steps = hparams.get('log_every_n_steps', 20)
        self.img_size = hparams.get('img_size')
        self.batch_size = hparams.get('batch_size', 16)
        self.n_display_images = hparams.get('n_display_images', 1)
        self.result_dir = hparams.get('result_dir','')
        self.log_image_every_epoch = hparams.get('log_image_every_epoch',10)
        self.reconst_weight = hparams.get('reconst_weight', 1.0)
        self.loss_type = hparams.get('loss_type', 'vgg')
        self.org = hparams.get('org', False)
        self.model = hparams.get('model', 'simple')
        assert self.model in ['simple', 'gated']
        if self.model == 'simple': 
            self.generator = InpaintingNet()
        else: 
            if self.org: 
                self.generator = InpaintSANetOrg(img_size=self.img_size)
            else: 
                self.generator = InpaintSANet(img_size=self.img_size)
        assert self.loss_type in ['pixel-wise', 'vgg']
        if self.loss_type == 'vgg': 
            self.loss_func1 = VGGPerceptualLoss(w = [1.0,1.0,1.0,1.0])
            self.loss_func2 = ReconLoss(1.0,1.0,1.0,1.0)
            for param in self.loss_func1.parameters():
                param.requires_grad = False
            for param in self.loss_func2.parameters():
                param.requires_grad = False
        else: 
            self.loss_func = ReconLoss(1.0,1.0,1.0,1.0)
            for param in self.loss_func.parameters():
                param.requires_grad = False
    @property
    def is_cuda(self):
        return next(self.parameters()).is_cuda
    
    
    def save_state_dict(self, path):
        torch.save(self.state_dict(), path)
        
    def general_step(self, batch, batch_idx):
        _, imgs, masks = batch
        # inpainting
        if self.model == 'gated': 
            coarse_imgs, recon_imgs = self.generator(imgs,masks)
        else: 
            recon_imgs = self.generator(imgs,masks)
            coarse_imgs = None

        if self.loss_type == 'pixel-wise': 
            loss, _, _ = self.loss_func(imgs, recon_imgs, masks, coarse_imgs)
            return loss
        elif self.loss_type == 'vgg': 
            
            #mean = imgs.new_tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
            #std = imgs.new_tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
            #vgg_loss = self.loss_func1(((recon_imgs*0.5+0.5)-mean)/std, ((imgs*0.5+0.5)-mean)/std)
            vgg_loss = self.loss_func1(recon_imgs, imgs)
            recon_loss, _, _ = self.loss_func2(imgs, recon_imgs, masks, coarse_imgs)
            return vgg_loss, recon_loss 
        else:
            raise ValueError(f'Unsupported loss type: {self.loss_type}')
    
    def training_step(self, batch, batch_idx):
        if self.loss_type =='vgg': 
            vgg_loss, recon_loss = self.general_step(batch, batch_idx)
            loss = vgg_loss + self.reconst_weight*recon_loss
        else: 
            loss = self.general_step(batch, batch_idx)
        if self.global_step % self.log_every_n_steps == 0: 
            tensorboard = self.logger.experiment
            tensorboard.add_scalars("step_loss",{"train_loss": loss}, global_step = self.global_step)
            if self.loss_type == 'vgg': 
                tensorboard.add_scalars("vgg_loss",{"train_loss": vgg_loss}, global_step = self.global_step)
                tensorboard.add_scalars("reconst_loss",{"train_loss": recon_loss}, global_step = self.global_step)
        return loss
    def training_epoch_end(self, outputs): 
        loss = torch.stack([x['loss'] for x in outputs]).mean()
        tensorboard = self.logger.experiment
        tensorboard.add_scalars("epoch_loss", {"train_loss": loss}, global_step = self.current_epoch)
    
    def validation_step(self, batch, batch_idx):
        _, imgs, masks = batch
        if self.model == 'gated': 
            coarse_imgs, recon_imgs = self.generator(imgs,masks)
        else: 
            recon_imgs = self.generator(imgs,masks)
            coarse_imgs = None

        if self.loss_type == 'pixel-wise': 
            loss, _, _ = self.loss_func(imgs, recon_imgs, masks, coarse_imgs)
        elif self.loss_type == 'vgg': 
            #mean = imgs.new_tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
            #std = imgs.new_tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
            #vgg_loss = self.loss_func1(((recon_imgs*0.5+0.5)-mean)/std, ((imgs*0.5+0.5)-mean)/std)

            vgg_loss = self.loss_func1(recon_imgs, imgs)
            recon_loss, _, _ = self.loss_func2(imgs, recon_imgs, masks, coarse_imgs)
            loss = vgg_loss + self.reconst_weight*recon_loss
        else:
            raise ValueError(f'Unsupported loss type: {self.loss_type}')
        complete_imgs = recon_imgs * masks + imgs * (1 - masks)

        if batch_idx == 0: 
            tensorboard = self.logger.experiment
            tensorboard.add_scalars("step_loss", {"val_loss": loss}, global_step = self.global_step)
            if self.loss_type == 'vgg': 
                tensorboard.add_scalars("vgg_loss",{"val_loss": vgg_loss}, global_step = self.global_step)
                tensorboard.add_scalars("reconst_loss",{"val_loss": recon_loss}, global_step = self.global_step)
            if self.current_epoch % self.log_image_every_epoch == 0: 

                val_save_dir = os.path.join(self.result_dir, "val_{}".format(self.current_epoch))
                val_save_real_dir = os.path.join(val_save_dir, "real")
                val_save_gen_dir = os.path.join(val_save_dir, "gen")
                if not os.path.exists(val_save_real_dir):
                    os.makedirs(val_save_real_dir)
                    os.makedirs(val_save_gen_dir)

                saved_images =img2photo(torch.cat([ imgs * (1 - masks),recon_imgs, imgs, complete_imgs], dim=2))
                h, w = saved_images.shape[1]//4, saved_images.shape[2]
                j= 0
                n_display_images = self.n_display_images if self.batch_size > self.n_display_images else self.batch_size
                for val_img in saved_images:
                    real_img = val_img[(2*h):(3*h), :,:]
                    gen_img = val_img[(3*h):,:,:] 
                    real_img = Image.fromarray(real_img.astype(np.uint8))
                    gen_img = Image.fromarray(gen_img.astype(np.uint8))
                    real_img.save(os.path.join(val_save_real_dir, "{}.png".format(j)))
                    gen_img.save(os.path.join(val_save_gen_dir, "{}.png".format(j)))
                    j += 1
                    if j == n_display_images: 
                        break
                tensorboard = self.logger.experiment
                tensorboard.add_images('val/imgs', saved_images[:n_display_images].astype(np.uint8), self.current_epoch, dataformats = 'NHWC')
                
        return  loss
    def validation_epoch_end(self,outputs):
        avg_loss = torch.stack([x for x in outputs]).mean() 
        tensorboard = self.logger.experiment
        tensorboard.add_scalars("epoch_loss", {"val_loss": avg_loss}, global_step = self.current_epoch)
        self.log('monitored_loss', avg_loss, prog_bar= True, logger = True)

    def test_step(self, batch, batch_idx):
        if self.loss_type =='vgg': 
            vgg_loss, recon_loss = self.general_step(batch, batch_idx)
            loss = vgg_loss + self.reconst_weight*recon_loss
        else: 
            loss = self.general_step(batch, batch_idx)
        if batch_idx == 0: 
            tensorboard = self.logger.experiment
            tensorboard.add_scalars("step_loss",{"test_loss": loss}, global_step = self.global_step)
            if self.loss_type == 'vgg': 
                tensorboard.add_scalars("vgg_loss",{"test_loss": vgg_loss}, global_step = self.global_step)
                tensorboard.add_scalars("reconst_loss",{"test_loss": recon_loss}, global_step = self.global_step)
        return loss

    def test_epoch_end(self, outputs): 
        avg_loss = torch.stack([x for x in outputs]).mean() 
        tensorboard = self.logger.experiment
        tensorboard.add_scalars("epoch_loss", {"test_loss": avg_loss}, global_step = self.current_epoch)

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), self.lr)
#        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20, verbose = True)
        return optimizer
#        return {
#             'optimizer': optimizer,
#             'lr_scheduler': scheduler,
#             'monitor': 'monitored_loss'
#        }

class InpaintingGConvModel(pl.LightningModule): 
    def __init__(self, hparams): 
        super().__init__()
        self.hparams = hparams
        self.lr = hparams.get('learning_rate',1e-4)
        self.decay = hparams.get('decay',0.0)
        self.org = hparams.get('org', False)
        self.img_size = hparams.get('img_size')
        self.batch_size = hparams.get('batch_size', 16)
        self.n_display_images = hparams.get('n_display_images', 1)
        self.result_dir = hparams.get('result_dir','')
        self.log_image_every_epoch = hparams.get('log_image_every_epoch',10)
        #self.reconst_weight = hparams.get('reconst_weight', 1.0)
        self.loss_type = hparams.get('loss_type', 'vgg')
        self.log_every_n_steps = hparams.get('log_every_n_steps',20)
        self.model = hparams['model']
        assert self.model in ['gated', 'simple']
        if self.model == 'gated': 
            if self.org:
                self.generator = InpaintSANetOrg(img_size=self.img_size)
                
            else:
                self.generator = InpaintSANet(img_size=self.img_size)
                
        else: 
            self.generator = InpaintingNet()
        if self.org: 
            self.discriminator = InpaintSADiscriminatorOrg(img_size=self.img_size)
        else: 
            self.discriminator = InpaintSADiscriminator(img_size=self.img_size)
        

        assert self.loss_type in ['pixel-wise', 'vgg']
        if self.loss_type == 'vgg': 
            self.loss_func = VGGPerceptualLoss(w = [1.0,1.0,1.0,1.0])
            for param in self.loss_func.parameters():
                param.requires_grad = False
        else: 
            self.loss_func = ReconLoss(1.0,1.0,1.0,1.0)
            for param in self.loss_func.parameters():
                param.requires_grad = False

    def forward(self, inputs): 
        return self.generator(inputs)
    def training_step(self, batch, batch_idx, optimizer_idx): 
        _, imgs, masks = batch 
        (optD, optG) = self.optimizers()
        #train discriminator
        if self.model == 'gated': 
            coarse_imgs, recon_imgs = self.generator(imgs,masks)
        else: 
            recon_imgs = self.generator(imgs,masks)
        complete_imgs = recon_imgs * masks + imgs * (1 - masks)
        pos_imgs = torch.cat([imgs, masks], dim=1)
        neg_imgs = torch.cat([complete_imgs, masks], dim=1)
        pos_neg_imgs = torch.cat([pos_imgs, neg_imgs], dim=0)
        pred_pos_neg = self.discriminator(pos_neg_imgs)
        pred_pos, pred_neg = torch.chunk(pred_pos_neg, 2, dim=0)
        Dloss = SNDisLoss()
        d_loss = Dloss(pred_pos, pred_neg)
        self.manual_backward(d_loss, optD, retain_graph = True)
        optD.step()
        optD.zero_grad()
        optG.zero_grad()
        #train generator 
        pred_neg = self.discriminator(neg_imgs)
        GANLoss = SNGenLoss(1.0)
        g_loss = GANLoss(pred_neg)
        if self.loss_type == 'vgg': 
            content_loss = self.loss_func(recon_imgs, imgs)
            Recon_Loss = ReconLoss(1.0,1.0,1.0,1.0)
            if self.model == 'gated': 
                _, r_occluded, r_non_occluded = Recon_Loss(imgs, recon_imgs, masks, coarse_imgs)
            else: 
                _, r_occluded, r_non_occluded = Recon_Loss(imgs, recon_imgs, masks)
            
        else: 
            if self.model == 'gated': 
                content_loss, r_occluded, r_non_occluded = self.loss_func(imgs, recon_imgs, masks, coarse_imgs)
            else: 
                content_loss, r_occluded, r_non_occluded = self.loss_func(imgs, recon_imgs, masks)
        whole_loss = g_loss + content_loss 
        self.manual_backward(whole_loss, optG)
        optG.step()
        optG.zero_grad()
        optD.zero_grad()
        if self.global_step % self.log_every_n_steps == 0:
            if self.global_rank == 0: 
                tensorboard = self.logger.experiment
                tensorboard.add_scalars("whole_loss",{"train_loss": whole_loss} , global_step = self.global_step)
                tensorboard.add_scalars("content_loss", {"train_loss": content_loss}, global_step = self.global_step)
                tensorboard.add_scalars("gan_loss", {"train_loss":g_loss}, global_step = self.global_step)
                tensorboard.add_scalars("discriminator_loss",{ "train_loss":d_loss}, global_step = self.global_step)
        return {'loss': content_loss, 'occluded': r_occluded, 'non_occluded': r_non_occluded}
    def training_epoch_end(self, outputs): 
        avg_occluded = torch.stack([x['occluded'] for x in outputs]).mean()
        avg_non_occluded = torch.stack([x['non_occluded'] for x in outputs]).mean()
        avg_content = torch.stack([x['loss'] for x in outputs]).mean()
        if self.global_rank == 0:
            tensorboard = self.logger.experiment
            tensorboard.add_scalars("occluded_epoch_loss", {"train_loss": avg_occluded}, global_step = self.current_epoch)
            tensorboard.add_scalars("non_occluded_epoch_loss", {"train_loss": avg_non_occluded}, global_step = self.current_epoch)
            tensorboard.add_scalars("content_epoch_loss", {"train_loss": avg_content}, global_step = self.current_epoch)

    def validation_step(self, batch, batch_idx): 
        _, imgs, masks = batch 
        #train discriminator
        if self.model == 'gated': 
            coarse_imgs, recon_imgs = self.generator(imgs,masks)
        else: 
            recon_imgs = self.generator(imgs,masks)
        complete_imgs = recon_imgs * masks + imgs * (1 - masks)
        pos_imgs = torch.cat([imgs, masks], dim=1)
        neg_imgs = torch.cat([complete_imgs, masks], dim=1)
        pos_neg_imgs = torch.cat([pos_imgs, neg_imgs], dim=0)
        pred_pos_neg = self.discriminator(pos_neg_imgs)
        pred_pos, pred_neg = torch.chunk(pred_pos_neg, 2, dim=0)
        
        GANLoss = SNGenLoss(1.0)
        g_loss = GANLoss(pred_neg)

        if self.loss_type == 'vgg': 
            content_loss = self.loss_func(recon_imgs, imgs)
            Recon_Loss = ReconLoss(1.0,1.0,1.0,1.0)
            if self.model == 'gated': 
                _, r_occluded, r_non_occluded = Recon_Loss(imgs, recon_imgs, masks, coarse_imgs)
            else: 
                _, r_occluded, r_non_occluded = Recon_Loss(imgs, recon_imgs, masks)
            
        else: 
            if self.model == 'gated': 
                content_loss, r_occluded, r_non_occluded = self.loss_func(imgs, recon_imgs, masks, coarse_imgs)
            else: 
                content_loss, r_occluded, r_non_occluded = self.loss_func(imgs, recon_imgs, masks)
        whole_loss = g_loss + content_loss


       
        Dloss = SNDisLoss()
        d_loss = Dloss(pred_pos, pred_neg)
       
        if batch_idx == 0:
            if self.global_rank == 0:
                tensorboard = self.logger.experiment
                tensorboard.add_scalars("whole_loss",{"val_loss": whole_loss} , global_step = self.global_step)
                tensorboard.add_scalars("content_loss", {"val_loss": content_loss}, global_step = self.global_step)
                tensorboard.add_scalars("gan_loss", {"val_loss":g_loss}, global_step = self.global_step)
                tensorboard.add_scalars("discriminator_loss",{ "val_loss":d_loss}, global_step = self.global_step)
                if self.current_epoch % self.log_image_every_epoch == 0: 
                    val_save_dir = os.path.join(self.result_dir, "val_{}".format(self.current_epoch))
                    val_save_real_dir = os.path.join(val_save_dir, "real")
                    val_save_gen_dir = os.path.join(val_save_dir, "gen")
                    if not os.path.exists(val_save_real_dir):
                        os.makedirs(val_save_real_dir)
                        os.makedirs(val_save_gen_dir)

                    saved_images =img2photo(torch.cat([ imgs * (1 - masks), recon_imgs, imgs, complete_imgs], dim=2))
                    h, w = saved_images.shape[1]//4, saved_images.shape[2]
                    j= 0
                    n_display_images = self.n_display_images if self.batch_size > self.n_display_images else self.batch_size
                    for val_img in saved_images:
                        real_img = val_img[(2*h):(3*h), :,:]
                        gen_img = val_img[(3*h):,:,:] 
                        real_img = Image.fromarray(real_img.astype(np.uint8))
                        gen_img = Image.fromarray(gen_img.astype(np.uint8))
                        real_img.save(os.path.join(val_save_real_dir, "{}.png".format(j)))
                        gen_img.save(os.path.join(val_save_gen_dir, "{}.png".format(j)))
                        j += 1
                        if j == n_display_images: 
                            break
                    tensorboard = self.logger.experiment
                    tensorboard.add_images('val/imgs', saved_images[:n_display_images].astype(np.uint8), self.current_epoch, dataformats = 'NHWC')

        return (r_occluded, r_non_occluded, content_loss)
    def validation_epoch_end(self, outputs): 
        avg_occluded = torch.stack([x[0] for x in outputs]).mean()
        avg_non_occluded = torch.stack([x[1] for x in outputs]).mean()
        avg_content = torch.stack([x[2] for x in outputs]).mean()
        if self.global_rank == 0:
            tensorboard = self.logger.experiment
            tensorboard.add_scalars("occluded_epoch_loss", {"val_loss": avg_occluded}, global_step = self.current_epoch)
            tensorboard.add_scalars("non_occluded_epoch_loss", {"val_loss": avg_non_occluded}, global_step = self.current_epoch)
            tensorboard.add_scalars("content_epoch_loss", {"val_loss": avg_content}, global_step = self.current_epoch)
            self.log('monitored_loss', avg_content, prog_bar= True, logger = True, sync_dist=True)

    def test_step(self, batch, batch_idx): 
        _, imgs, masks = batch 
        #train discriminator
        coarse_imgs, recon_imgs = self.generator(imgs,masks)
        complete_imgs = recon_imgs * masks + imgs * (1 - masks)
        pos_imgs = torch.cat([imgs, masks], dim=1)
        neg_imgs = torch.cat([complete_imgs, masks], dim=1)
        pos_neg_imgs = torch.cat([pos_imgs, neg_imgs], dim=0)
        pred_pos_neg = self.discriminator(pos_neg_imgs)
        pred_pos, pred_neg = torch.chunk(pred_pos_neg, 2, dim=0)
        
        GANLoss = SNGenLoss(1.0)
        g_loss = GANLoss(pred_neg)

        if self.loss_type == 'vgg': 
            content_loss = self.loss_func(recon_imgs, imgs)
            Recon_Loss = ReconLoss(1.0,1.0,1.0,1.0)
            if self.model == 'gated': 
                _, r_occluded, r_non_occluded = Recon_Loss(imgs, recon_imgs, masks, coarse_imgs)
            else: 
                _, r_occluded, r_non_occluded = Recon_Loss(imgs, recon_imgs, masks)
            
        else: 
            if self.model == 'gated': 
                content_loss, r_occluded, r_non_occluded = self.loss_func(imgs, recon_imgs, masks, coarse_imgs)
            else: 
                content_loss, r_occluded, r_non_occluded = self.loss_func(imgs, recon_imgs, masks)
        whole_loss = g_loss + content_loss
       
        Dloss = SNDisLoss()
        d_loss = Dloss(pred_pos, pred_neg)
       
        if batch_idx == 0:
            if self.global_rank == 0:
                tensorboard = self.logger.experiment
                tensorboard.add_scalars("whole_loss",{"test_loss": whole_loss} , global_step = self.global_step)
                tensorboard.add_scalars("content_loss", {"test_loss": content_loss}, global_step = self.global_step)
                tensorboard.add_scalars("gan_loss", {"test_loss":g_loss}, global_step = self.global_step)
                tensorboard.add_scalars("discriminator_loss",{ "test_loss":d_loss}, global_step = self.global_step)
        return  (r_occluded, r_non_occluded, content_loss)
    def test_epoch_end(self, outputs): 
        avg_occluded = torch.stack([x[0] for x in outputs]).mean()
        avg_non_occluded = torch.stack([x[1] for x in outputs]).mean()
        avg_content = torch.stack([x[2] for x in outputs]).mean()
        if self.global_rank == 0:
            tensorboard = self.logger.experiment
            tensorboard.add_scalars("occluded_epoch_loss", {"test_loss": avg_occluded}, global_step = self.current_epoch)
            tensorboard.add_scalars("non_occluded_epoch_loss", {"test_loss": avg_non_occluded}, global_step = self.current_epoch)
            tensorboard.add_scalars("content_epoch_loss", {"test_loss": avg_content}, global_step = self.current_epoch)
    def configure_optimizers(self): 
        optG = torch.optim.Adam(self.generator.parameters(), lr=self.lr, weight_decay=self.decay)
        optD = torch.optim.Adam(self.discriminator.parameters(), lr=4*self.lr, weight_decay=self.decay)
        return optD, optG


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
        self.reconst_weight = hparams.get('reconst_weight', 1.0)
        self.log_every_n_steps = hparams.get('log_every_n_steps', 1)
        
        flow_root = hparams.get('flow_root', None)
        inpainting_root = hparams.get('inpainting_root', None)
        supervised_flow = hparams.get('supervised_flow', False)
        self.flow_pred = SimpleFlowNet()
        self.occ_pred = SimpleOcclusionNet()
        self.inpainting = InpaintingNet()
        if flow_root:
            if supervised_flow: 
                self.flow_pred = FlowModel.load_from_checkpoint(flow_root).model
            else: 
                self.flow_pred = FlowStageModel.load_from_checkpoint(flow_root).flow_pred
        if inpainting_root:
            self.inpainting = InpaintingStageModel.load_from_checkpoint(inpainting_root).model
        
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
        
        output = F.grid_sample(img, vgrid, align_corners=True)
        
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
            imgs, _, occ = batch
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
        
        # calculate the errors
        smoothness_term = first_order_smoothness_loss(img1, flow_pred)
        photo_error = photometric_error(img_warped * (1 - occ_pred), img1 * (1 - occ_pred))
        reconst_error = photometric_error(img_warped * occ_pred, img1 * occ_pred)
        # calculate BCE error if occ is available
        if occ is not None:
            bce_loss = F.binary_cross_entropy(occ_pred, occ)
            return photo_error, reconst_error, smoothness_term, bce_loss
        return photo_error, reconst_error, smoothness_term
    
    
    def training_step(self, batch, batch_idx):
        losses = self.general_step(batch, batch_idx, 'train')
        if len(losses) == 3:
            photo_error, reconst_error, smoothness_term = losses[0], losses[1], losses[2]
        else:
            photo_error, reconst_error, smoothness_term, bce_loss = losses[0], losses[1], losses[2], losses[3]
        loss = photo_error + self.reconst_weight * reconst_error + self.smoothness_weight * smoothness_term

        if self.global_step % self.log_every_n_steps == 0: 
            tensorboard = self.logger.experiment
            tensorboard.add_scalar("train_photometric", photo_error, global_step = self.global_step)
            tensorboard.add_scalar("train_reconst", reconst_error, global_step = self.global_step)
            tensorboard.add_scalar("train_smoothness", smoothness_term, global_step = self.global_step)
            if bce_loss is not None:
                tensorboard.add_scalar("train_bce_loss", bce_loss, global_step = self.global_step)
        return loss
    def training_epoch_end(self, outputs): 
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        tensorboard = self.logger.experiment
        tensorboard.add_scalars("losses", {"train_loss": avg_loss}, global_step = self.current_epoch)
    
    
    def validation_step(self, batch, batch_idx):
        losses = self.general_step(batch, batch_idx, 'val')
        if len(losses) == 3:
            photo_error, reconst_error, smoothness_term = losses[0], losses[1], losses[2]
        else:
            photo_error, reconst_error, smoothness_term, bce_loss = losses[0], losses[1], losses[2], losses[3]
        loss = photo_error + self.reconst_weight * reconst_error + self.smoothness_weight * smoothness_term
        if batch_idx == 0: 
            tensorboard = self.logger.experiment
            tensorboard.add_scalar("val_photometric", photo_error, global_step = self.global_step)
            tensorboard.add_scalar("val_reconst", reconst_error, global_step = self.global_step)
            tensorboard.add_scalar("val_smoothness", smoothness_term, global_step = self.global_step)
            if bce_loss is not None:
                tensorboard.add_scalar("val_bce_loss", bce_loss, global_step = self.global_step)
        return loss
    
    def validation_epoch_end(self, outputs): 
        avg_loss = torch.stack([x for x in outputs]).mean()
        tensorboard = self.logger.experiment
        tensorboard.add_scalars("losses", {"val_loss": avg_loss}, global_step = self.current_epoch)
        self.log('monitored_loss', avg_loss, prog_bar= True, logger = True)

    def test_step(self, batch, batch_idx):
        losses = self.general_step(batch, batch_idx, 'test')
        if len(losses) == 3:
            photo_error, reconst_error, smoothness_term = losses[0], losses[1], losses[2]
        else:
            photo_error, reconst_error, smoothness_term, bce_loss = losses[0], losses[1], losses[2], losses[3]
        loss = photo_error + self.reconst_weight * reconst_error + self.smoothness_weight * smoothness_term
        if batch_idx == 0:
            tensorboard = self.logger.experiment
            tensorboard.add_scalar("test_photometric", photo_error, global_step = self.global_step)
            tensorboard.add_scalar("test_reconst", reconst_error, global_step = self.global_step)
            tensorboard.add_scalar("test_smoothness", smoothness_term, global_step = self.global_step)
            if bce_loss is not None:
                tensorboard.add_scalar("test_bce_loss", bce_loss, global_step = self.global_step)
        return loss

    def test_epoch_end(self, outputs): 
        avg_loss = torch.stack([x for x in outputs]).mean()
        tensorboard = self.logger.experiment
        tensorboard.add_scalars("losses", {"test_loss": avg_loss}, global_step = self.current_epoch)

    def configure_optimizers(self):
        return Adam(self.parameters(), self.lr)


class TwoStageModelGC(pl.LightningModule):
    """
    Training with two stages with ground truth optical flow:
    First stage: optical flow and occlusion map prediction
    Second stage: inpainting network predicts pixel value for the occluded regions 
    """
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.lr = hparams['learning_rate']
        self.reconst_weight = hparams.get('reconst_weight', 1.0)
        self.log_every_n_steps = hparams.get('log_every_n_steps', 1)
        self.occ_pred = SimpleOcclusionNet()
        inpainting_root = hparams.get('inpainting_root', None)
        self.inpainting_stage = hparams.get('inpainting_stage', 'gated')

        self.smooth1_weight = hparams.get('smooth1_weight', 1.0)
        self.smooth2_weight = hparams.get('smooth2_weight', 0.0)
        
        self.batch_size = hparams.get('batch_size', 16)
        self.n_display_images = hparams.get('n_display_images', 1)
        self.result_dir = hparams.get('result_dir','')
        self.log_image_every_epoch = hparams.get('log_image_every_epoch',10)
        print('result dir inside inpainting model is {}'.format(self.result_dir))
        
        self.inpainting = InpaintingGConvModel.load_from_checkpoint(inpainting_root).generator
        
        
        
        # we will freeze the inpainting network
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
        
        output = F.grid_sample(img, vgrid, align_corners=True)
        
        return output
    
    @property
    def is_cuda(self):
        return next(self.parameters()).is_cuda
    
    
    def save_state_dict(self, path):
        torch.save(self.state_dict(), path)
        
    def general_step(self, batch, batch_idx, mode):
        occ = None
        if not isinstance(batch, (list, tuple)):
            imgs = batch
        elif len(batch) == 2:
            imgs, flow = batch
        elif len(batch) == 3:
            imgs, flow, occ = batch
        else:
            raise ValueError('Not supported dataset')
        img1, img2 = imgs[:, 0:3, :, :], imgs[:, 3:6, :, :]
        # warp image use ground truth optical flow
        img_warped = self.warp(img2, flow)
        # occlusion prediction
        occ_pred_soft = self.occ_pred(imgs)
        occ_pred = (torch.where(occ_pred_soft > 0.5, 1.0, 0.0) - occ_pred_soft).detach() + occ_pred_soft
        
        #add smoothness loss to occlusion map 
        smoothness_loss = first_order_smoothness_loss(img_warped, occ_pred_soft)
        # get completed image
        if self.inpainting_stage =='simple': 
            img_completed = self.inpainting(img_warped, occ_pred_soft)
        elif self.inpainting_stage =='gated' or self.inpainting_stage == 'gated_org': 
            _,img_completed = self.inpainting(img_warped, occ_pred_soft)
        
        # calculate the reconstruction error
        #photometric_error = charbonnier_loss((img_warped - img1) * (1 - occ_pred), reduction=False).sum() / (3*(1 - occ_pred).sum() + 1e-16)
        #reconst_error = charbonnier_loss(torch.abs(img1 - img_completed) * occ_pred, reduction=False).sum() / (3*occ_pred.sum() + 1e-16)
        photo_error = photometric_error(img_warped * (1 - occ_pred_soft), img1 * (1 - occ_pred_soft))
        photo_error_occluded = photometric_error(img_warped * occ_pred_soft, img1 *occ_pred_soft)
        #reconst_error = photometric_error(img_completed * occ_pred_soft, img1 * occ_pred_soft)

        #----------------------------------------------------------------------------------
        reconstructed_img = occ_pred_soft*img_completed + (1-occ_pred_soft)*img1
        reconst_error = photometric_error(occ_pred_soft*reconstructed_img, occ_pred_soft*img1)
        #----------------------------------------------------------------------------------


        # calculate BCE error if occ is available
        if occ is not None:
            bce_loss = F.binary_cross_entropy(occ_pred_soft, occ)
            return photo_error, reconst_error, smoothness_loss, bce_loss, photo_error_occluded
        return photo_error, reconst_error, smoothness_loss, photo_error_occluded
    
    
    def training_step(self, batch, batch_idx):
        losses = self.general_step(batch, batch_idx, 'train')
        bce_loss = None
        if len(losses) == 3:
            photo_error, reconst_error, smoothness_loss, photo_error_occluded = losses[0], losses[1], losses[2], losses[3]
        else:
            photo_error, reconst_error, smoothness_loss, bce_loss, photo_error_occluded = losses[0], losses[1], losses[2], losses[3], losses[4]
        loss = photo_error + self.reconst_weight * reconst_error + self.smooth1_weight * smoothness_loss
        if self.global_step % self.log_every_n_steps == 0: 
            tensorboard = self.logger.experiment
            tensorboard.add_scalar("train_photometric", photo_error, global_step = self.global_step)
            tensorboard.add_scalar("train_photometric_occluded", photo_error_occluded, global_step = self.global_step)
            tensorboard.add_scalar("train_reconst", reconst_error, global_step = self.global_step)
            tensorboard.add_scalar("train_smoothness", smoothness_loss, global_step = self.global_step)
            if bce_loss is not None:
                tensorboard.add_scalar("train_bce_loss", bce_loss, global_step = self.global_step)
        return loss
    def on_after_backward(self):
        # example to inspect gradient information in tensorboard
        if self.global_step % self.log_every_n_steps == 0:  # don't make the tf file huge
            for tag, param in self.occ_pred.named_parameters():
                self.logger.experiment.add_histogram( tag, param.grad, global_step=self.global_step)
    def training_epoch_end(self, outputs): 
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        tensorboard = self.logger.experiment
        tensorboard.add_scalars("losses", {"train_loss": avg_loss}, global_step = self.current_epoch)
    
    
    def validation_step(self, batch, batch_idx):
        #losses = self.general_step(batch, batch_idx, 'val')
        bce_loss = None
        occ = None
        if not isinstance(batch, (list, tuple)):
            imgs = batch
        elif len(batch) == 2:
            imgs, flow = batch
        elif len(batch) == 3:
            imgs, flow, occ = batch
        else:
            raise ValueError('Not supported dataset')
        img1, img2 = imgs[:, 0:3, :, :], imgs[:, 3:6, :, :]
        # occlusion prediction
        occ_pred_soft = self.occ_pred(imgs)
        occ_pred = (torch.where(occ_pred_soft > 0.5, 1.0, 0.0) - occ_pred_soft).detach() + occ_pred_soft
        # warp image use ground truth optical flow
        img_warped = self.warp(img2, flow)
        masked_imgs =  img_warped * (1 - occ_pred)
        # get completed image
        if self.inpainting_stage =='simple': 
            img_completed = self.inpainting(img_warped, occ_pred_soft)
        elif self.inpainting_stage =='gated' or self.inpainting_stage == 'gated_org': 
            _,img_completed = self.inpainting(img_warped, occ_pred_soft)
        
        # calculate the reconstruction error
        #photometric_error = charbonnier_loss((img_warped - img1) * (1 - occ_pred), reduction=False).sum() / (3*(1 - occ_pred).sum() + 1e-16)
        #reconst_error = charbonnier_loss(torch.abs(img1 - img_completed) * occ_pred, reduction=False).sum() / (3*occ_pred.sum() + 1e-16)
        photo_error = photometric_error(img_warped * (1 - occ_pred_soft), img1 * (1 - occ_pred_soft))
        photo_error_occluded = photometric_error(img_warped * occ_pred_soft, img1 *occ_pred_soft)
        
        #add smoothness loss to occlusion map 
        smoothness_loss = first_order_smoothness_loss(img_warped, occ_pred_soft)

        #reconst_error = photometric_error(img_completed * occ_pred, img1 * occ_pred)
        reconstructed_img = occ_pred_soft*img_completed + (1-occ_pred_soft)*img1
        reconst_error = photometric_error(occ_pred_soft*reconstructed_img, occ_pred_soft*img1)
        # calculate BCE error if occ is available
        if occ is not None:
            bce_loss = F.binary_cross_entropy(occ_pred_soft, occ)
        loss = photo_error + self.reconst_weight * reconst_error + self.smooth1_weight * smoothness_loss

        if batch_idx == 0:
            if self.global_rank == 0:
                tensorboard = self.logger.experiment
                tensorboard.add_scalar("val_photometric", photo_error, global_step = self.global_step)
                tensorboard.add_scalar("val_photometric_occluded", photo_error_occluded, global_step = self.global_step)
                tensorboard.add_scalar("val_reconst", reconst_error, global_step = self.global_step)
                tensorboard.add_scalar("val_smoothness", smoothness_loss, global_step = self.global_step)
                if bce_loss is not None:
                    tensorboard.add_scalar("val_bce_loss", bce_loss, global_step = self.global_step)
                if self.current_epoch % self.log_image_every_epoch == 0: 
                    val_save_dir = os.path.join(self.result_dir, "val_{}".format(self.current_epoch))
                    val_save_real_dir = os.path.join(val_save_dir, "ground truth")
                    val_save_pred_dir = os.path.join(val_save_dir, "predicted")
                    if not os.path.exists(val_save_real_dir):
                        os.makedirs(val_save_real_dir)
                        os.makedirs(val_save_pred_dir)

                    saved_occs =img2photo(torch.cat([occ, occ_pred, occ_pred_soft], dim=2), rgb = False)
                    saved_images =img2photo(torch.cat([img1, img2,img_warped, masked_imgs, img_completed], dim=2))
                    h, w = saved_occs.shape[1]//3, saved_occs.shape[2]
                    j= 0
                    n_display_images = self.n_display_images if self.batch_size > self.n_display_images else self.batch_size
                    for val_occ in saved_occs:
                        real_occ = val_occ[:h, :,:].squeeze()
                        pred_occ = val_occ[h:2*h,:,:] .squeeze()
                        real_occ = Image.fromarray(real_occ.astype(np.uint8)*255, 'L')
                        pred_occ = Image.fromarray(pred_occ.astype(np.uint8)*255, 'L')
                        real_occ.save(os.path.join(val_save_real_dir, "{}.png".format(j)))
                        pred_occ.save(os.path.join(val_save_pred_dir, "{}.png".format(j)))
                        j += 1
                        if j == n_display_images: 
                            break
                    tensorboard = self.logger.experiment
                    tensorboard.add_images('occlusion mask', (saved_occs[:n_display_images]*255).astype(np.uint8),self.current_epoch, dataformats = 'NHWC')
                    tensorboard.add_images('warped and inpainted image', saved_images[:n_display_images].astype(np.uint8),self.current_epoch, dataformats = 'NHWC')
        return loss
    
    def validation_epoch_end(self, outputs): 
        avg_loss = torch.stack([x for x in outputs]).mean()
        tensorboard = self.logger.experiment
        tensorboard.add_scalars("losses", {"val_loss": avg_loss}, global_step = self.current_epoch)
        self.log('monitored_loss', avg_loss, prog_bar= True, logger = True)

    def test_step(self, batch, batch_idx):
        losses = self.general_step(batch, batch_idx, 'test')
        bce_loss = None
        if len(losses) == 3:
            photo_error, reconst_error, smoothness_loss, photo_error_occluded = losses[0], losses[1], losses[2], losses[3]
        else:
            photo_error, reconst_error, smoothness_loss, bce_loss, photo_error_occluded = losses[0], losses[1], losses[2], losses[3], losses[4]
        loss = photo_error + self.reconst_weight * reconst_error + self.smooth1_weight * smoothness_loss
        if batch_idx == 0:
            tensorboard = self.logger.experiment
            tensorboard.add_scalar("test_photometric", photo_error, global_step = self.global_step)
            tensorboard.add_scalar("test_photometric_occluded", photo_error_occluded, global_step = self.global_step)
            tensorboard.add_scalar("test_reconst", reconst_error, global_step = self.global_step)
            tensorboard.add_scalar("test_smoothness", smoothness_loss, global_step = self.global_step)
            if bce_loss is not None:
                tensorboard.add_scalar("test_bce_loss", bce_loss, global_step = self.global_step)
        return loss

    def test_epoch_end(self, outputs): 
        avg_loss = torch.stack([x for x in outputs]).mean()
        tensorboard = self.logger.experiment
        tensorboard.add_scalars("losses", {"test_loss": avg_loss}, global_step = self.current_epoch)
    
    def configure_optimizers(self):
        return Adam(self.parameters(), self.lr)
