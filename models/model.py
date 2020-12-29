"""Test optical flow prediction network with ground truth optical flow"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

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

from Forward_Warp import forward_warp
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
    loss_dx = (flow_dx_norm) * torch.exp(-alpha * img_dx_norm)
    loss_dy = (flow_dy_norm) * torch.exp(-alpha * img_dy_norm)
    loss = charbonnier_loss(loss_dx, reduction = reduction) + charbonnier_loss(loss_dy, reduction = reduction)
    
    return loss
def img2photo(imgs):
    return ((imgs+1)*127.5).transpose(1,2).transpose(2,3).detach().cpu().numpy()

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
        self.log_every_n_steps = hparams.get('log_every_n_steps', 20)
        self.occ_aware = hparams.get('occ_aware', False)
        
        model = self.hparams.get('model', 'simple')
        if model == 'simple':
            self.flow_pred = SimpleFlowNet()
        elif model == 'pwc':
            self.flow_pred = PWCNet()
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
        
        output = F.grid_sample(img, vgrid, align_corners=False)
        
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
        flow_pred = self(imgs)
        img_warped = self.warp(img2, flow_pred)
        
        # calculate photometric error
        photometric_error = charbonnier_loss(img_warped - img1)
        smoothness_term = edge_aware_smoothness_loss(img1, flow_pred)
        second_order_error = second_order_photometric_error(img_warped, img1)
        #calculate difference between predicted flow and ground truth flow
        mse_loss = torch.nn.MSELoss()
        flow_error = mse_loss(flow_pred, flow)
        return photometric_error, smoothness_term, second_order_error, flow_error
    
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
        #calculate difference between predicted flow and ground truth flow
        mse_loss = torch.nn.MSELoss()
        flow_error = mse_loss(flow_pred, flow)
        return photometric_error, smoothness_term, second_order_error, flow_error
    
    def general_step_occ_aware(self, batch, batch_idx, mode):
        if len(batch) == 2:
            imgs, flow = batch
        elif len(batch) == 3:
            imgs, flow, occ = batch
        else:
            raise ValueError('Not supported dataset')
        img1, img2 = imgs[:, 0:3, :, :], imgs[:, 3:6, :, :]
        # flow prediction
        flow_pred = self(imgs)
        img_warped = self.warp(img2, flow_pred)
        # range map calculation
        with torch.no_grad():
            # backward flow prediction
            back_flow_pred = self(torch.cat((img2, img1), dim=1))
            # calculate range map
            range_map = self.compute_range_map(back_flow_pred)
            # compute occlusion mask
            # 0: non-occluded, 1: occluded
            occ_pred = 1. - torch.clamp(range_map, min=0.0, max=1.0)
            
        # calculate photometric error
        photometric_error = (charbonnier_loss(img_warped - img1, reduction=False) * (1 - occ_pred)).sum() / (3*(1 - occ_pred).sum() + 1e-16)
        second_order_error = (second_order_photometric_error(img_warped, img1, reduction=False) * (1 - occ_pred)).sum() / (3*(1 - occ_pred).sum() + 1e-16)
        smoothness_term = edge_aware_smoothness_loss(img1, flow_pred)
        #calculate difference between predicted flow and ground truth flow
        mse_loss = torch.nn.MSELoss()
        flow_error = mse_loss(flow_pred, flow)
        # calculate the photometric error in occluded region
        photometric_error_occ =  (charbonnier_loss(img_warped - img1, reduction=False) * occ_pred).sum() / (3 * occ_pred.sum() + 1e-16)
        if occ is not None:
            occ_error = F.binary_cross_entropy(occ, occ_pred)
            return photometric_error, smoothness_term, second_order_error, flow_error, photometric_error_occ, occ_error
        return photometric_error, smoothness_term, second_order_error, flow_error, photometric_error_occ
    
    def training_step(self, batch, batch_idx):
        if not self.occ_aware:
            if not self.with_occ:
                photometric_error, smoothness_term, second_order_error, flow_error = self.general_step(batch, batch_idx, 'train')
            else:
                photometric_error, smoothness_term, second_order_error, flow_error = self.general_step_occ(batch, batch_idx, 'train')
        else:
            losses = self.general_step_occ_aware(batch, batch_idx, 'train')
            if len(losses) == 5:
                photometric_error, smoothness_term, second_order_error, flow_error, photometric_error_occ = losses[0], losses[1], losses[2], losses[3], losses[4]
            else:
                photometric_error, smoothness_term, second_order_error, flow_error, photometric_error_occ, occ_error = losses[0], losses[1], losses[2], losses[3], losses[4], losses[5]
                
        loss = photometric_error + self.smoothness_weight * smoothness_term + self.second_order_weight * second_order_error
        
        if self.global_step % self.log_every_n_steps == 0: 
            tensorboard = self.logger.experiment
            tensorboard.add_scalar("train_photometric", photometric_error, global_step = self.global_step)
            tensorboard.add_scalar("train_smoothness", smoothness_term, global_step = self.global_step)
            tensorboard.add_scalar("train_second_order", second_order_error, global_step = self.global_step)
            tensorboard.add_scalar("train_flow_error", flow_error, global_step = self.global_step)
            if occ_error is not None:
                tensorboard.add_scalar("train_occ_error", occ_error, global_step = self.global_step)
            if photometric_error_occ is not None:
                tensorboard.add_scalar("train_photometric_occ", photometric_error_occ, global_step = self.global_step)
        return loss
    
    def training_epoch_end(self, outputs): 
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        tensorboard = self.logger.experiment
        tensorboard.add_scalars("losses", {"train_loss": avg_loss}, global_step = self.current_epoch)
    
    def validation_step(self, batch, batch_idx):
        if not self.occ_aware:
            if not self.with_occ:
                photometric_error, smoothness_term, second_order_error, flow_error = self.general_step(batch, batch_idx, 'val')
            else:
                photometric_error, smoothness_term, second_order_error, flow_error = self.general_step_occ(batch, batch_idx, 'val')
        else:
            losses = self.general_step_occ_aware(batch, batch_idx, 'train')
            if len(losses) == 5:
                photometric_error, smoothness_term, second_order_error, flow_error, photometric_error_occ = losses[0], losses[1], losses[2], losses[3], losses[4]
            else:
                photometric_error, smoothness_term, second_order_error, flow_error, photometric_error_occ, occ_error = losses[0], losses[1], losses[2], losses[3], losses[4], losses[5]
                
        loss = photometric_error + self.smoothness_weight * smoothness_term + self.second_order_weight * second_order_error

        if batch_idx == 0: 
            tensorboard = self.logger.experiment
            tensorboard.add_scalar("val_photometric", photometric_error, global_step = self.global_step)
            tensorboard.add_scalar("val_smoothness", smoothness_term, global_step = self.global_step)
            tensorboard.add_scalar("val_second_order", second_order_error, global_step = self.global_step)
            tensorboard.add_scalar("val_flow_error", flow_error, global_step = self.global_step)
            if occ_error is not None:
                tensorboard.add_scalar("val_occ_error", occ_error, global_step = self.global_step)
            if photometric_error_occ is not None:
                tensorboard.add_scalar("val_photometric_occ", photometric_error_occ, global_step = self.global_step)
        return loss

    def validation_epoch_end(self, outputs): 
        avg_loss = torch.stack([x for x in outputs]).mean()
        tensorboard = self.logger.experiment
        tensorboard.add_scalars("losses", {"val_loss": avg_loss}, global_step = self.current_epoch)
        self.log('monitored_loss', avg_loss, prog_bar= True, logger = True)    
    
    def test_step(self, batch, batch_idx):
        if not self.occ_aware:
            if not self.with_occ:
                photometric_error, smoothness_term, second_order_error, flow_error = self.general_step(batch, batch_idx, 'test')
            else:
                photometric_error, smoothness_term, second_order_error, flow_error = self.general_step_occ(batch, batch_idx, 'test')
        else:
            losses = self.general_step_occ_aware(batch, batch_idx, 'train')
            if len(losses) == 5:
                photometric_error, smoothness_term, second_order_error, flow_error, photometric_error_occ = losses[0], losses[1], losses[2], losses[3], losses[4]
            else:
                photometric_error, smoothness_term, second_order_error, flow_error, photometric_error_occ, occ_error = losses[0], losses[1], losses[2], losses[3], losses[4], losses[5]
                
        loss = photometric_error + self.smoothness_weight * smoothness_term + self.second_order_weight * second_order_error
        
        if batch_idx == 0:
            tensorboard = self.logger.experiment
            tensorboard.add_scalar("test_photometric", photometric_error, global_step = self.global_step)
            tensorboard.add_scalar("test_smoothness", smoothness_term, global_step = self.global_step)
            tensorboard.add_scalar("test_second_order", second_order_error, global_step = self.global_step)
            tensorboard.add_scalar("test_flow_error", flow_error, global_step = self.global_step)
            if occ_error is not None:
                tensorboard.add_scalar("test_occ_error", occ_error, global_step = self.global_step)
            if photometric_error_occ is not None:
                tensorboard.add_scalar("val_photometric_occ", photometric_error_occ, global_step = self.global_step)
        return loss
    def test_epoch_end(self, outputs): 
        avg_loss = torch.stack([x for x in outputs]).mean()
        tensorboard = self.logger.experiment
        tensorboard.add_scalars("losses", {"test_loss": avg_loss}, global_step = self.current_epoch)
    
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
        self.lr = hparams.get('learning_rate',1e-4)
        self.second_order_weight = hparams.get('second_order_weight', 0.0)
        model = hparams.get('model', 'simple')
        self.model = InpaintingNet()
        self.log_every_n_steps = hparams.get('log_every_n_steps', 20)
        self.batch_size = hparams.get('batch_size', 16)
        self.n_display_images = hparams.get('n_display_images', 1)
        self.result_dir = hparams.get('result_dir','')
        self.log_image_every_epoch = hparams.get('log_image_every_epoch',10)
        print('result dir inside inpainting model is {}'.format(self.result_dir))
    @property
    def is_cuda(self):
        return next(self.parameters()).is_cuda
    
    
    def save_state_dict(self, path):
        torch.save(self.state_dict(), path)
        
    def general_step(self, batch, batch_idx, mode):
        imgs, complete_imgs, occlusion_map = batch
        # inpainting
        pred_imgs = self.model(complete_imgs, occlusion_map)
        #reconst_error = (charbonnier_loss(pred_imgs - complete_imgs, reduction=False) * occlusion_map).sum() / (3*occlusion_map.sum() + 1e-16)
        second_order_error = (second_order_photometric_error(pred_imgs, complete_imgs, reduction=False) * occlusion_map).sum() / (3*occlusion_map.sum() + 1e-16)
        Recon_Loss = ReconLoss(1.0,1.0,1.0,1.0)
        recon_loss, rhole, runhole = Recon_Loss(complete_imgs, pred_imgs, occlusion_map)
        return recon_loss,rhole, runhole, second_order_error
    
    
    def training_step(self, batch, batch_idx):
        recon_loss,rhole, runhole, second_order_error = self.general_step(batch, batch_idx, 'train')
        #loss = reconst_error + self.second_order_weight * second_order_error

        if self.global_step % self.log_every_n_steps == 0: 
            tensorboard = self.logger.experiment
            tensorboard.add_scalars("second_order",{"train_loss": second_order_error}, global_step = self.global_step)
        return {'loss': recon_loss, 'occluded': rhole, 'non_occluded': runhole}
    def training_epoch_end(self, outputs): 
        avg_occluded = torch.stack([x['occluded'] for x in outputs]).mean()
        avg_non_occluded = torch.stack([x['non_occluded'] for x in outputs]).mean()
        avg_recon = torch.stack([x['loss'] for x in outputs]).mean()
        tensorboard = self.logger.experiment
        tensorboard.add_scalars("occluded_epoch_loss", {"train_loss": avg_occluded}, global_step = self.current_epoch)
        tensorboard.add_scalars("non_occluded_epoch_loss", {"train_loss": avg_non_occluded}, global_step = self.current_epoch)
        tensorboard.add_scalars("recon_epoch_loss", {"train_loss": avg_recon}, global_step = self.current_epoch)
    
    def validation_step(self, batch, batch_idx):
        _, imgs, masks = batch
        recon_imgs = self.model(imgs, masks)
        second_order_error = (second_order_photometric_error(recon_imgs,imgs, reduction=False) * masks).sum() / (3*masks.sum() + 1e-16)
        Recon_Loss = ReconLoss(1.0,1.0,1.0,1.0)
        recon_loss, rhole, runhole = Recon_Loss(imgs, recon_imgs, masks)
        complete_imgs = recon_imgs * masks + imgs * (1 - masks)
        #return recon_loss,rhole, runhole, second_order_error
        
        if batch_idx == 0: 
            tensorboard = self.logger.experiment
            tensorboard.add_scalars("second_order", {"val_loss": second_order_error}, global_step = self.global_step)

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
                
        return  (rhole, runhole, recon_loss)
    def validation_epoch_end(self,outputs): 
        avg_occluded = torch.stack([x[0] for x in outputs]).mean()
        avg_non_occluded = torch.stack([x[1] for x in outputs]).mean()
        avg_recon = torch.stack([x[2] for x in outputs]).mean()
        tensorboard = self.logger.experiment
        tensorboard.add_scalars("occluded_epoch_loss", {"val_loss": avg_occluded}, global_step = self.current_epoch)
        tensorboard.add_scalars("non_occluded_epoch_loss", {"val_loss": avg_non_occluded}, global_step = self.current_epoch)
        tensorboard.add_scalars("recon_epoch_loss", {"val_loss": avg_recon}, global_step = self.current_epoch)
        self.log('monitored_loss', avg_recon, prog_bar= True, logger = True)

    def test_step(self, batch, batch_idx):
        recon_loss, rhole, runhole, second_order_error = self.general_step(batch, batch_idx, 'test')
        #loss = reconst_error + self.second_order_weight * second_order_error
        
        if batch_idx == 0: 
            tensorboard = self.logger.experiment
            tensorboard.add_scalars("second_order", {"test_loss": second_order_error}, global_step = self.global_step)
        return  (rhole, runhole, recon_loss)
    def test_epoch_end(self, outputs): 
        avg_occluded = torch.stack([x[0] for x in outputs]).mean()
        avg_non_occluded = torch.stack([x[1] for x in outputs]).mean()
        avg_recon = torch.stack([x[2] for x in outputs]).mean()
        tensorboard = self.logger.experiment
        tensorboard.add_scalars("occluded_epoch_loss", {"test_loss": avg_occluded}, global_step = self.current_epoch)
        tensorboard.add_scalars("non_occluded_epoch_loss", {"test_loss": avg_non_occluded}, global_step = self.current_epoch)
        tensorboard.add_scalars("recon_epoch_loss", {"test_loss": avg_recon}, global_step = self.current_epoch)
    
    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), self.lr)
        return optimizer
#         scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=20)
#         return {
#             'optimizer': optimizer,
#             'lr_scheduler': scheduler,
#             'monitor': 'val_reconst'
#         }

class InpaintingGConvModel(pl.LightningModule): 
    def __init__(self, hparams): 
        super().__init__()
        self.hparams = hparams
        self.lr = hparams.get('learning_rate',1e-4)
        self.decay = hparams.get('decay',0.0)
        self.org = hparams.get('org', False)
        self.img_size = hparams.get('image_size', (64, 128))
        self.batch_size = hparams.get('batch_size', 16)
        self.n_display_images = hparams.get('n_display_images', 1)
        self.result_dir = hparams.get('result_dir','')
        self.log_image_every_epoch = hparams.get('log_image_every_epoch',10)
        print('result dir inside inpainting model is {}'.format(self.result_dir))
        if self.org:
            self.generator = InpaintSANetOrg(img_size=self.img_size)
            self.discriminator = InpaintSADiscriminatorOrg(img_size=self.img_size)
        else:
            self.generator = InpaintSANet(img_size=self.img_size)
            self.discriminator = InpaintSADiscriminator(img_size=self.img_size)
        self.log_every_n_steps = hparams.get('log_every_n_steps',20)
        
    def forward(self, inputs): 
        return self.generator(inputs)
    def training_step(self, batch, batch_idx, optimizer_idx): 
        _, imgs, masks = batch 
        (optD, optG) = self.optimizers()
        #train discriminator
        coarse_imgs, recon_imgs = self.generator(imgs,masks)
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
        Recon_Loss = ReconLoss(1.0,1.0,1.0,1.0)
        g_loss = GANLoss(pred_neg)
        r_loss, r_occluded, r_non_occluded = Recon_Loss(imgs, recon_imgs, masks, coarse_imgs)
        whole_loss = g_loss + r_loss
        self.manual_backward(whole_loss, optG)
        optG.step()
        optG.zero_grad()
        optD.zero_grad()
        if self.global_step % self.log_every_n_steps == 0: 
            tensorboard = self.logger.experiment
            tensorboard.add_scalars("whole_loss",{"train_loss": whole_loss} , global_step = self.global_step)
            tensorboard.add_scalars("recon_loss", {"train_loss": r_loss}, global_step = self.global_step)
            tensorboard.add_scalars("gan_loss", {"train_loss":g_loss}, global_step = self.global_step)
            tensorboard.add_scalars("discriminator_loss",{ "train_loss":d_loss}, global_step = self.global_step)
        return {'loss': r_loss, 'occluded': r_occluded, 'non_occluded': r_non_occluded}
    def training_epoch_end(self, outputs): 
        avg_occluded = torch.stack([x['occluded'] for x in outputs]).mean()
        avg_non_occluded = torch.stack([x['non_occluded'] for x in outputs]).mean()
        avg_recon = torch.stack([x['loss'] for x in outputs]).mean()
        tensorboard = self.logger.experiment
        tensorboard.add_scalars("occluded_epoch_loss", {"train_loss": avg_occluded}, global_step = self.current_epoch)
        tensorboard.add_scalars("non_occluded_epoch_loss", {"train_loss": avg_non_occluded}, global_step = self.current_epoch)
        tensorboard.add_scalars("recon_epoch_loss", {"train_loss": avg_recon}, global_step = self.current_epoch)

    def validation_step(self, batch, batch_idx): 
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
        Recon_Loss = ReconLoss(1.0,1.0,1.0,1.0)
        g_loss = GANLoss(pred_neg)
        r_loss, r_occluded, r_non_occluded = Recon_Loss(imgs, recon_imgs, masks, coarse_imgs)
        whole_loss = g_loss + r_loss
       
        Dloss = SNDisLoss()
        d_loss = Dloss(pred_pos, pred_neg)
       
        if batch_idx == 0:
            tensorboard = self.logger.experiment
            tensorboard.add_scalars("whole_loss",{"val_loss": whole_loss} , global_step = self.global_step)
            tensorboard.add_scalars("recon_loss", {"val_loss": r_loss}, global_step = self.global_step)
            tensorboard.add_scalars("gan_loss", {"val_loss":g_loss}, global_step = self.global_step)
            tensorboard.add_scalars("discriminator_loss",{ "val_loss":d_loss}, global_step = self.global_step)
            if self.current_epoch % self.log_image_every_epoch == 0: 

                val_save_dir = os.path.join(self.result_dir, "val_{}".format(self.current_epoch))
                val_save_real_dir = os.path.join(val_save_dir, "real")
                val_save_gen_dir = os.path.join(val_save_dir, "gen")
                if not os.path.exists(val_save_real_dir):
                    os.makedirs(val_save_real_dir)
                    os.makedirs(val_save_gen_dir)

                saved_images =img2photo(torch.cat([ imgs * (1 - masks), coarse_imgs, recon_imgs, imgs, complete_imgs], dim=2))
                h, w = saved_images.shape[1]//5, saved_images.shape[2]
                j= 0
                n_display_images = self.n_display_images if self.batch_size > self.n_display_images else self.batch_size
                for val_img in saved_images:
                    real_img = val_img[(3*h):(4*h), :,:]
                    gen_img = val_img[(4*h):,:,:] 
                    real_img = Image.fromarray(real_img.astype(np.uint8))
                    gen_img = Image.fromarray(gen_img.astype(np.uint8))
                    real_img.save(os.path.join(val_save_real_dir, "{}.png".format(j)))
                    gen_img.save(os.path.join(val_save_gen_dir, "{}.png".format(j)))
                    j += 1
                    if j == n_display_images: 
                        break
                tensorboard = self.logger.experiment
                tensorboard.add_images('val/imgs', saved_images[:n_display_images].astype(np.uint8), self.current_epoch, dataformats = 'NHWC')

        return (r_occluded, r_non_occluded, r_loss)
    def validation_epoch_end(self, outputs): 
        avg_occluded = torch.stack([x[0] for x in outputs]).mean()
        avg_non_occluded = torch.stack([x[1] for x in outputs]).mean()
        avg_recon = torch.stack([x[2] for x in outputs]).mean()
        tensorboard = self.logger.experiment
        tensorboard.add_scalars("occluded_epoch_loss", {"val_loss": avg_occluded}, global_step = self.current_epoch)
        tensorboard.add_scalars("non_occluded_epoch_loss", {"val_loss": avg_non_occluded}, global_step = self.current_epoch)
        tensorboard.add_scalars("recon_epoch_loss", {"val_loss": avg_recon}, global_step = self.current_epoch)
        self.log('monitored_loss', avg_recon, prog_bar= True, logger = True)

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
        Recon_Loss = ReconLoss(1.0,1.0,1.0,1.0)
        g_loss = GANLoss(pred_neg)
        r_loss, r_occluded, r_non_occluded  = Recon_Loss(imgs, recon_imgs, masks, coarse_imgs)
        whole_loss = g_loss + r_loss
       
        Dloss = SNDisLoss()
        d_loss = Dloss(pred_pos, pred_neg)
       
        if batch_idx == 0:
            tensorboard = self.logger.experiment
            tensorboard.add_scalars("whole_loss",{"test_loss": whole_loss} , global_step = self.global_step)
            tensorboard.add_scalars("recon_loss", {"test_loss": r_loss}, global_step = self.global_step)
            tensorboard.add_scalars("gan_loss", {"test_loss":g_loss}, global_step = self.global_step)
            tensorboard.add_scalars("discriminator_loss",{ "test_loss":d_loss}, global_step = self.global_step)
        return  (r_occluded, r_non_occluded, r_loss)
    def test_epoch_end(self, outputs): 
        avg_occluded = torch.stack([x[0] for x in outputs]).mean()
        avg_non_occluded = torch.stack([x[1] for x in outputs]).mean()
        avg_recon = torch.stack([x[2] for x in outputs]).mean()
        tensorboard = self.logger.experiment
        tensorboard.add_scalars("occluded_epoch_loss", {"test_loss": avg_occluded}, global_step = self.current_epoch)
        tensorboard.add_scalars("non_occluded_epoch_loss", {"test_loss": avg_non_occluded}, global_step = self.current_epoch)
        tensorboard.add_scalars("recon_epoch_loss", {"test_loss": avg_recon}, global_step = self.current_epoch)
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
        smoothness_term = (edge_aware_smoothness_loss(img1, flow_pred, reduction=False) * (1 - occ_pred)).sum() / (3*(1 - occ_pred).sum() + 1e-16)
        photometric_error = charbonnier_loss((img_warped - img1) * (1 - occ_pred), reduction=True)
        reconst_error = charbonnier_loss((img1 - img_completed) * occ_pred, reduction=True)
        # calculate BCE error if occ is available
        if occ is not None:
            bce_loss = F.binary_cross_entropy(occ_pred, occ)
            return photometric_error, reconst_error, smoothness_term, bce_loss
        return photometric_error, reconst_error, smoothness_term
    
    
    def training_step(self, batch, batch_idx):
        losses = self.general_step(batch, batch_idx, 'train')
        if len(losses) == 3:
            photometric_error, reconst_error, smoothness_term = losses[0], losses[1], losses[2]
        else:
            photometric_error, reconst_error, smoothness_term, bce_loss = losses[0], losses[1], losses[2], losses[3]
        loss = photometric_error + self.reconst_weight * reconst_error + self.smoothness_weight * smoothness_term

        if self.global_step % self.log_every_n_steps == 0: 
            tensorboard = self.logger.experiment
            tensorboard.add_scalar("train_photometric", photometric_error, global_step = self.global_step)
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
            photometric_error, reconst_error, smoothness_term = losses[0], losses[1], losses[2]
        else:
            photometric_error, reconst_error, smoothness_term, bce_loss = losses[0], losses[1], losses[2], losses[3]
        loss = photometric_error + self.reconst_weight * reconst_error + self.smoothness_weight * smoothness_term
        if batch_idx == 0: 
            tensorboard = self.logger.experiment
            tensorboard.add_scalar("val_photometric", photometric_error, global_step = self.global_step)
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
            photometric_error, reconst_error, smoothness_term = losses[0], losses[1], losses[2]
        else:
            photometric_error, reconst_error, smoothness_term, bce_loss = losses[0], losses[1], losses[2], losses[3]
        loss = photometric_error + self.reconst_weight * reconst_error + self.smoothness_weight * smoothness_term
        if batch_idx == 0:
            tensorboard = self.logger.experiment
            tensorboard.add_scalar("test_photometric", photometric_error, global_step = self.global_step)
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
        self.inpainting = InpaintingNet()
        inpainting_root = hparams.get('inpainting_root', None)
        if inpainting_root:
            self.inpainting = InpaintingStageModel.load_from_checkpoint(inpainting_root).model
        
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
            imgs, flow = batch
        elif len(batch) == 3:
            imgs, flow, occ = batch
        else:
            raise ValueError('Not supported dataset')
        img1, img2 = imgs[:, 0:3, :, :], imgs[:, 3:6, :, :]
        # occlusion prediction
        occ_pred = self.occ_pred(imgs)
        # warp image use ground truth optical flow
        img_warped = self.warp(img2, flow)
        
        # get occluded image
        img_occluded = img_warped * (1 - occ_pred) # 1: occluded 0: non-occluded

        # get completed image
        img_completed = self.inpainting(img_occluded)
        
        # calculate the reconstruction error
        #photometric_error = charbonnier_loss((img_warped - img1) * (1 - occ_pred), reduction=False).sum() / (3*(1 - occ_pred).sum() + 1e-16)
        #reconst_error = charbonnier_loss(torch.abs(img1 - img_completed) * occ_pred, reduction=False).sum() / (3*occ_pred.sum() + 1e-16)
        photometric_error = charbonnier_loss((img_warped - img1) * (1 - occ_pred), reduction=True) 
        reconst_error = charbonnier_loss((img1 - img_completed) * occ_pred, reduction=True)
        # calculate BCE error if occ is available
        if occ is not None:
            bce_loss = F.binary_cross_entropy(occ_pred, occ)
            return photometric_error, reconst_error, bce_loss
        return photometric_error, reconst_error
    
    
    def training_step(self, batch, batch_idx):
        losses = self.general_step(batch, batch_idx, 'train')
        if len(losses) == 2:
            photometric_error, reconst_error = losses[0], losses[1]
        else:
            photometric_error, reconst_error, bce_loss = losses[0], losses[1], losses[2]
        loss = photometric_error + self.reconst_weight * reconst_error
        if self.global_step % self.log_every_n_steps == 0: 
            tensorboard = self.logger.experiment
            tensorboard.add_scalar("train_photometric", photometric_error, global_step = self.global_step)
            tensorboard.add_scalar("train_reconst", reconst_error, global_step = self.global_step)
            if bce_loss is not None:
                tensorboard.add_scalar("train_bce_loss", bce_loss, global_step = self.global_step)
        return loss

    def training_epoch_end(self, outputs): 
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        tensorboard = self.logger.experiment
        tensorboard.add_scalars("losses", {"train_loss": avg_loss}, global_step = self.current_epoch)
    
    
    def validation_step(self, batch, batch_idx):
        losses = self.general_step(batch, batch_idx, 'val')
        if len(losses) == 2:
            photometric_error, reconst_error = losses[0], losses[1]
        else:
            photometric_error, reconst_error, bce_loss = losses[0], losses[1], losses[2]
        loss = photometric_error + self.reconst_weight * reconst_error
        if batch_idx == 0:
            tensorboard = self.logger.experiment
            tensorboard.add_scalar("val_photometric", photometric_error, global_step = self.global_step)
            tensorboard.add_scalar("val_reconst", reconst_error, global_step = self.global_step)
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
        if len(losses) == 2:
            photometric_error, reconst_error = losses[0], losses[1]
        else:
            photometric_error, reconst_error, bce_loss = losses[0], losses[1], losses[2]
        loss = photometric_error + self.reconst_weight * reconst_error
        if batch_idx == 0:
            tensorboard = self.logger.experiment
            tensorboard.add_scalar("test_photometric", photometric_error, global_step = self.global_step)
            tensorboard.add_scalar("test_reconst", reconst_error, global_step = self.global_step)
            if bce_loss is not None:
                tensorboard.add_scalar("test_bce_loss", bce_loss, global_step = self.global_step)
        return loss

    def test_epoch_end(self, outputs): 
        avg_loss = torch.stack([x for x in outputs]).mean()
        tensorboard = self.logger.experiment
        tensorboard.add_scalars("losses", {"test_loss": avg_loss}, global_step = self.current_epoch)
    
    def configure_optimizers(self):
        return Adam(self.parameters(), self.lr)
