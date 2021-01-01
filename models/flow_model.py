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
from models.networks.pwc_net import PWCNet

class FlowModel(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.lr = hparams['learning_rate']
        model = self.hparams.get('model', 'simple')
        self.model = model
        displacement = self.hparams.get('displacement', 4)
        if model == 'simple':
            self.flow_pred = SimpleFlowNet()
        elif model == 'pwc':
            self.flow_pred = FlowNetCV(displacement=displacement)
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
        if self.model == 'pwc':
            out, _ = self.flow_pred(x)
        else:
            out = self.flow_pred(x)
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
        if not isinstance(batch, (list, tuple)):
            raise ValueError('Not supported dataset')
        elif len(batch) == 2:
            imgs, flow = batch
        elif len(batch) == 3:
            imgs, flow, _ = batch
        else:
            raise ValueError('Not supported dataset')
        flow_pred = self(imgs)
        
        loss = F.mse_loss(flow_pred, flow)
        
        return loss
    
    
    def training_step(self, batch, batch_idx):
        loss = self.general_step(batch, batch_idx, 'train')
        return loss
    def training_epoch_end(self, outputs): 
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        tensorboard = self.logger.experiment
        tensorboard.add_scalars("losses", {"train_loss": avg_loss}, global_step = self.current_epoch)
    
    
    def validation_step(self, batch, batch_idx):
        loss = self.general_step(batch, batch_idx, 'val')
        
        return loss
    def validation_epoch_end(self, outputs): 
        avg_loss = torch.stack([x for x in outputs]).mean()
        tensorboard = self.logger.experiment
        tensorboard.add_scalars("losses", {"val_loss": avg_loss}, global_step = self.current_epoch)
        self.log('monitored_loss', avg_loss, prog_bar= True, logger = True)
    
    def test_step(self, batch, batch_idx):
        loss = self.general_step(batch, batch_idx, 'test')
        return loss
    def test_epoch_end(self, outputs): 
        avg_loss = torch.stack([x for x in outputs]).mean()
        tensorboard = self.logger.experiment
        tensorboard.add_scalars("losses", {"test_loss": avg_loss}, global_step = self.current_epoch)
    
    def configure_optimizers(self):
        return Adam(self.parameters(), self.lr)
        