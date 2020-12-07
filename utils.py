import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from models.data.utils.flow_utils import flow2img, evaluate_flow

def warp(img, flow, is_mask=True):
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
        if is_mask:
            mask = torch.ones(img.size())
            if img.is_cuda:
                mask = mask.cuda()
            mask = F.grid_sample(mask, vgrid, align_corners=False)

            mask[mask <0.9999] = 0
            mask[mask >0] = 1
            output = output * mask
        return output

def visualize_inpainting(img, complete_img, predict_img, occlusion_map):
    
    img = img.detach().cpu().numpy().transpose(1, 2, 0)
    complete_img = complete_img.detach().cpu().numpy().transpose(1, 2, 0)
    predict_img = predict_img.detach().cpu().numpy().transpose(1, 2, 0)
    occlusion_map = occlusion_map.detach().cpu().numpy().transpose(1, 2, 0)
    full_img = (1 - occlusion_map) * complete_img + (occlusion_map) * predict_img
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 2, 1)
    plt.imshow(img/2.0 + 0.5)
    plt.title('occluded image')

    plt.subplot(2, 2, 2)
    plt.imshow(complete_img/2.0 + 0.5)
    plt.axis('off')
    plt.title('original image')

    plt.subplot(2, 2, 3)
    plt.imshow(predict_img/2.0 + 0.5)
    plt.axis('off')
    plt.title('predict image')
    
    plt.subplot(2, 2, 4)
    plt.imshow(full_img/2.0 + 0.5)
    plt.axis('off')
    plt.title('complete image')
    plt.show()
    
def visualize_flow(imgs, img_pred_warped, img_warped, predicted_flow, flow):    
    img1, img2 = imgs[0:3, :, :], imgs[3:6, :, :]
    img1 = img1.detach().cpu().numpy().transpose(1, 2, 0)
    img2 = img2.detach().cpu().numpy().transpose(1, 2, 0)
    img_pred_warped = img_pred_warped.detach().cpu().numpy().transpose(1, 2, 0)
    img_warped = img_warped.detach().cpu().numpy().transpose(1, 2, 0)
    predicted_flow = predicted_flow.detach().cpu().numpy().transpose(1, 2, 0)
    flow = flow.detach().cpu().numpy().transpose(1, 2, 0)
    pred_flow_viz = flow2img(predicted_flow)
    flow_viz = flow2img(flow)

    plt.figure(figsize=(12, 9))
    plt.subplot(3, 2, 1)
    plt.imshow(img1/2.0 + 0.5)
    plt.axis('off')
    plt.title('image 1')

    plt.subplot(3, 2, 2)
    plt.imshow(img2/2.0 + 0.5)
    plt.axis('off')
    plt.title('image 2')

    plt.subplot(3, 2, 3)
    plt.imshow(img_pred_warped/2.0 + 0.5)
    plt.axis('off')
    plt.title('predicted warped image')

    plt.subplot(3, 2, 4)
    plt.imshow(img_warped/2.0 + 0.5)
    plt.axis('off')
    plt.title('warped image')
    
    plt.subplot(3, 2, 5)
    plt.imshow(pred_flow_viz)
    plt.axis('off')
    plt.title('predicted optical flow')
    
    plt.subplot(3, 2, 6)
    plt.imshow(flow_viz)
    plt.axis('off')
    plt.title('optical flow')
    plt.show()
    
def visualize(imgs, img_pred_warped, img_warped, img_occluded, img_completed, pred_flow, flow, occ, img_inpainted):    
    img1, img2 = imgs[0:3, :, :], imgs[3:6, :, :]
    img1 = img1.detach().cpu().numpy().transpose(1, 2, 0)
    img2 = img2.detach().cpu().numpy().transpose(1, 2, 0)
    img_pred_warped = img_pred_warped.detach().cpu().numpy().transpose(1, 2, 0)
    img_warped = img_warped.detach().cpu().numpy().transpose(1, 2, 0)
    img_occluded = img_occluded.detach().cpu().numpy().transpose(1, 2, 0)
    img_completed = img_completed.detach().cpu().numpy().transpose(1, 2, 0)
    pred_flow = pred_flow.detach().cpu().numpy().transpose(1, 2, 0)
    pred_flow_viz = flow2img(pred_flow)
    flow = flow.detach().cpu().numpy().transpose(1, 2, 0)
    flow_viz = flow2img(flow)
    occ = occ.detach().cpu().numpy().transpose(1, 2, 0)
    img_inpainted = img_inpainted.detach().cpu().numpy().transpose(1, 2, 0)
    
    plt.figure(figsize=(12, 15))
    plt.subplot(5, 2, 1)
    plt.imshow(img1/2.0 + 0.5)
    plt.axis('off')
    plt.title('image 1')

    plt.subplot(5, 2, 2)
    plt.imshow(img2/2.0 + 0.5)
    plt.axis('off')
    plt.title('image 2')

    plt.subplot(5, 2, 3)
    plt.imshow(img_pred_warped/2.0 + 0.5)
    plt.axis('off')
    plt.title('predicted warped image')

    plt.subplot(5, 2, 4)
    plt.imshow(img_warped/2.0 + 0.5)
    plt.axis('off')
    plt.title('warped image')
    
    plt.subplot(5, 2, 5)
    plt.imshow(img_occluded/2.0 + 0.5)
    plt.axis('off')
    plt.title('occluded image')
    
    plt.subplot(5, 2, 6)
    plt.imshow(img_completed/2.0 + 0.5)
    plt.axis('off')
    plt.title('completed image')
    
    plt.subplot(5, 2, 7)
    plt.imshow(pred_flow_viz)
    plt.axis('off')
    plt.title('pred flow')
    
    plt.subplot(5, 2, 8)
    plt.imshow(flow_viz)
    plt.axis('off')
    plt.title('flow')
    
    plt.subplot(5, 2, 9)
    plt.imshow(occ, cmap='gray')
    plt.axis('off')
    plt.title('pred occlusion')
    
    plt.subplot(5, 2, 10)
    plt.imshow(img_inpainted/2.0 + 0.5)
    plt.axis('off')
    plt.title('inpainted image')
#     print('epe error for predicted flow:', evaluate_flow(pred_flow, flow))
#     print('photometric error for predicted flow',np.abs(img_pred_warped - img1).mean())
#     print('photometric error for ground truth flow', np.abs(img_warped - img1).mean())
#     print('photometric error for completed image', np.abs(img_completed - img1).mean())
    plt.show()