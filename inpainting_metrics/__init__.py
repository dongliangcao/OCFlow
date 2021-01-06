from .ssim.ssim import ssim
from .psnr.psnr import psnr
import torch
def calculate_ssim(model, dataloader, device, type = 'gated'):
    print(device)
    model.to(device)
    model.eval()
    assert type in ['gated', 'simple'], 'Unknown network type'
    ssim_score = 0
    total = 0
    with torch.no_grad(): 
        for batch in dataloader: 
            _, imgs, masks = batch
            imgs = imgs.to(device)
            masks = masks.to(device)
            batch_size = imgs.size(0) 
            if type =='gated': 
                coarse_imgs, recon_imgs = model(imgs,masks)
            else: 
                recon_imgs = model(imgs, masks)
            complete_imgs = recon_imgs * masks + imgs * (1 - masks)
            ssim_score = ssim_score + batch_size * ssim(imgs, complete_imgs, window_size = 4, size_average = True)
            total = total + batch_size
    model.to('cpu')
    return (ssim_score/total).cpu().item()
def calculate_psnr(model, dataloader): 
    psnr_value = 0
    num = 0
    for batch in dataloader: 
        _, imgs, masks = batch
        batch_size = imgs.size(0) 
        coarse_imgs, recon_imgs = model(imgs,masks)
        complete_imgs = recon_imgs * masks + imgs * (1 - masks)
        for img, complete_img in zip(imgs, complete_imgs): 
            psnr_value = psnr_value + psnr(img.detach().cpu().numpy(), complete_img.detach().cpu().numpy())
            num = num +1
    return psnr_value/num