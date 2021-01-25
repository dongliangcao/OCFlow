from .ssim.ssim import ssim
from .psnr.psnr import psnr
from .fid.fid import calculate_fid_given_imgs
import torch
def calculate_ssim(model, dataloader, device, type = 'simple'):
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

def calculate_fid(model, dataloader, device, type ='simple'): 
    print(device)
    model.to(device)
    model.eval()
    assert type in ['gated', 'simple'], 'Unknown network type'

    with torch.no_grad(): 
        list_imgs = []
        list_complete_imgs = []
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

            list_imgs.append(imgs)
            list_complete_imgs.append(complete_imgs)
    
    imgs = torch.cat(list_imgs, dim = 0).to('cpu') # size [B,3,H,W]
    complete_imgs = torch.cat(list_complete_imgs, dim = 0).to('cpu') #size [B,3,H,W]
    fid_value = calculate_fid_given_imgs(imgs, complete_imgs, 64,cuda = False if device == 'cpu' else True,dims = 2048)
    model.to('cpu')
    return fid_value
def calculate_psnr(model, dataloader, device, type ='simple'): 
    psnr_value = 0
    num = 1
    
    print(device)
    model.to(device)
    model.eval()
    assert type in ['gated', 'simple'], 'Unknown network type'

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
            for img, complete_img in zip(imgs, complete_imgs): 
                psnr_value = psnr_value + psnr((img.detach().cpu().numpy()+1)*127.5, (complete_img.detach().cpu().numpy()+1)*127.5)
                num = num +1
        print(img.shape)
    model.to('cpu')
    return psnr_value/num