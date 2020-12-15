import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import init 
    
import numpy as np

def get_pad(in_,  ksize, stride, dilation=1):
    if isinstance(in_, (list, tuple)):
        in_h, in_w = in_[0], in_[1]
        out_h = np.ceil(float(in_h)/stride)
        out_h = int(((out_h - 1) * stride + dilation*(ksize-1) + 1 - in_h)/2)
        out_w = np.ceil(float(in_w)/stride)
        out_w = int(((out_w - 1) * stride + dilation*(ksize-1) + 1 - in_w)/2)
        return (out_h, out_w)
    out_ = np.ceil(float(in_)/stride)
    return int(((out_ - 1) * stride + dilation*(ksize-1) + 1 - in_)/2)

class Conv2dWithProj(nn.Module):
    """
    Covolution layer with 1x1 conv projection 
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=False, proj_ratio=4, spectral_norm=False):
        super(Conv2dWithProj, self).__init__()
        inter_channels = max(in_channels // proj_ratio, 1)
        self.conv1 = nn.Conv2d(in_channels, inter_channels, 1, bias=bias)
        self.conv2 = nn.Conv2d(inter_channels, inter_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias)
        self.conv3 = nn.Conv2d(inter_channels, out_channels, 1, bias=bias)
        if spectral_norm:
            self.conv1 = nn.utils.spectral_norm(self.conv1)
            self.conv2 = nn.utils.spectral_norm(self.conv2)
            self.conv3 = nn.utils.spectral_norm(self.conv3)
    def forward(self, x):
        x = self.conv3(self.conv2(self.conv1(x)))
        return x
    
class GatedConv2dWithActivation(nn.Module):
    """
    Gated Convlution layer with activation (default activation:LeakyReLU)
    Params: same as conv2d
    Input: The feature from last layer "I"
    Output:\phi(f(I))*\sigmoid(g(I))
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=False, batch_norm=True, activation=torch.nn.LeakyReLU(0.2, inplace=True), proj_ratio=4):
        super(GatedConv2dWithActivation, self).__init__()
        self.activation = activation
        
        self.conv2d = Conv2dWithProj(in_channels, out_channels, kernel_size, stride, padding, dilation, bias, proj_ratio)
        self.mask_conv2d = Conv2dWithProj(in_channels, out_channels, kernel_size, stride, padding, dilation, bias, proj_ratio)
        
        if batch_norm:
            self.batch_norm = nn.BatchNorm2d(out_channels)
        
        self.sigmoid = nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                
    def gated(self, mask):
        return self.sigmoid(mask)
    
    def forward(self, input):
        x = self.conv2d(input)
        mask = self.mask_conv2d(input)
        if self.activation is not None:
            x = self.activation(x) * self.gated(mask)
        else:
            x = x * self.gated(mask)
        if self.batch_norm is not None:
            return self.batch_norm(x)
        else:
            return x
        
class GatedDeConv2dWithActivation(torch.nn.Module):
    """
    Gated DeConvlution layer with activation (default activation:LeakyReLU)
    resize + conv
    Params: same as conv2d
    Input: The feature from last layer "I"
    Output:\phi(f(I))*\sigmoid(g(I))
    """
    def __init__(self, scale_factor, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True, batch_norm=True, activation=torch.nn.LeakyReLU(0.2, inplace=True), proj_ratio=4):
        super(GatedDeConv2dWithActivation, self).__init__()
        self.conv2d = GatedConv2dWithActivation(in_channels, out_channels, kernel_size, stride, padding, dilation, bias, batch_norm, activation, proj_ratio)
        self.scale_factor = scale_factor

    def forward(self, input):
        x = F.interpolate(input, scale_factor=2)
        return self.conv2d(x)
    
class SNGatedConv2dWithActivation(torch.nn.Module):
    """
    Gated Convolution with spetral normalization
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True, batch_norm=True, activation=torch.nn.LeakyReLU(0.2, inplace=True), proj_ratio=4):
        super(SNGatedConv2dWithActivation, self).__init__()
        self.activation = activation
        
        self.conv2d = Conv2dWithProj(in_channels, out_channels, kernel_size, stride, padding, dilation, bias, proj_ratio, spectral_norm=True)
        self.mask_conv2d = Conv2dWithProj(in_channels, out_channels, kernel_size, stride, padding, dilation, bias, proj_ratio, spectral_norm=True)
        
        if batch_norm:
            self.batch_norm = nn.BatchNorm2d(out_channels)
        self.sigmoid = nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)

    def gated(self, mask):
        return self.sigmoid(mask)

    def forward(self, input):
        x = self.conv2d(input)
        mask = self.mask_conv2d(input)
        if self.activation is not None:
            x = self.activation(x) * self.gated(mask)
        else:
            x = x * self.gated(mask)
        if self.batch_norm is not None:
            return self.batch_norm(x)
        else:
            return x
        
class SNGatedDeConv2dWithActivation(torch.nn.Module):
    """
    Gated DeConvlution layer with activation (default activation:LeakyReLU)
    resize + conv
    Params: same as conv2d
    Input: The feature from last layer "I"
    Output:\phi(f(I))*\sigmoid(g(I))
    """
    def __init__(self, scale_factor, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True, batch_norm=True, activation=torch.nn.LeakyReLU(0.2, inplace=True), proj_ratio=4):
        super(SNGatedDeConv2dWithActivation, self).__init__()
        self.conv2d = SNGatedConv2dWithActivation(in_channels, out_channels, kernel_size, stride, padding, dilation, bias, batch_norm, activation, proj_ratio)
        self.scale_factor = scale_factor

    def forward(self, input):
        x = F.interpolate(input, scale_factor=2)
        return self.conv2d(x)
    
class SNConvWithActivation(torch.nn.Module):
    """
    SN convolution for spetral normalization conv
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True, activation=torch.nn.LeakyReLU(0.2, inplace=True), proj_ratio=4):
        super(SNConvWithActivation, self).__init__()
        self.conv2d = Conv2dWithProj(in_channels, out_channels, kernel_size, stride, padding, dilation, bias, proj_ratio, spectral_norm=True)
        self.activation = activation
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
    def forward(self, input):
        x = self.conv2d(input)
        if self.activation is not None:
            return self.activation(x)
        else:
            return x
        
class Self_Attn(nn.Module):
    """Self attention layer"""
    def __init__(self, in_dim, with_attn=False):
        super(Self_Attn, self).__init__()
        self.in_channels = in_dim
        self.with_attn = with_attn
        self.query_conv = nn.Conv2d(in_dim, in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_dim, in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_dim, in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, x):
        """
            args :
                x : input feature maps (B x C x W x H)
            returns :
                out : self attention value + input feature
                attention: B x N x N (N is WxH)
        """
        B, C, W, H = x.size()
        proj_query = self.query_conv(x).view(B, -1, W*H).permute(0, 2, 1) # B x N x C
        proj_key = self.key_conv(x).view(B, -1, W*H) # B x C x N
        attention = self.softmax(torch.bmm(proj_query, proj_key)) # B x N x N
        proj_value = self.value_conv(x).view(B, -1, W*H) # B x C x N
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(B, C, W, H)
        
        out = self.gamma * out + x
        if self.with_attn:
            return out, attention
        else:
            return out
        
class InpaintSANet(torch.nn.Module):
    """
    Inpaint generator, input should be 4*256*256, where 3*256*256 is the masked image, 1*256*256 for mask
    """
    def __init__(self, n_in_channel=4, img_size=(64, 128)):
        super(InpaintSANet, self).__init__()
        cnum = 32
        h, w = img_size[0], img_size[1]
        # coarse prediction network
        self.coarse_net = nn.Sequential(
            #input is 4*H*W, but it is full convolution network, so it can be larger than 256
            GatedConv2dWithActivation(n_in_channel, cnum, 5, 1, padding=get_pad(img_size, 5, 1), proj_ratio=1),
            # downsample
            GatedConv2dWithActivation(cnum, 2*cnum, 4, 2, padding=get_pad(img_size, 4, 2)),
            GatedConv2dWithActivation(2*cnum, 2*cnum, 3, 1, padding=get_pad((h//2, w//2), 3, 1)),
            #downsample
            GatedConv2dWithActivation(2*cnum, 4*cnum, 4, 2, padding=get_pad((h//2, w//2), 4, 2)),
            GatedConv2dWithActivation(4*cnum, 4*cnum, 3, 1, padding=get_pad((h//4, w//4), 3, 1)),
            GatedConv2dWithActivation(4*cnum, 4*cnum, 3, 1, padding=get_pad((h//4, w//4), 3, 1)),
            # dilated convlution
            GatedConv2dWithActivation(4*cnum, 4*cnum, 3, 1, dilation=2, padding=get_pad((h//4, w//4), 3, 1, 2)),
            GatedConv2dWithActivation(4*cnum, 4*cnum, 3, 1, dilation=4, padding=get_pad((h//4, w//4), 3, 1, 4)),
            GatedConv2dWithActivation(4*cnum, 4*cnum, 3, 1, dilation=8, padding=get_pad((h//4, w//4), 3, 1, 8)),
            GatedConv2dWithActivation(4*cnum, 4*cnum, 3, 1, dilation=16, padding=get_pad((h//4, w//4), 3, 1, 16)),
            GatedConv2dWithActivation(4*cnum, 4*cnum, 3, 1, padding=get_pad((h//4, w//4), 3, 1)),
            #Self_Attn(4*cnum),
            GatedConv2dWithActivation(4*cnum, 4*cnum, 3, 1, padding=get_pad((h//4, w//4), 3, 1)),
            # upsample
            GatedDeConv2dWithActivation(2, 4*cnum, 2*cnum, 3, 1, padding=get_pad((h//2, w//2), 3, 1)),
            #Self_Attn(2*cnum, 'relu'),
            GatedConv2dWithActivation(2*cnum, 2*cnum, 3, 1, padding=get_pad((h//2, w//2), 3, 1)),
            GatedDeConv2dWithActivation(2, 2*cnum, cnum, 3, 1, padding=get_pad(img_size, 3, 1)),

            GatedConv2dWithActivation(cnum, cnum//2, 3, 1, padding=get_pad(img_size, 3, 1)),
            #Self_Attn(cnum//2),
            GatedConv2dWithActivation(cnum//2, 3, 3, 1, padding=get_pad(img_size, 3, 1), activation=nn.Tanh())
        )
        # refine prediction network
        self.refine_conv_net = nn.Sequential(
            # input is 4xHxW
            GatedConv2dWithActivation(n_in_channel, cnum, 5, 1, padding=get_pad(img_size, 5, 1), proj_ratio=1),
            # downsample
            GatedConv2dWithActivation(cnum, cnum, 4, 2, padding=get_pad(img_size, 4, 2)),
            GatedConv2dWithActivation(cnum, 2*cnum, 3, 1, padding=get_pad((h//2, w//2), 3, 1)),
            # downsample
            GatedConv2dWithActivation(2*cnum, 2*cnum, 4, 2, padding=get_pad((h//2, w//2), 4, 2)),
            GatedConv2dWithActivation(2*cnum, 4*cnum, 3, 1, padding=get_pad((h//4, w//4), 3, 1)),
            GatedConv2dWithActivation(4*cnum, 4*cnum, 3, 1, padding=get_pad((h//4, w//4), 3, 1)),
            GatedConv2dWithActivation(4*cnum, 4*cnum, 3, 1, padding=get_pad((h//4, w//4), 3, 1)),
            GatedConv2dWithActivation(4*cnum, 4*cnum, 3, 1, dilation=2, padding=get_pad((h//4, w//4), 3, 1, 2)),
            GatedConv2dWithActivation(4*cnum, 4*cnum, 3, 1, dilation=4, padding=get_pad((h//4, w//4), 3, 1, 4)),
            #Self_Attn(4*cnum),
            GatedConv2dWithActivation(4*cnum, 4*cnum, 3, 1, dilation=8, padding=get_pad((h//4, w//4), 3, 1, 8)),

            GatedConv2dWithActivation(4*cnum, 4*cnum, 3, 1, dilation=16, padding=get_pad((h//4, w//4), 3, 1, 16))
        )
        # self-attention network
        self.refine_attn = Self_Attn(4*cnum, with_attn=False)
        # refine upsample network
        self.refine_upsample_net = nn.Sequential(
            GatedConv2dWithActivation(4*cnum, 4*cnum, 3, 1, padding=get_pad((h//4, w//4), 3, 1)),

            GatedConv2dWithActivation(4*cnum, 4*cnum, 3, 1, padding=get_pad((h//4, w//4), 3, 1)),
            GatedDeConv2dWithActivation(2, 4*cnum, 2*cnum, 3, 1, padding=get_pad((h//2, w//2), 3, 1)),
            GatedConv2dWithActivation(2*cnum, 2*cnum, 3, 1, padding=get_pad((h//2, w//2), 3, 1)),
            GatedDeConv2dWithActivation(2, 2*cnum, cnum, 3, 1, padding=get_pad(img_size, 3, 1)),

            GatedConv2dWithActivation(cnum, cnum//2, 3, 1, padding=get_pad(img_size, 3, 1)),
            #Self_Attn(cnum, 'relu'),
            GatedConv2dWithActivation(cnum//2, 3, 3, 1, padding=get_pad(img_size, 3, 1), activation=nn.Tanh()),
        )


    def forward(self, imgs, masks):
        # Coarse
        masked_imgs =  imgs * (1 - masks)
        input_imgs = torch.cat([masked_imgs, masks], dim=1)
        
        x = self.coarse_net(input_imgs)
#         x = torch.clamp(x, -1., 1.)
        coarse_x = x
        
        # Refine
        masked_imgs = imgs * (1 - masks) + coarse_x * masks
        input_imgs = torch.cat([masked_imgs, masks], dim=1)
        x = self.refine_conv_net(input_imgs)
        x = self.refine_attn(x)
        x = self.refine_upsample_net(x)
#         x = torch.clamp(x, -1., 1.)
        return x
    
class InpaintSADirciminator(nn.Module):
    def __init__(self, n_in_channel=4, img_size=(64, 128)):
        super(InpaintSADirciminator, self).__init__()
        cnum = 32
        h, w = img_size[0], img_size[1]
        self.discriminator_net = nn.Sequential(
            SNConvWithActivation(n_in_channel, 2*cnum, 4, 2, padding=get_pad(img_size, 5, 2)),
            SNConvWithActivation(2*cnum, 4*cnum, 4, 2, padding=get_pad((h//2, w//2), 5, 2)),
            SNConvWithActivation(4*cnum, 8*cnum, 4, 2, padding=get_pad((h//4, w//4), 5, 2)),
            SNConvWithActivation(8*cnum, 8*cnum, 4, 2, padding=get_pad((h//8, w//8), 5, 2)),
            SNConvWithActivation(8*cnum, 8*cnum, 4, 2, padding=get_pad((h//16, w//16), 5, 2)),
#             SNConvWithActivation(8*cnum, 8*cnum, 4, 2, padding=get_pad((h//32, w//32), 5, 2)),
#             Self_Attn(8*cnum, 'relu'),
#             SNConvWithActivation(8*cnum, 8*cnum, 4, 2, padding=get_pad(4, 5, 2)),
        )

    def forward(self, input):
        x = self.discriminator_net(input)
        x = x.view((x.size(0),-1))
        return x

class SNDisLoss(nn.Module):
    """
    The loss for sngan discriminator
    """
    def __init__(self, weight=1):
        super(SNDisLoss, self).__init__()
        self.weight = weight

    def forward(self, pos, neg):
        #return self.weight * (torch.sum(F.relu(-1+pos)) + torch.sum(F.relu(-1-neg)))/pos.size(0)
        return self.weight * (torch.mean(F.relu(1.-pos)) + torch.mean(F.relu(1.+neg)))


class SNGenLoss(nn.Module):
    """
    The loss for sngan generator
    """
    def __init__(self, weight=1):
        super(SNGenLoss, self).__init__()
        self.weight = weight

    def forward(self, neg):
        return - self.weight * torch.mean(neg)    
    
class ReconLoss(nn.Module):
    """
    Reconstruction loss contain l1 loss, may contain perceptual loss

    """
    def __init__(self, chole_alpha, cunhole_alpha, rhole_alpha, runhole_alpha):
        super(ReconLoss, self).__init__()
        self.chole_alpha = chole_alpha
        self.cunhole_alpha = cunhole_alpha
        self.rhole_alpha = rhole_alpha
        self.runhole_alpha = runhole_alpha

    def forward(self, imgs, coarse_imgs, recon_imgs, masks):
        masks_viewed = masks.view(masks.size(0), -1)
        return self.rhole_alpha*torch.mean(torch.abs(imgs - recon_imgs) * masks / masks_viewed.mean(1).view(-1,1,1,1))  + \
                self.runhole_alpha*torch.mean(torch.abs(imgs - recon_imgs) * (1. - masks) / (1. - masks_viewed.mean(1).view(-1,1,1,1)))  + \
                self.chole_alpha*torch.mean(torch.abs(imgs - coarse_imgs) * masks / masks_viewed.mean(1).view(-1,1,1,1))   + \
                self.cunhole_alpha*torch.mean(torch.abs(imgs - coarse_imgs) * (1. - masks) / (1. - masks_viewed.mean(1).view(-1,1,1,1)))