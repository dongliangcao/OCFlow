import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

class InitialBlock(nn.Module):
    """
    Initial block downsample to input with max pooling and convolution with stride=2
    Input size: [N, C, H, W]
    Output size: [N, 16, ceil(H/2), ceil(W/2)]
    """
    def __init__(self, in_channels):
        super(InitialBlock, self).__init__()
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.conv = nn.Conv2d(in_channels, 16 - in_channels, 3, padding=1, stride=2)
        self.bn = nn.BatchNorm2d(16)
        self.prelu = nn.PReLU(16)
        
    def forward(self, x):
        x = torch.cat((self.max_pool(x), self.conv(x)), dim=1)
        x = self.bn(x)
        x = self.prelu(x)
        return x
    
class BottleNeck(nn.Module):
    """
    Basic block in the ENet
    Main branch + Residual block
    If downsample:
        Main branch: max pooling
        Residual block: 2x2 conv with stride 2 + 3x3 conv + 1x1 conv
    If upsample:
        Main branch: max unpooling
        Residual block: 1x1 conv + 3x3 transposed conv + 1x1 conv
    If original:
        Main branch: identity
        Residual block: 1x1 conv + 3x3 conv/asymmetric 5x5 conv + 1x1 conv
    """
    def __init__(self, in_channels, out_channels=None,
                dilation=1, downsample=False, proj_ratio=4,
                upsample=False, asymmetric=False, regularize=True,
                p_drop=None, use_prelu=True):
        super(BottleNeck, self).__init__()
        
        self.pad = 0
        self.upsample = upsample
        self.downsample = downsample
        if not out_channels:
            out_channels = in_channels
        else:
            self.pad = out_channels - in_channels
            
        if regularize:
            assert p_drop is not None
        if downsample:
            assert not upsample
        if upsample:
            assert not downsample
            
        inter_channels = in_channels // proj_ratio
        
        # Main branch
        if upsample:
            self.spatil_conv = nn.Conv2d(in_channels, out_channels, 1, bias=False)
            self.bn_up = nn.BatchNorm2d(out_channels)
            self.unpool = nn.MaxUnpool2d(kernel_size=2, stride=2)
        elif downsample:
            # the indices are used for unpooling
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        
        # Bottleneck
        # first convolution layer, reduce the dimensionality 
        if downsample:
            # downsample with stride=2
            self.conv1 = nn.Conv2d(in_channels, inter_channels, 2, stride=2, bias=False)
        else:
            # 1x1 conv
            self.conv1 = nn.Conv2d(in_channels, inter_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(inter_channels)
        self.prelu1 = nn.PReLU() if use_prelu else nn.ReLU(inplace=True)
        
        # second convolution layer (main conv)
        if asymmetric:
            # first 1x5 kernel size, then 5x1 kernel size
            self.conv2 = nn.Sequential(
                nn.Conv2d(inter_channels, inter_channels, kernel_size=(1, 5), padding=(0, 2)),
                nn.BatchNorm2d(inter_channels),
                nn.PReLU(),
                nn.Conv2d(inter_channels, inter_channels, kernel_size=(5, 1), padding=(2, 0))
            )
        elif upsample:
            # upsample
            self.conv2 = nn.ConvTranspose2d(inter_channels, inter_channels, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False)
        else:
            self.conv2 = nn.Conv2d(inter_channels, inter_channels, 3, padding=dilation, dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm2d(inter_channels)
        self.prelu2 = nn.PReLU() if use_prelu else nn.ReLU(inplace=True)
        
        # third convolution layer, increase dimensionality to out_channels
        # 1x1 conv
        self.conv3 = nn.Conv2d(inter_channels, out_channels, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.prelu3 = nn.PReLU() if use_prelu else nn.ReLU(inplace=True)
        
        self.regularizer = nn.Dropout2d(p_drop) if regularize else None
        self.prelu_out = nn.PReLU() if use_prelu else nn.ReLU(inplace=True)
        
    def forward(self, x, indices=None, output_size=None):
        # Main branch
        identity = x
        if self.upsample:
            assert (indices is not None) and (output_size is not None)
            identity = self.bn_up(self.spatil_conv(identity))
            if identity.size() != indices.size():
                pad = (indices.size(3) - identity.size(3), 0, indices.size(2) - identity.size(2), 0)
                identity = F.pad(identity, pad, "constant", 0)
            identity = self.unpool(identity, indices=indices)
        elif self.downsample:
            identity, idx = self.pool(identity)
            
        # zero padding for extra channels
        if self.pad > 0:
            extras = torch.zeros((identity.size(0), self.pad, identity.size(2), identity.size(3)))
            if identity.is_cuda:
                extras = extras.cuda()
            identity = torch.cat((identity, extras), dim=1)
            
        # Bottleneck
        x = self.prelu1(self.bn1(self.conv1(x))) # first conv
        x = self.prelu2(self.bn2(self.conv2(x))) # second conv
        x = self.prelu3(self.bn3(self.conv3(x))) # third conv
        if self.regularizer:
            x = self.regularizer(x)
            
        # When the input dim is odd, we might have a mismatch of one pixel
        if identity.size() != x.size():
            pad = (identity.size(3) - x.size(3), 0, identity.size(2) - x.size(2), 0)
            x = F.pad(x, pad, "constant", 0)

        x += identity
        x = self.prelu_out(x)
        
        if self.downsample:
            return x, idx
        return x
    
class EFlowNet(nn.Module):
    def __init__(self, in_channels=6):
        super(EFlowNet, self).__init__()
        self.initial = InitialBlock(in_channels)
        
        # Encoder part
        # Stage 1
        self.bottleneck10 = BottleNeck(16, 64, downsample=True, p_drop=0.01)
        self.bottleneck11 = BottleNeck(64, p_drop=0.01)
        self.bottleneck12 = BottleNeck(64, p_drop=0.01)
        self.bottleneck13 = BottleNeck(64, p_drop=0.01)
        self.bottleneck14 = BottleNeck(64, p_drop=0.01)
        
        # Stage 2
        self.bottleneck20 = BottleNeck(64, 128, downsample=True, p_drop=0.1)
        self.bottleneck21 = BottleNeck(128, p_drop=0.1)
        self.bottleneck22 = BottleNeck(128, dilation=2, p_drop=0.1)
        self.bottleneck23 = BottleNeck(128, asymmetric=True, p_drop=0.1)
        self.bottleneck24 = BottleNeck(128, dilation=4, p_drop=0.1)
        self.bottleneck25 = BottleNeck(128, p_drop=0.1)
        self.bottleneck26 = BottleNeck(128, dilation=8, p_drop=0.1)
        self.bottleneck27 = BottleNeck(128, asymmetric=True, p_drop=0.1)
        self.bottleneck28 = BottleNeck(128, dilation=16, p_drop=0.1)
        
        # Stage 3, repeat Stage 2 without downsample
        self.bottleneck31 = BottleNeck(128, p_drop=0.1)
        self.bottleneck32 = BottleNeck(128, dilation=2, p_drop=0.1)
        self.bottleneck33 = BottleNeck(128, asymmetric=True, p_drop=0.1)
        self.bottleneck34 = BottleNeck(128, dilation=4, p_drop=0.1)
        self.bottleneck35 = BottleNeck(128, p_drop=0.1)
        self.bottleneck36 = BottleNeck(128, dilation=8, p_drop=0.1)
        self.bottleneck37 = BottleNeck(128, asymmetric=True, p_drop=0.1)
        self.bottleneck38 = BottleNeck(128, dilation=16, p_drop=0.1)
        
        # Decoder part
        # Stage 4
        self.bottleneck40 = BottleNeck(128, 64, upsample=True, p_drop=0.1, use_prelu=False)
        self.bottleneck41 = BottleNeck(64, p_drop=0.1, use_prelu=False)
        self.bottleneck42 = BottleNeck(64, p_drop=0.1, use_prelu=False)
        
        # Stage 5
        self.bottleneck50 = BottleNeck(64, 16, upsample=True, p_drop=0.1, use_prelu=False)
        self.bottleneck51 = BottleNeck(16, p_drop=0.1, use_prelu=False)

        # Stage 6
        self.predict_flow = predict_flow(16)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    init.uniform_(m.bias)
                init.xavier_uniform_(m.weight)
                
    def forward(self, x):
        x = self.initial(x)
        # Encoder part
        # Stage 1
        sz1 = x.size()
        x, indices1 = self.bottleneck10(x)
        x = self.bottleneck11(x)
        x = self.bottleneck12(x)
        x = self.bottleneck13(x)
        x = self.bottleneck14(x)

        # Stage 2
        sz2 = x.size()
        x, indices2 = self.bottleneck20(x)
        x = self.bottleneck21(x)
        x = self.bottleneck22(x)
        x = self.bottleneck23(x)
        x = self.bottleneck24(x)
        x = self.bottleneck25(x)
        x = self.bottleneck26(x)
        x = self.bottleneck27(x)
        x = self.bottleneck28(x)

        # Stage 3
        x = self.bottleneck31(x)
        x = self.bottleneck32(x)
        x = self.bottleneck33(x)
        x = self.bottleneck34(x)
        x = self.bottleneck35(x)
        x = self.bottleneck36(x)
        x = self.bottleneck37(x)
        x = self.bottleneck38(x)

        # Decoder part
        # Stage 4
        x = self.bottleneck40(x, indices=indices2, output_size=sz2)
        x = self.bottleneck41(x)
        x = self.bottleneck42(x)

        # Stage 5
        x = self.bottleneck50(x, indices=indices1, output_size=sz1)
        x = self.bottleneck51(x)

        # Stage 6
        flow = self.predict_flow(x)
        return self.upsample(flow)
    
class EFlowNet2(nn.Module):
    """
    Predict optical flow at different resolution, compared with EFlowNet
    """
    def __init__(self, in_channels=6):
        super(EFlowNet2, self).__init__()
        self.initial = InitialBlock(in_channels)
        
        # Encoder part
        # Stage 1
        self.bottleneck10 = BottleNeck(16, 64, downsample=True, p_drop=0.01)
        self.bottleneck11 = BottleNeck(64, p_drop=0.01)
        self.bottleneck12 = BottleNeck(64, p_drop=0.01)
        self.bottleneck13 = BottleNeck(64, p_drop=0.01)
        self.bottleneck14 = BottleNeck(64, p_drop=0.01)
        
        # Stage 2
        self.bottleneck20 = BottleNeck(64, 128, downsample=True, p_drop=0.1)
        self.bottleneck21 = BottleNeck(128, p_drop=0.1)
        self.bottleneck22 = BottleNeck(128, dilation=2, p_drop=0.1)
        self.bottleneck23 = BottleNeck(128, asymmetric=True, p_drop=0.1)
        self.bottleneck24 = BottleNeck(128, dilation=4, p_drop=0.1)
        self.bottleneck25 = BottleNeck(128, p_drop=0.1)
        self.bottleneck26 = BottleNeck(128, dilation=8, p_drop=0.1)
        self.bottleneck27 = BottleNeck(128, asymmetric=True, p_drop=0.1)
        self.bottleneck28 = BottleNeck(128, dilation=16, p_drop=0.1)
        
        # Stage 3, repeat Stage 2 without downsample
        self.bottleneck31 = BottleNeck(128, p_drop=0.1)
        self.bottleneck32 = BottleNeck(128, dilation=2, p_drop=0.1)
        self.bottleneck33 = BottleNeck(128, asymmetric=True, p_drop=0.1)
        self.bottleneck34 = BottleNeck(128, dilation=4, p_drop=0.1)
        self.bottleneck35 = BottleNeck(128, p_drop=0.1)
        self.bottleneck36 = BottleNeck(128, dilation=8, p_drop=0.1)
        self.bottleneck37 = BottleNeck(128, asymmetric=True, p_drop=0.1)
        self.bottleneck38 = BottleNeck(128, dilation=16, p_drop=0.1)
        
        # Decoder part
        # Stage 4
        self.bottleneck40 = BottleNeck(128 + 2, 64, upsample=True, p_drop=0.1, use_prelu=False)
        self.bottleneck41 = BottleNeck(64, p_drop=0.1, use_prelu=False)
        self.bottleneck42 = BottleNeck(64, p_drop=0.1, use_prelu=False)
        
        # Stage 5
        self.bottleneck50 = BottleNeck(64 + 2, 16, upsample=True, p_drop=0.1, use_prelu=False)
        self.bottleneck51 = BottleNeck(16, p_drop=0.1, use_prelu=False)

        # flow prediction
        self.predict_flow3 = predict_flow(128)
        self.predict_flow4 = predict_flow(64)
        self.predict_flow5 = predict_flow(16)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.bias is not None:
                    init.uniform_(m.bias)
                init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    init.uniform_(m.bias)
                init.xavier_uniform_(m.weight)
                
    def forward(self, x):
        x = self.initial(x)
        # Encoder part
        # Stage 1
        sz1 = x.size()
        x, indices1 = self.bottleneck10(x)
        x = self.bottleneck11(x)
        x = self.bottleneck12(x)
        x = self.bottleneck13(x)
        x = self.bottleneck14(x)

        # Stage 2
        sz2 = x.size()
        x, indices2 = self.bottleneck20(x)
        x = self.bottleneck21(x)
        x = self.bottleneck22(x)
        x = self.bottleneck23(x)
        x = self.bottleneck24(x)
        x = self.bottleneck25(x)
        x = self.bottleneck26(x)
        x = self.bottleneck27(x)
        x = self.bottleneck28(x)

        # Stage 3
        x = self.bottleneck31(x)
        x = self.bottleneck32(x)
        x = self.bottleneck33(x)
        x = self.bottleneck34(x)
        x = self.bottleneck35(x)
        x = self.bottleneck36(x)
        x = self.bottleneck37(x)
        x = self.bottleneck38(x)
        flow3 = self.predict_flow3(x)
        
        # Decoder part
        # Stage 4
        x = torch.cat((x, flow3), dim=1)
        x = self.bottleneck40(x, indices=indices2, output_size=sz2)
        x = self.bottleneck41(x)
        x = self.bottleneck42(x)
        flow4 = self.predict_flow4(x)

        # Stage 5
        x = torch.cat((x, flow4), dim=1)
        x = self.bottleneck50(x, indices=indices1, output_size=sz1)
        x = self.bottleneck51(x)

        flow = self.predict_flow5(x)
        return self.upsample(flow)
    
def predict_flow(in_channels):
    return nn.Conv2d(in_channels,2,kernel_size=3,stride=1,padding=1,bias=True)