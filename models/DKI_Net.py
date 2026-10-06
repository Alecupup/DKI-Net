import torch
import torch.nn as nn
from torchvision import models
from torch.nn import functional as F
from utils.misc import initialize_weights
from models.Transformer_Interaction import mit

args = {'hidden_size': 128*3,
        'mlp_dim': 256*3,
        'num_heads': 4,
        'num_layers': 2,
        'dropout_rate': 0.}
        
def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class ECA(nn.Module):
    """轻量通道注意力（ECA）"""
    def __init__(self, channels, k_size=3):
        super(ECA, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)  # [B, C, 1, 1]
        y = y.squeeze(-1).transpose(-1, -2)  # [B, 1, C]
        y = self.conv(y)
        y = self.sigmoid(y)
        y = y.transpose(-1, -2).unsqueeze(-1)  # [B, C, 1, 1]
        return x * y.expand_as(x)


class DBCFE(nn.Module):
    """双尺度DBCFE：high + low 特征融合（低风险细节增强版）"""
    def __init__(self, in_channels=128, low_channels=64, fcat_channels=128, out_channels=256):
        super(DBCFE, self).__init__()
        self.fuse12 = nn.Sequential(
            conv1x1(in_channels * 2, fcat_channels),
            nn.BatchNorm2d(fcat_channels),
            nn.ReLU(inplace=True)
        )
        self.fuse21 = nn.Sequential(
            conv1x1(in_channels * 2, fcat_channels),
            nn.BatchNorm2d(fcat_channels),
            nn.ReLU(inplace=True)
        )
        self.eca = ECA(fcat_channels, k_size=3)
        self.out_conv = nn.Sequential(
            conv1x1(fcat_channels + in_channels, out_channels),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.low_fuse12 = nn.Sequential(
            conv1x1(low_channels * 2, low_channels),
            nn.BatchNorm2d(low_channels),
            nn.ReLU(inplace=True)
        )
        self.low_fuse21 = nn.Sequential(
            conv1x1(low_channels * 2, low_channels),
            nn.BatchNorm2d(low_channels),
            nn.ReLU(inplace=True)
        )
        self.low_eca = ECA(low_channels, k_size=3)
        self.low_out = nn.Sequential(
            conv1x1(low_channels * 2, out_channels),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, f1, f2, f1_low=None, f2_low=None):
        fcat_12 = self.fuse12(torch.cat([f1, f2], dim=1))
        fcat_21 = self.fuse21(torch.cat([f2, f1], dim=1))
        fcat = self.eca(fcat_12 + fcat_21)
        fdiff = torch.abs(f1 - f2)
        fh = self.out_conv(torch.cat([fcat, fdiff], dim=1))

        if f1_low is not None and f2_low is not None:
            lcat_12 = self.low_fuse12(torch.cat([f1_low, f2_low], dim=1))
            lcat_21 = self.low_fuse21(torch.cat([f2_low, f1_low], dim=1))
            lcat = self.low_eca(lcat_12 + lcat_21)
            ldiff = torch.abs(f1_low - f2_low)
            fl = self.low_out(torch.cat([lcat, ldiff], dim=1))
            fl = F.interpolate(fl, size=fh.shape[2:], mode='bilinear', align_corners=False)
            fh = fh + fl

        return fh


class _DecoderBlock(nn.Module):
    def __init__(self, in_channels_high, in_channels_low, out_channels, scale_ratio=1):
        super(_DecoderBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels_high, in_channels_high, kernel_size=2, stride=2)
        in_channels = in_channels_high + in_channels_low//scale_ratio
        self.transit = nn.Sequential(
            conv1x1(in_channels_low, in_channels_low//scale_ratio),
            nn.BatchNorm2d(in_channels_low//scale_ratio),
            nn.ReLU(inplace=True) )
        self.decode = nn.Sequential(
            conv3x3(in_channels, out_channels),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True) )

    def forward(self, x, low_feat):
        x = self.up(x)
        low_feat = self.transit(low_feat)
        x = torch.cat((x, low_feat), dim=1)
        x = self.decode(x)
        return x

class FCN(nn.Module):
    def __init__(self, in_channels=3, pretrained=True):
        super(FCN, self).__init__()
        # Load ImageNet-pretrained ResNet34 weights through torchvision.
        # No local checkpoint is used here.
        try:
            weights = models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
            resnet = models.resnet34(weights=weights)
        except AttributeError:
            # Compatibility with older torchvision versions.
            resnet = models.resnet34(pretrained=pretrained)

        newconv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        try:
            newconv1.weight.data[:, 0:3, :, :].copy_(resnet.conv1.weight.data[:, 0:3, :, :])
            if in_channels > 3:
                newconv1.weight.data[:, 3:in_channels, :, :].copy_(resnet.conv1.weight.data[:, 0:in_channels-3, :, :])
        except Exception:
            nn.init.kaiming_normal_(newconv1.weight, mode='fan_out', nonlinearity='relu')
          
        self.layer0 = nn.Sequential(newconv1, resnet.bn1, resnet.relu)
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        for n, m in self.layer3.named_modules():
            if 'conv1' in n or 'downsample.0' in n:
                m.stride = (1, 1)
        for n, m in self.layer4.named_modules():
            if 'conv1' in n or 'downsample.0' in n:
                m.stride = (1, 1)
        self.head = nn.Sequential(nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0, bias=False),
                                  nn.BatchNorm2d(128), nn.ReLU())
        initialize_weights(self.head)
                                  
    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes:
            downsample = nn.Sequential(
                conv1x1(inplanes, planes, stride),
                nn.BatchNorm2d(planes) )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

class ResBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(ResBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
    
    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class DKI_Net(nn.Module):
    def __init__(self, in_channels=3, num_classes=7, input_size=512):
        super(DKI_Net, self).__init__()
        feat_size = input_size//4
        self.FCN = FCN(in_channels, pretrained=True)
        self.DBCFE = DBCFE(in_channels=128, low_channels=64, fcat_channels=128, out_channels=256)
        self.resCD = self._make_layer(ResBlock, 256, 128, 6, stride=1)
        self.transformer = mit(img_size=feat_size, in_chans=128*3, embed_dim=128*3, num_kernels=16)
        
        self.DecCD = _DecoderBlock(128, 128, 128, scale_ratio=2)
        self.Dec1  = _DecoderBlock(128, 64,  128)
        self.Dec2  = _DecoderBlock(128, 64,  128)
        
        self.classifierA = nn.Conv2d(128, num_classes, kernel_size=1)
        self.classifierB = nn.Conv2d(128, num_classes, kernel_size=1)
        self.classifierCD = nn.Sequential(nn.Conv2d(128, 64, kernel_size=1), nn.BatchNorm2d(64), nn.ReLU(), nn.Conv2d(64, 1, kernel_size=1))
            
        initialize_weights(
            self.Dec1, self.Dec2, self.classifierA, self.classifierB, self.DBCFE, self.resCD, self.DecCD, self.classifierCD
        )
    
    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes:
            downsample = nn.Sequential(
                conv1x1(inplanes, planes, stride),
                nn.BatchNorm2d(planes) )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
    
    def base_forward(self, x):
        
        x = self.FCN.layer0(x) #size:1/2
        x = self.FCN.maxpool(x) #size:1/4
        x_low = self.FCN.layer1(x) #size:1/4
        x = self.FCN.layer2(x_low) #size:1/8
        x = self.FCN.layer3(x)
        x = self.FCN.layer4(x)
        x = self.FCN.head(x)
        return x, x_low
    
    def CD_forward(self, x1, x2, x1_low=None, x2_low=None):
        # 双尺度 DBCFE：高层 + 低层联合补细节
        x = self.DBCFE(x1, x2, x1_low, x2_low)
        xc = self.resCD(x)
        return x1, x2, xc
    
    def forward(self, x1, x2):
        x_size = x1.size()
        
        x1, x1_low = self.base_forward(x1)
        x2, x2_low = self.base_forward(x2)
        x1, x2, xc = self.CD_forward(x1, x2, x1_low, x2_low)

        x1 = self.Dec1(x1, x1_low)
        x2 = self.Dec2(x2, x2_low)        
        xc_low = torch.cat([x1_low, x2_low], 1)
        xc = self.DecCD(xc, xc_low)
                
        x = torch.cat([x1, x2, xc], 1)
        x = self.transformer(x)
        x1 = x[:, 0:128, :, :]
        x2 = x[:, 128:256, :, :]
        xc = x[:, 256:, :, :]
        
        out1 = self.classifierA(x1)
        out2 = self.classifierB(x2)
        change = self.classifierCD(xc)
        
        return F.upsample(change, x_size[2:], mode='bilinear'), F.upsample(out1, x_size[2:], mode='bilinear'), F.upsample(out2, x_size[2:], mode='bilinear')